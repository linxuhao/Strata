// src/prefill/gdn_tc.cuh - the chunked DeltaNet recurrence's scan on the tensor cores (STRATA_GDN_TC=1), sm_80+.
//
// Included inside the anonymous namespace of src/prefill/kernels.cu and of src/prefill/gdn_rec_parity.cu, after
// gdn_chunk_scan_kernel (the harness runs the engine's kernels verbatim: one text, included twice).  It uses S, HK, HV,
// C, VPK, GCH, gdn_cp16 / gdn_cp_commit / gdn_cp_wait_all and the prep's scratch layout (tm, pm, gm) from there and
// reads no environment.
//
// gdn_chunk_scan_kernel's chunk algebra (kernels.cu, "STRATA_GDN_CHUNKED=1"), every product transposed so that a
// warp's slice of the state is the mma A operand and never leaves its registers: a block per value head, warp w owns
// value columns 16w..16w+15 of S^T (16 columns x 128 rows = 16 m16n8 f32 accumulator tiles, 64 registers a lane) and
// all 32 tokens of a chunk, so not one product couples two warps:
//   X^T  (16c x 32i) = S0^T . K^T        f16x3, k = state row (8 k16-steps), n = token (4 tiles)
//   QS^T (16c x 32t) = S0^T . Q^T        the same A fragments
//   Y^T  (16c x 32i) = beta_i (V^T - e_i X^T)                                     elementwise on the accumulators
//   D^T  (16c x 32i) = Y^T . T^T         tf32x3 m16n8k8, k = token i' (4 steps), n = token i: T is zero above its
//                                        diagonal, so the tiles with k-step > n-tile are skipped (10 of 16)
//   o^T  (16c x 32t) = e_t QS^T + D^T . P^T   f16x3, k = i (2 steps), n = t: P zero above its diagonal (6 of 8 tiles)
//   Dd^T (16c x 32i) = f_i D^T,  f_i = exp(gL - gamma_i)
//   S^T  (16c x 128r) = eL S0^T + Dd^T . K    f16x3, k = token (2 steps), n = state row (16 tiles)
// 336 mma a warp a chunk; the only shared data are the staged inputs (k and q as f16 hi / lo images, v, T, P, the
// per-token scalars), one barrier a chunk.
//
// Numerics (the harness holds the result within 1e-4 of gdn_rec_kh_kernel, as the FP32 chunked pair): an f16 operand
// is split x = hi + lo, hi = f16(x), lo = f16(x - hi), and a product takes hi.hi + hi.lo + lo.hi, dropping lo.lo: with
// both parts normal that is within 3 x 2^-22 |a||b| of the exact product, the 3xTF32 bound, at half the tensor time.
// For the low part to be normal the operand's largest entry must sit in [2^14, 2^15), which a power of two does
// exactly: k and q (unit vectors after gdn_l2_kernel) carry 2^8, P (|P| <= 1) 2^8, the state 2^es from a warp max
// per chunk, D^T 2^ed likewise; the powers are taken out of the per-token scalars and of the accumulator before and
// after the mma (one FMUL an element, exact).  T = (I + A)^-1 has no a-priori bound and Y carries v's magnitude, so
// D^T = Y^T T^T runs on tf32 (the f32 exponent range), split the same way with cvt.rna.tf32.
#if !defined(__HIPCC__)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_GDN_TC_SM80 1
#else
#define STRATA_GDN_TC_SM80 0   // device code for older cards: the wrappers trap (gdn_rec_tc never launches there)
#endif

constexpr int GTC_KP = S + 8;     // f16 k / q rows: a 272-B stride puts the 8 rows of a ldmatrix 8x8 on 8 distinct 16-B lanes
constexpr int GTC_VP = S + 4;     // f32 v rows: lane (gid, tig) reads word 264 tig + gid, 32 distinct banks
constexpr int GTC_TP = GCH + 4;   // f32 T rows: word 36 gid + tig, distinct banks
constexpr int GTC_PP = GCH + 8;   // f16 P rows: an 80-B stride, 8 rows on distinct 16-B lanes
constexpr int GTC_CW = S / 16;    // consumer warps a block: 16 value columns each
// dynamic shared memory a block, two buffers each: k / q hi / lo, v, (z), T, P hi / lo, the token scalars
// (beta, exp(gamma), exp(gL - gamma), gamma) as float4, (the norm's o^2 partials), exp(gL): 123,920 B without the
// fused norm, 159,760 with it (the 166,912 opt-in of GA100)
template <bool NORM>
constexpr size_t kTcScanSmem = 2 * (4 * GCH * GTC_KP * 2 + GCH * GTC_VP * 4 + (NORM ? GCH * GTC_VP * 4 : 0) +
                                    GCH * GTC_TP * 4 + 2 * GCH * GTC_PP * 2 + GCH * 16 + (NORM ? GCH * GTC_CW * 4 : 0)) + 16;
static_assert(kTcScanSmem<false> == 123920 && kTcScanSmem<true> == 159760, "gdn_tc.cuh: the shared-memory plan");

__device__ __forceinline__ float gtc_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float gtc_pow2(int e) { return __int_as_float((e + 127) << 23); }   // 2^e, |e| < 127
// the power of two that puts m = max |x| >= 0 in [2^14, 2^15): f16 hi never overflows (65504) and the low part of
// every entry within 2^-17 of the max stays normal; clamped so that the compensating powers stay far from f32's range
__device__ __forceinline__ int gtc_scale_exp(float m) {
    if (m == 0.0f) return 0;
    const int e = 14 - ((__float_as_int(m) >> 23) - 127);
    return e < -40 ? -40 : (e > 40 ? 40 : e);
}

// ---- the tensor-core wrappers (PTX ISA "Matrix Fragments": lane = 4 gid + tig)
//   m16n8k16 f16: A a0 = (row gid, cols 2tig, 2tig+1) a1 = (gid+8, ..) a2 = (gid, 2tig+8..) a3 = (gid+8, 2tig+8..);
//                 B b0 = (k 2tig, 2tig+1; n gid) b1 = (k 2tig+8..; n gid); C/D c0 = (gid, 2tig) c1 = (gid, 2tig+1)
//                 c2 = (gid+8, 2tig) c3 = (gid+8, 2tig+1) - so two adjacent C tiles ARE the next product's A fragment
//   m16n8k8 tf32: A a0 = (gid, tig) a1 = (gid+8, tig) a2 = (gid, tig+4) a3 = (gid+8, tig+4); B b0 = (k tig, n gid)
//                 b1 = (k tig+4, n gid); C as above
//   ldmatrix m8n8.x4.b16: lanes 8m..8m+7 give the row addresses of matrix m; a lane gets (row gid, cols 2tig, 2tig+1)
//                 of each matrix (the lower column in the low half), or with .trans (rows 2tig, 2tig+1; col gid)
__device__ __forceinline__ void gtc_ldsm_x4(uint32_t (&r)[4], const void* smem) {
#if STRATA_GDN_TC_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u; (void) smem; __trap();
#endif
}
__device__ __forceinline__ void gtc_ldsm_x4_trans(uint32_t (&r)[4], const void* smem) {
#if STRATA_GDN_TC_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u; (void) smem; __trap();
#endif
}
__device__ __forceinline__ void gtc_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if STRATA_GDN_TC_SM80
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void) c; (void) a; (void) b0; (void) b1; __trap();
#endif
}
__device__ __forceinline__ void gtc_mma_tf32(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if STRATA_GDN_TC_SM80
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void) c; (void) a; (void) b0; (void) b1; __trap();
#endif
}
// the three passes of a split product, small terms first (as qsa_select.cu): lo.hi, hi.lo, hi.hi
__device__ __forceinline__ void gtc_mma3_f16(float (&c)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0,
                                             uint32_t bh1, uint32_t bl0, uint32_t bl1) {
    gtc_mma_f16(c, al, bh0, bh1);
    gtc_mma_f16(c, ah, bl0, bl1);
    gtc_mma_f16(c, ah, bh0, bh1);
}
__device__ __forceinline__ void gtc_mma3_tf32(float (&c)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0,
                                              uint32_t bh1, uint32_t bl0, uint32_t bl1) {
    gtc_mma_tf32(c, al, bh0, bh1);
    gtc_mma_tf32(c, ah, bl0, bl1);
    gtc_mma_tf32(c, ah, bh0, bh1);
}
__device__ __forceinline__ uint32_t gtc_tf32_hi(float x) {
#if STRATA_GDN_TC_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}
__device__ __forceinline__ void gtc_tf32_split(float x, uint32_t& hi, uint32_t& lo) {
    hi = gtc_tf32_hi(x);
    lo = gtc_tf32_hi(x - __uint_as_float(hi));   // the residual is exact in f32
}
// x0, x1 -> packed f16 hi (x0 in the low half) and lo = f16(x - hi): x - hi is exact in f32 (hi carries x's exponent)
__device__ __forceinline__ void gtc_split_f16x2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    union { __half2 h; uint32_t u; } cv;
    cv.h = __floats2half2_rn(x0, x1);
    hi = cv.u;
    const float2 r = __half22float2(cv.h);
    cv.h = __floats2half2_rn(x0 - r.x, x1 - r.y);
    lo = cv.u;
}
// four consecutive elements of an f16 hi / lo row from a float4 x scale (an 8-byte store each)
__device__ __forceinline__ void gtc_split_store4(uint16_t* hi, uint16_t* lo, float4 v, float scale) {
    uint32_t h0, l0, h1, l1;
    gtc_split_f16x2(v.x * scale, v.y * scale, h0, l0);
    gtc_split_f16x2(v.z * scale, v.w * scale, h1, l1);
    *reinterpret_cast<uint2*>(hi) = make_uint2(h0, h1);
    *reinterpret_cast<uint2*>(lo) = make_uint2(l0, l1);
}
// a m16n8 f32 accumulator tile as the A tile of a m16n8k8 tf32 product (m = the same 16 rows, k = the tile's 8
// columns): a0 = C(gid, tig) sits in lane 4 gid + (tig >> 1), its c0 (tig even) or c1 (odd); a1 the same lane's c2 /
// c3; a2 = C(gid, tig + 4) in the lane two further on; a3 its c2 / c3.  8 shuffles.
__device__ __forceinline__ void gtc_c2a_tf32(const float (&c)[4], float (&a)[4]) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int src0 = 4 * gid + (tig >> 1), src1 = src0 + 2;
    const bool odd = tig & 1;
    const float x0 = __shfl_sync(0xffffffffu, c[0], src0), x1 = __shfl_sync(0xffffffffu, c[1], src0);
    const float x2 = __shfl_sync(0xffffffffu, c[2], src0), x3 = __shfl_sync(0xffffffffu, c[3], src0);
    const float y0 = __shfl_sync(0xffffffffu, c[0], src1), y1 = __shfl_sync(0xffffffffu, c[1], src1);
    const float y2 = __shfl_sync(0xffffffffu, c[2], src1), y3 = __shfl_sync(0xffffffffu, c[3], src1);
    a[0] = odd ? x1 : x0; a[1] = odd ? x3 : x2; a[2] = odd ? y1 : y0; a[3] = odd ? y3 : y2;
}

// ---- the products of a chunk, a warp each (the self-test of gdn_rec_parity --selftest-frag runs them one by one)
// X^T = S^T K^T and QS^T = S^T Q^T: A = the state fragments x sA (= 2^es) split in f16 hi / lo just in time per k16-step
// (tiles 2kk and 2kk + 1 of S^T are that step's A fragment), B = the f16 hi / lo images of k and q ([GCH][GTC_KP] in
// shared memory, rows = tokens) by ldmatrix: matrices (n-tile 2hf + (lane >> 4), k-half (lane >> 3) & 1), a lane's
// row is token 8 n-tile + (lane & 7).  The accumulators come out as 2^(8 + es) times the products.  2 x 96 mma.
__device__ __forceinline__ void gtc_prod_state(const float (&s)[16][4], float sA, const uint16_t* kh, const uint16_t* kl,
                                               const uint16_t* qh, const uint16_t* ql, float (&xt)[4][4], float (&qs)[4][4]) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * GTC_KP + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int kk = 0; kk < S / 16; ++kk) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(s[2 * kk][0] * sA, s[2 * kk][1] * sA, ah[0], al[0]);
        gtc_split_f16x2(s[2 * kk][2] * sA, s[2 * kk][3] * sA, ah[1], al[1]);
        gtc_split_f16x2(s[2 * kk + 1][0] * sA, s[2 * kk + 1][1] * sA, ah[2], al[2]);
        gtc_split_f16x2(s[2 * kk + 1][2] * sA, s[2 * kk + 1][3] * sA, ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int o = off + 16 * hf * GTC_KP + 16 * kk;
            uint32_t bh[4], bl[4];
            gtc_ldsm_x4(bh, kh + o);
            gtc_ldsm_x4(bl, kl + o);
            gtc_mma3_f16(xt[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(xt[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            gtc_ldsm_x4(bh, qh + o);
            gtc_ldsm_x4(bl, ql + o);
            gtc_mma3_f16(qs[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(qs[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
// D^T += Y^T T^T on tf32x3: A = the Y^T tile of k-step ks through gtc_c2a_tf32, B = T[i = 8 nt + gid][i' = 8 ks + tig]
// and [.. + 4] from the f32 [GCH][GTC_TP] copy, split at use.  SKIP leaves out the tiles above T's diagonal (ks > nt):
// they would add exact zeros.  30 (SKIP) or 48 mma.
template <bool SKIP>
__device__ __forceinline__ void gtc_prod_yt(const float (&yt)[4][4], const float* t, float (&dt)[4][4]) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int ks = 0; ks < GCH / 8; ++ks) {
        float a[4];
        gtc_c2a_tf32(yt[ks], a);
        uint32_t ah[4], al[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) gtc_tf32_split(a[q], ah[q], al[q]);
#pragma unroll
        for (int nt = 0; nt < GCH / 8; ++nt) {
            if (SKIP && nt < ks) continue;
            const float* tp = t + (8 * nt + gid) * GTC_TP + 8 * ks + tig;
            uint32_t bh[2], bl[2];
            gtc_tf32_split(tp[0], bh[0], bl[0]);
            gtc_tf32_split(tp[4], bh[1], bl[1]);
            gtc_mma3_tf32(dt[nt], ah, al, bh[0], bh[1], bl[0], bl[1]);
        }
    }
}
// o^T += (sD D^T) (2^8 P^T) on f16x3: A = the D^T tiles x sD (= 2^ed) split per k16-step, B = the f16 hi / lo images
// of P x 2^8 ([GCH][GTC_PP], rows = tokens t) by ldmatrix as in gtc_prod_state.  SKIP leaves out the tiles above P's
// diagonal: tokens t < 16 (n-tiles 0, 1) see no i >= 16 (k-step 1).  18 (SKIP) or 24 mma.
template <bool SKIP>
__device__ __forceinline__ void gtc_prod_dt(const float (&dt)[4][4], float sD, const uint16_t* ph, const uint16_t* pl,
                                            float (&ot)[4][4]) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * GTC_PP + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int ks = 0; ks < GCH / 16; ++ks) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(dt[2 * ks][0] * sD, dt[2 * ks][1] * sD, ah[0], al[0]);
        gtc_split_f16x2(dt[2 * ks][2] * sD, dt[2 * ks][3] * sD, ah[1], al[1]);
        gtc_split_f16x2(dt[2 * ks + 1][0] * sD, dt[2 * ks + 1][1] * sD, ah[2], al[2]);
        gtc_split_f16x2(dt[2 * ks + 1][2] * sD, dt[2 * ks + 1][3] * sD, ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            if (SKIP && ks == 1 && hf == 0) continue;
            const int o = off + 16 * hf * GTC_PP + 16 * ks;
            uint32_t bh[4], bl[4];
            gtc_ldsm_x4(bh, ph + o);
            gtc_ldsm_x4(bl, pl + o);
            gtc_mma3_f16(ot[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(ot[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
// S^T += Dd^T (2^8 K) on f16x3: A = the (already scaled) Dd^T tiles split per k16-step, B = the same k images read the
// other way by ldmatrix.trans: matrices (n-tile 2hf + (lane >> 4), k-half (lane >> 3) & 1), a lane's row is token
// 16 ks + 8 k-half + (lane & 7), its 16-B piece state rows 8 n-tile..  96 mma.
__device__ __forceinline__ void gtc_update_state(float (&s)[16][4], const float (&dd)[4][4], const uint16_t* kh,
                                                 const uint16_t* kl) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * ((lane >> 3) & 1) + (lane & 7)) * GTC_KP + 8 * (lane >> 4);
#pragma unroll
    for (int ks = 0; ks < GCH / 16; ++ks) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(dd[2 * ks][0], dd[2 * ks][1], ah[0], al[0]);
        gtc_split_f16x2(dd[2 * ks][2], dd[2 * ks][3], ah[1], al[1]);
        gtc_split_f16x2(dd[2 * ks + 1][0], dd[2 * ks + 1][1], ah[2], al[2]);
        gtc_split_f16x2(dd[2 * ks + 1][2], dd[2 * ks + 1][3], ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < S / 16; ++hf) {
            const int o = off + 16 * ks * GTC_KP + 16 * hf;
            uint32_t bh[4], bl[4];
            gtc_ldsm_x4_trans(bh, kh + o);
            gtc_ldsm_x4_trans(bl, kl + o);
            gtc_mma3_f16(s[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(s[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
#endif   // !__HIPCC__
