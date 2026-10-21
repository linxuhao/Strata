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

// the warp's largest of non-negative floats: their bits order as unsigned integers, so one redux.sync (sm_80).  Under
// the active mask rather than the full one: the callers are whole warps, and after the stager warps' return the
// compiler cannot prove it and would guard every full-mask warp operation with a reconvergence call that saves and
// restores registers around it (264 B of stack a thread).
__device__ __forceinline__ float gtc_warp_max(float v) {
#if STRATA_GDN_TC_SM80
    return __uint_as_float(__reduce_max_sync(__activemask(), __float_as_uint(v)));
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(__activemask(), v, o));
    return v;
#endif
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
// a float4 from shared memory the compiler does not keep: the per-token scalars are read in three phases of a chunk,
// a component each, and kept across them they cost 24 registers a thread
__device__ __forceinline__ float4 gtc_lds128(const float4* p) {
    float4 v;
#if STRATA_GDN_TC_SM80
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
#else
    v = *p;
#endif
    return v;
}
// a m16n8 f32 accumulator tile as the A tile of a m16n8k8 tf32 product (m = the same 16 rows, k = the tile's 8
// columns): a0 = C(gid, tig) sits in lane 4 gid + (tig >> 1), its c0 (tig even) or c1 (odd); a1 the same lane's c2 /
// c3; a2 = C(gid, tig + 4) in the lane two further on; a3 its c2 / c3.  8 shuffles.
__device__ __forceinline__ void gtc_c2a_tf32(const float (&c)[4], float (&a)[4]) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int src0 = 4 * gid + (tig >> 1), src1 = src0 + 2;
    const bool odd = tig & 1;
    const unsigned mk = __activemask();   // a whole warp (see gtc_warp_max)
    const float x0 = __shfl_sync(mk, c[0], src0), x1 = __shfl_sync(mk, c[1], src0);
    const float x2 = __shfl_sync(mk, c[2], src0), x3 = __shfl_sync(mk, c[3], src0);
    const float y0 = __shfl_sync(mk, c[0], src1), y1 = __shfl_sync(mk, c[1], src1);
    const float y2 = __shfl_sync(mk, c[2], src1), y3 = __shfl_sync(mk, c[3], src1);
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

// ---- the scan: a block per value head (48 blocks, one wave on 48 of the CMP 170HX's 70 SMs), 8 consumer warps of 16
// value columns each walking every chunk of the super-block with their state slice in registers (written back at the
// end as gdn_chunk_scan_kernel does), the same prep scratch (tm, pm, gm).
// STAGERS: two more warps stage the chunks (k / q rows loaded, split to f16 hi / lo and stored - not cp.async, the
// split needs the value in a register - v by cp.async, T copied, P split, the token scalars) a chunk ahead of the
// consumers, handed over with named barriers: the consumers bar.sync FULL(b) at a chunk's start and bar.arrive
// EMPTY(b) at its end, the stagers bar.sync EMPTY(b) before refilling a buffer and bar.arrive FULL(b) after; 320
// threads, 200 registers a thread.  Chunks 0 and 1 are staged without waiting; the stager of chunk 2 waits for the
// consumers' arrive at the end of chunk 0, so there is no cycle.
// Without STAGERS (v0, 256 threads): the consumers stage cooperatively and one __syncthreads a chunk publishes the
// buffer just staged and ends every read of the other one.  The next chunk's v goes by cp.async at the chunk's
// start; its k and q rows, T, P and scalars are loaded, split and stored after the state update, where only the 64
// state registers are live (held across the products they cost 32 registers a thread and ptxas spilled at 255), and
// the rows of the chunk after next are prefetched into L2 at the chunk's start so those loads are L2 hits.
// NORM (step 5: the output norm fused, the FP16 output written here) is not implemented yet; oc_out takes the FP32
// output as gdn_chunk_scan_kernel writes it.
__device__ __forceinline__ void gtc_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void gtc_bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
template <bool NORM, bool STAGERS>
__global__ void __launch_bounds__((GTC_CW + (STAGERS ? 2 : 0)) * 32, 1)
gdn_tc_scan_kernel(float* __restrict__ state, const float* __restrict__ h, const float* __restrict__ beta,
                   const float* __restrict__ tm, const float* __restrict__ pm, const float* __restrict__ gm,
                   float* __restrict__ oc_out, int64_t t0, int nch, int64_t T) {
    static_assert(!NORM, "gdn_tc_scan_kernel: the fused output norm is not implemented yet");
    constexpr int NT = (GTC_CW + (STAGERS ? 2 : 0)) * 32, NS = STAGERS ? 64 : NT;   // threads; staging threads
    constexpr int BAR_FULL = 1, BAR_EMPTY = 3;                                     // named barriers FULL(b), EMPTY(b)
    extern __shared__ __align__(16) unsigned char gtc_sm[];
    uint16_t* kh = reinterpret_cast<uint16_t*>(gtc_sm);                 // [2][GCH][GTC_KP]  f16 of k x 2^8 (hi)
    uint16_t* kl = kh + 2 * GCH * GTC_KP;                               // [2][GCH][GTC_KP]  what hi left out (lo)
    uint16_t* qh = kl + 2 * GCH * GTC_KP;                               // q likewise
    uint16_t* ql = qh + 2 * GCH * GTC_KP;
    float* sv = reinterpret_cast<float*>(ql + 2 * GCH * GTC_KP);        // [2][GCH][GTC_VP]  v
    float* st = sv + 2 * GCH * GTC_VP;                                  // [2][GCH][GTC_TP]  T
    uint16_t* ph = reinterpret_cast<uint16_t*>(st + 2 * GCH * GTC_TP);  // [2][GCH][GTC_PP]  P x 2^8 hi / lo
    uint16_t* pl = ph + 2 * GCH * GTC_PP;
    float4* sc = reinterpret_cast<float4*>(pl + 2 * GCH * GTC_PP);      // [2][GCH]  (beta, exp(gamma), exp(gL - gamma), gamma)
    float* seL = reinterpret_cast<float*>(sc + 2 * GCH);                // [2]  exp(gL)
    const int vh = blockIdx.x, qhd = vh % HK;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const size_t rs = (size_t) HV * S;
    // staging in pieces: k and q rows as 2 x GCH x S / 4 float4 (piece p: q from GCH x S / 4 on, token t, floats
    // r..r + 3), v as GCH x S / 4, T and P as GCH x GCH / 4 each; NS threads take the pieces p = stid + NS m
    auto kq_at = [&](int p, bool& isq, int& t, int& r) {
        isq = p >= GCH * S / 4;
        const int pp = p & (GCH * S / 4 - 1);
        t = pp / (S / 4);
        r = (pp % (S / 4)) * 4;
    };
    auto load_kq4 = [&](int p, int64_t c_t0, int n) {   // rows past the chunk's end: zero
        bool isq; int t, r;
        kq_at(p, isq, t, r);
        return t < n ? __ldg(reinterpret_cast<const float4*>(h + (c_t0 + t) * C + (isq ? 0 : HK * S) + qhd * S + r))
                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    auto store_kq4 = [&](int b, int p, float4 v) {
        bool isq; int t, r;
        kq_at(p, isq, t, r);
        const int o = (b * GCH + t) * GTC_KP + r;
        gtc_split_store4((isq ? qh : kh) + o, (isq ? ql : kl) + o, v, 256.0f);
    };
    auto stage_v4 = [&](int b, int p, int64_t c_t0, int n) {
        const int t = p / (S / 4), r = (p % (S / 4)) * 4;
        float* dst = sv + (b * GCH + t) * GTC_VP + r;
        if (t < n) gdn_cp16(dst, h + (c_t0 + t) * C + 2 * HK * S + vh * S + r);
        else *reinterpret_cast<float4*>(dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    auto load_tp4 = [&](int ch, int p, float4& t4, float4& p4) {   // the prep's f32 [GCH][GCH] rows
        const int t = p / (GCH / 4), i = (p % (GCH / 4)) * 4;
        const int64_t at = (((int64_t) ch * HV + vh) * GCH + t) * GCH + i;
        t4 = __ldg(reinterpret_cast<const float4*>(tm + at));
        p4 = __ldg(reinterpret_cast<const float4*>(pm + at));
    };
    auto store_tp4 = [&](int b, int p, float4 t4, float4 p4) {
        const int t = p / (GCH / 4), i = (p % (GCH / 4)) * 4;
        *reinterpret_cast<float4*>(st + (b * GCH + t) * GTC_TP + i) = t4;
        const int o = (b * GCH + t) * GTC_PP + i;
        gtc_split_store4(ph + o, pl + o, p4, 256.0f);
    };
    // the token scalars, one warp, a lane per token: beta (0 past the chunk's end, as the prep's gate 0 keeps gamma)
    auto load_sc = [&](int ch, int64_t c_t0, int n, float& bet, float& gam, float& gL) {
        bet = lane < n ? __ldg(beta + (c_t0 + lane) * HV + vh) : 0.0f;
        const float* g = gm + ((int64_t) ch * HV + vh) * GCH;
        gam = __ldg(g + lane);
        gL = __ldg(g + GCH - 1);
    };
    auto store_sc = [&](int b, float bet, float gam, float gL) {
        sc[b * GCH + lane] = make_float4(bet, expf(gam), expf(gL - gam), gam);
        if (lane == 0) seL[b] = expf(gL);
    };
    if constexpr (STAGERS) {
        if (warp >= GTC_CW) {   // the stager warps: a chunk into its buffer, a chunk ahead of the consumers
            const int stid = tid - GTC_CW * 32;
            for (int ch = 0; ch < nch; ++ch) {
                const int b = ch & 1;
                const int64_t c_t0 = t0 + (int64_t) ch * GCH;
                const int n = (int) min((int64_t) GCH, T - c_t0);
                if (ch >= 2) gtc_bar_sync(BAR_EMPTY + b, NT);   // the consumers finished chunk ch - 2 (this buffer)
                constexpr int KQ4 = 2 * GCH * S / 4 / NS, V4 = GCH * S / 4 / NS, TP4 = GCH * GCH / 4 / NS;
                float4 kq[KQ4];
#pragma unroll
                for (int m = 0; m < KQ4; ++m) kq[m] = load_kq4(stid + NS * m, c_t0, n);
#pragma unroll
                for (int m = 0; m < V4; ++m) stage_v4(b, stid + NS * m, c_t0, n);
                gdn_cp_commit();
                float4 t4[TP4], p4[TP4];
#pragma unroll
                for (int m = 0; m < TP4; ++m) load_tp4(ch, stid + NS * m, t4[m], p4[m]);
                float bet = 0.0f, gam = 0.0f, gL = 0.0f;
                if (stid < 32) load_sc(ch, c_t0, n, bet, gam, gL);
#pragma unroll
                for (int m = 0; m < KQ4; ++m) store_kq4(b, stid + NS * m, kq[m]);
#pragma unroll
                for (int m = 0; m < TP4; ++m) store_tp4(b, stid + NS * m, t4[m], p4[m]);
                if (stid < 32) store_sc(b, bet, gam, gL);
                gdn_cp_wait_all();
                __threadfence_block();
                gtc_bar_arrive(BAR_FULL + b, NT);   // chunk ch landed
            }
            return;
        }
    }
    const int c0 = 16 * warp;   // the warp's value columns c0 + gid and c0 + gid + 8
    // the state slice as S^T fragments: tile j = rows 8j..8j+7; s[j][0] = S[8j + 2tig][c0 + gid], [1] the row after,
    // [2] column + 8, [3] both (64 scalar loads a lane, 32-B runs, once a launch)
    float s[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float* p = state + (size_t) (8 * j + 2 * tig) * rs + (size_t) vh * S + c0 + gid;
        s[j][0] = p[0]; s[j][1] = p[rs]; s[j][2] = p[8]; s[j][3] = p[rs + 8];
    }
    constexpr int KQ4 = STAGERS ? 1 : 2 * GCH * S / 4 / NT, V4 = STAGERS ? 1 : GCH * S / 4 / NT;   // v0's pieces a thread
    if constexpr (!STAGERS) {   // chunk 0 into buffer 0
        const int n0 = (int) min((int64_t) GCH, T - t0);
        float4 kq[KQ4], t4, p4;
        float bet = 0.0f, gam = 0.0f, gL = 0.0f;
#pragma unroll
        for (int m = 0; m < KQ4; ++m) kq[m] = load_kq4(tid + NT * m, t0, n0);
        load_tp4(0, tid, t4, p4);
        if (warp == 0) load_sc(0, t0, n0, bet, gam, gL);
#pragma unroll
        for (int m = 0; m < V4; ++m) stage_v4(0, tid + NT * m, t0, n0);
        gdn_cp_commit();
#pragma unroll
        for (int m = 0; m < KQ4; ++m) store_kq4(0, tid + NT * m, kq[m]);
        store_tp4(0, tid, t4, p4);
        if (warp == 0) store_sc(0, bet, gam, gL);
        gdn_cp_wait_all();
    }
    const float out_scale = rsqrtf((float) S);
    for (int ch = 0; ch < nch; ++ch) {
        const int b = ch & 1;
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        const bool more = ch + 1 < nch;
        const int64_t n_t0 = c_t0 + GCH;
        const int nn = more ? (int) min((int64_t) GCH, T - n_t0) : 0;
        if constexpr (STAGERS) {
            gtc_bar_sync(BAR_FULL + b, NT);   // chunk ch landed (the stagers arrived)
        } else {
            __syncthreads();   // chunk ch's buffer complete (every thread's cp.async waited); every read of the other one done
            if (more) {   // the next chunk's v into the other buffer (chunk ch - 1 finished reading it before the barrier);
                          // the k and q rows and T / P of the chunk after next into L2 (its k / q rows: 32 x 2 x 512 B)
#pragma unroll
                for (int m = 0; m < V4; ++m) stage_v4(b ^ 1, tid + NT * m, n_t0, nn);
                gdn_cp_commit();
                const int64_t f_t0 = n_t0 + GCH;
                const int t = tid >> 3, l = tid & 7;
                if (f_t0 + t < T) gdn_pf_l2(h + (f_t0 + t) * C + (l < 4 ? HK * S : 0) + qhd * S + 32 * (l & 3));
                if (ch + 2 < nch && tid < GCH) {
                    gdn_pf_l2(tm + (((int64_t) (ch + 2) * HV + vh) * GCH + tid) * GCH);
                    gdn_pf_l2(pm + (((int64_t) (ch + 2) * HV + vh) * GCH + tid) * GCH);
                }
            }
        }
        const uint16_t* bkh = kh + b * GCH * GTC_KP;
        const uint16_t* bkl = kl + b * GCH * GTC_KP;
        const uint16_t* bqh = qh + b * GCH * GTC_KP;
        const uint16_t* bql = ql + b * GCH * GTC_KP;
        const uint16_t* bph = ph + b * GCH * GTC_PP;
        const uint16_t* bpl = pl + b * GCH * GTC_PP;
        const float* bsv = sv + b * GCH * GTC_VP;
        const float* bst = st + b * GCH * GTC_TP;
        const float4* bsc = sc + b * GCH;
        // (a) X^T = S0^T K^T and QS^T = S0^T Q^T, the state x 2^es (its largest entry in [2^14, 2^15))
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) m = fmaxf(m, fabsf(s[j][q]));
        const int es = gtc_scale_exp(gtc_warp_max(m));
        float xt[4][4] = {}, qs[4][4] = {};
        gtc_prod_state(s, gtc_pow2(es), bkh, bkl, bqh, bql, xt, qs);
        // (b) Y^T = beta_i (V^T - e'_i X^T), e'_i = exp(gamma_i) 2^-(8 + es): lane (gid, tig) holds tokens
        // 8nt + 2tig and + 1, columns c0 + gid and + 8
        const float eS = gtc_pow2(-(8 + es));
        float yt[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float4 s0 = gtc_lds128(bsc + 8 * nt + 2 * tig), s1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1);
            const float* v = bsv + (8 * nt + 2 * tig) * GTC_VP + c0 + gid;
            const float e0 = s0.y * eS, e1 = s1.y * eS;
            yt[nt][0] = s0.x * fmaf(-e0, xt[nt][0], v[0]);
            yt[nt][1] = s1.x * fmaf(-e1, xt[nt][1], v[GTC_VP]);
            yt[nt][2] = s0.x * fmaf(-e0, xt[nt][2], v[8]);
            yt[nt][3] = s1.x * fmaf(-e1, xt[nt][3], v[GTC_VP + 8]);
        }
        // (c) D^T = Y^T T^T on tf32x3 (T above its diagonal skipped)
        float dt[4][4] = {};
        gtc_prod_yt<true>(yt, bst, dt);
        // (d) o^T = e_t QS^T + D^T P^T: the accumulator starts as 2^(ed + 8) e_t QS^T (QS^T came out x 2^(8 + es)),
        // D^T goes in x 2^ed, P x 2^8; the output is the accumulator x 2^-(ed + 8), then rsqrt(S) as gdn_chunk_scan_kernel
        float md = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) md = fmaxf(md, fabsf(dt[nt][q]));
        const int ed = gtc_scale_exp(gtc_warp_max(md));
        const float sD = gtc_pow2(ed), sO = gtc_pow2(ed + 8), sOi = gtc_pow2(-(ed + 8)), eO = gtc_pow2(ed - es);
        float ot[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float e0 = gtc_lds128(bsc + 8 * nt + 2 * tig).y * eO, e1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1).y * eO;
            ot[nt][0] = qs[nt][0] * e0; ot[nt][1] = qs[nt][1] * e1; ot[nt][2] = qs[nt][2] * e0; ot[nt][3] = qs[nt][3] * e1;
        }
        gtc_prod_dt<true>(dt, sD, bph, bpl, ot);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int t = 8 * nt + 2 * tig + (q & 1);
                if (t < n) oc_out[(c_t0 + t) * rs + (size_t) vh * S + c0 + gid + 8 * (q >> 1)] = (ot[nt][q] * sOi) * out_scale;
            }
        // (e) S^T = eL S0^T + Dd^T K, Dd^T = f_i D^T: the accumulator x (eL 2^(ed + 8)) before (one rounding, as the
        // s *= gL of gdn_chunk_scan_kernel), Dd^T in x 2^ed, K x 2^8, the accumulator x 2^-(ed + 8) after
        const float sPre = seL[b] * sO;
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) s[j][q] *= sPre;
        float dd[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float f0 = gtc_lds128(bsc + 8 * nt + 2 * tig).z * sD, f1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1).z * sD;
            dd[nt][0] = dt[nt][0] * f0; dd[nt][1] = dt[nt][1] * f1; dd[nt][2] = dt[nt][2] * f0; dd[nt][3] = dt[nt][3] * f1;
        }
        gtc_update_state(s, dd, bkh, bkl);
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) s[j][q] *= sOi;
        if constexpr (STAGERS) {
            gtc_bar_arrive(BAR_EMPTY + b, NT);   // every read of this buffer by this warp is done
        } else {
            if (more) {   // the next chunk's k, q, T, P and scalars into the other buffer
                float4 kq[KQ4], t4, p4;
                float bet = 0.0f, gam = 0.0f, gL = 0.0f;
#pragma unroll
                for (int m = 0; m < KQ4; ++m) kq[m] = load_kq4(tid + NT * m, n_t0, nn);
                load_tp4(ch + 1, tid, t4, p4);
                if (warp == 0) load_sc(ch + 1, n_t0, nn, bet, gam, gL);
#pragma unroll
                for (int m = 0; m < KQ4; ++m) store_kq4(b ^ 1, tid + NT * m, kq[m]);
                store_tp4(b ^ 1, tid, t4, p4);
                if (warp == 0) store_sc(b ^ 1, bet, gam, gL);
            }
            gdn_cp_wait_all();   // this thread's v copies for the next chunk; the barrier at the top publishes them
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float* p = state + (size_t) (8 * j + 2 * tig) * rs + (size_t) vh * S + c0 + gid;
        p[0] = s[j][0]; p[rs] = s[j][1]; p[8] = s[j][2]; p[rs + 8] = s[j][3];
    }
}
#endif   // !__HIPCC__
