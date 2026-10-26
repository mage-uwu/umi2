/* tss_scan.cu: the SSM-tree layer's recurrence (nanotss_torch.py _ssm_fwd's time loop and _ssm_bwd's reverse-time
 * adjoints) as two fused kernels, for training on the GPU.
 *
 * In PyTorch each time step of the scan is some sixty small kernels (gathers, scatters, sums), so an update of 1024
 * sequences x 64 steps launched hundreds of thousands of them and the GPU waited on launches. Here one thread block
 * runs one (sequence, tree) pair through every time step, its tree's state rows (NODES x n x p floats) held in shared
 * memory, one thread per state element. Each thread only ever touches its own element of every node, so the
 * recurrence itself needs no exchange between threads: the forward exchanges one sum per tree level (the routing
 * score, a warp-shuffle sum), the alternate walks advance in lockstep, the R sums wait to the end of the step, and
 * the backward exchanges nothing until one batched reduction per step. The arithmetic is the PyTorch loop's (the
 * decay exp(-(a dt)) in double rounded to float, the same products); sums run in another order, so results agree to
 * float rounding (a routing decision can flip where a score sits on zero, as between the C and PyTorch builds).
 * Rollouts (one step, no tape) read and write the visited rows in place instead of staging the tree. With lens (one per
 * sequence: the update's minibatches are episodes cut at their ends, padded to the longest) each block stops at its
 * sequence's end, forward and backward: the padded steps' outputs are masked and their adjoints zero, so the valid ones are
 * as before; the caller zero-fills the outputs the padded steps leave (a valid row, finite values). The model, its
 * .tss files and the CPU / browser forward pass are unchanged: this is training only.
 *
 * Plain C launchers on raw device pointers (as tssenv_kernels.cu), called through ctypes:
 *   nvcc -O3 --fmad=false -arch=sm_86 -shared -Xcompiler -fPIC -o libtss_scan.so tss_scan.cu
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define MAXD 7      /* tree depth at most (D + 1 levels) */

/* the shapes the kernels take: a row of the [n, p] state inside one warp, whole warps */
static int fits(int D, int n, int p) { return D >= 1 && D <= MAXD && p <= 32 && 32 % p == 0 && (n * p) % 32 == 0 && n * p <= 1024; }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float col_sum(float v, int p) {   /* over the warp's lanes of the same column q (lanes 0..p-1 hold it) */
    for (int o = 16; o >= p; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float row_sum(float v, int p) {   /* over the p lanes of the same row i */
    for (int o = p >> 1; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* forward. Inputs [B, L, ...] row-major; h [B, TN, n, p] in and out (TN = T x N); per (b, t, tree) outputs:
 * node (global row in the layer), s, dec, R [p] per level; rs [p]; with train, the state rows read (hprev, bf16 or
 * float), and the alternate walks' rows, scores and R. DIRECT: the rows in h itself (one step), not a staged copy. */
template <int D, bool DIRECT>
__global__ void k_ssm_fwd(int B, int L, int T, int N, int n, int p, float cap, int train, int bf16,
        float *h, const float *dt, const float *Bt, const float *Ct, const float *V, const float *Z, const float *s_lin,
        const float *bias, const float *ea,
        int64_t *node_t, float *s_t, float *dec_t, float *R_t, float *rs,
        void *hprev, int64_t *anode, float *as_t, float *aR_t, const int64_t *lens) {
    constexpr int D1 = D + 1, AL = D * (D + 1) / 2, NR = D1 + AL;
    extern __shared__ float sh[];
    const int bt = blockIdx.x, b = bt / T, tr = bt % T, np = n * p, e = threadIdx.x, i = e / p, q = e % p;
    const int lane = e & 31, wid = e >> 5, nw = np >> 5, TN = T * N, base_row = tr * N;
    float *hb = h + ((size_t)b * TN + base_row) * np;
    float *S = DIRECT ? hb : sh;                  /* the tree's state: N x n x p */
    float *decN = sh + (DIRECT ? 0 : N * np);     /* [2][N] every node's decay this step (by step parity) */
    float *sN = decN + 2 * N;                     /* [2][N] its score's input part, s_lin + bias */
    float *zp = sN + 2 * N;                       /* [2][D][nw] per-warp parts of the scores (by reduction parity) */
    float *colp = zp + 2 * D * nw;                /* [NR][nw][p] per-warp parts of the R sums */
    int *aloc = (int *)(colp + NR * nw * p);      /* [AL] the alternate walks' nodes */
    if (!DIRECT) for (int k = e; k < N * np; k += np) S[k] = hb[k];
    int red = 0;
    const int Lb = lens ? (int)min((int64_t)L, lens[b]) : L;   /* this sequence's own length */
    for (int t = 0; t < Lb; t++) {
        const size_t bl = (size_t)b * L + t, blt = bl * T + tr;
        const float dtt = dt[blt], Bi = Bt[blt * n + i], Ci = Ct[blt * n + i], Vq = V[bl * p + q], Zq = Z[bl * p + q], gB = dtt * Bi;
        float *dN = decN + (t & 1) * N, *sNt = sN + (t & 1) * N;
        for (int k = e; k < N; k += np) { const int row = base_row + k; dN[k] = (float)exp(-(double)(ea[row] * dtt)); sNt[k] = s_lin[bl * TN + row] + bias[row]; }
        __syncthreads();
        float cp[NR];   /* this thread's C h products at each node visited: the path, then the alternates */
        int path[D1]; float pS[D1];
        int cur = 0;
#pragma unroll
        for (int kk = 0; kk < D1; kk++) {
            const int loc = cur;
            float dec = dN[loc]; if (cap < 1 && dec > cap) dec = cap;
            const float ha = S[loc * np + e];
            if (train) { const size_t o = (blt * D1 + kk) * np + e; if (bf16) ((__nv_bfloat16 *)hprev)[o] = __float2bfloat16(ha); else ((float *)hprev)[o] = ha; }
            const float hv = ha * dec;
            S[loc * np + e] = hv + gB * Vq;
            cp[kk] = Ci * hv;
            float *zb = zp + (red & 1) * D * nw;
            const float z = warp_sum(cp[kk] * Zq);
            if (lane == 0) zb[wid] = z;
            __syncthreads(); red++;
            float zs = 0; for (int w = 0; w < nw; w++) zs += zb[w];
            const float s = sNt[loc] + zs;
            path[kk] = loc; pS[kk] = s;
            if (e == 0) { node_t[blt * D1 + kk] = base_row + loc; s_t[blt * D1 + kk] = s; dec_t[blt * D1 + kk] = dec; }
            cur = 2 * loc + 1 + (s > 0);
        }
        if (train) {   /* the alternate walks, in lockstep: from the sibling of each level's choice down to the leaves (rows only read) */
            int wl[D];
#pragma unroll
            for (int w = 0; w < D; w++) wl[w] = 2 * path[w] + 2 - (pS[w] > 0);
#pragma unroll
            for (int k = 0; k < D; k++) {
                float *zb = zp + (red & 1) * D * nw;
#pragma unroll
                for (int w = 0; w < D - k; w++) {   /* walk w has D - w nodes; its first sits at w D - w (w - 1) / 2 */
                    const int ai = w * D - w * (w - 1) / 2 + k;
                    cp[D1 + ai] = Ci * S[wl[w] * np + e];
                    const float z = warp_sum(cp[D1 + ai] * Zq);
                    if (lane == 0) zb[w * nw + wid] = z;
                }
                __syncthreads(); red++;
#pragma unroll
                for (int w = 0; w < D - k; w++) {
                    const int ai = w * D - w * (w - 1) / 2 + k, loc = wl[w];
                    float zs = 0; for (int x = 0; x < nw; x++) zs += zb[w * nw + x];
                    const float s = sNt[loc] + dN[loc] * zs;
                    if (e == 0) { anode[blt * AL + ai] = base_row + loc; as_t[blt * AL + ai] = s; aloc[ai] = loc; }
                    wl[w] = 2 * loc + 1 + (s > 0);
                }
            }
        }
        const int nr = train ? NR : D1;   /* the R sums: over the rows i of each node's products */
#pragma unroll
        for (int r = 0; r < NR; r++) if (r < nr) { const float v = col_sum(cp[r], p); if (lane < p) colp[(r * nw + wid) * p + lane] = v; }
        __syncthreads();
        for (int o = e; o < nr * p; o += np) {
            const int r = o / p, qq = o % p;
            float v = 0; for (int w = 0; w < nw; w++) v += colp[(r * nw + w) * p + qq];
            if (r < D1) R_t[(blt * D1 + r) * p + qq] = v;
            else aR_t[(blt * AL + r - D1) * p + qq] = v * dN[aloc[r - D1]];
        }
        if (e < p) {
            float v = 0;
            for (int r = 0; r < D1; r++) { float x = 0; for (int w = 0; w < nw; w++) x += colp[(r * nw + w) * p + e]; v += x; }
            rs[blt * p + e] = v;
        }
    }
    if (!DIRECT) for (int k = e; k < N * np; k += np) hb[k] = S[k];
}

/* backward: the recurrent adjoints in reverse time for one (sequence, tree), lam (the state adjoint) in shared memory.
 * dR [B, L, T, D1, p] = drs + ds Z (from the caller); outputs dBt, dCt [B, L, T, n], dVp [B, L, T, p] (this tree's share
 * of dV), ddt [B, L, T], GA [B, L, T, D1]. Every sum is linear in the per-element terms, so each thread accumulates its
 * own over the levels and the block reduces once per step. */
template <int D>
__global__ void k_ssm_bwd(int B, int L, int T, int N, int n, int p, int bf16,
        const int64_t *node_t, const void *hprev, const float *dec_t, const float *dR, const float *Bt, const float *Ct,
        const float *V, const float *dt, const float *eaN,
        float *dBt, float *dCt, float *dVp, float *ddt, float *GA, const int64_t *lens) {
    constexpr int D1 = D + 1;
    extern __shared__ float sh[];
    const int bt = blockIdx.x, b = bt / T, tr = bt % T, np = n * p, e = threadIdx.x, i = e / p, q = e % p;
    const int lane = e & 31, wid = e >> 5, nw = np >> 5, base_row = tr * N;
    float *lam = sh;                              /* N x n x p */
    float *cpart = lam + N * np;                  /* [2][nw][p] per-warp parts of dV (by step parity) */
    float *fpart = cpart + 2 * nw * p;            /* [2][D1 + 1][nw] per-warp parts of ddec per level, and dgain */
    for (int k = e; k < N * np; k += np) lam[k] = 0;
    const int Lb = lens ? (int)min((int64_t)L, lens[b]) : L;   /* from its own end: the padded steps' adjoints are zero */
    for (int t = Lb - 1; t >= 0; t--) {
        const size_t bl = (size_t)b * L + t, blt = bl * T + tr;
        const float gv = dt[blt], Bi = Bt[blt * n + i], Ci = Ct[blt * n + i], Vq = V[bl * p + q];
        float aB = 0, aV = 0, aC = 0, ag = 0, dd[D1], dk[D1], ek[D1];
#pragma unroll
        for (int kk = 0; kk < D1; kk++) {
            const size_t nk = blt * D1 + kk;
            const int loc = (int)(node_t[nk] - base_row);
            const float dec = dec_t[nk], dRq = dR[nk * p + q];
            const float hp = bf16 ? __bfloat162float(((const __nv_bfloat16 *)hprev)[nk * np + e]) : ((const float *)hprev)[nk * np + e];
            const float la = lam[loc * np + e];
            aB += la * Vq; aV += la * Bi; aC += dec * (hp * dRq); ag += la * Vq * Bi;
            const float dht = la + Ci * dRq;
            dd[kk] = dht * hp; dk[kk] = dec; ek[kk] = eaN[nk];
            lam[loc * np + e] = dec * dht;
        }
        const float rB = row_sum(aB, p), rC = row_sum(aC, p);
        if (q == 0) { dBt[blt * n + i] = gv * rB; dCt[blt * n + i] = rC; }
        float *cp = cpart + (t & 1) * nw * p, *fp = fpart + (t & 1) * (D1 + 1) * nw;
        const float cV = col_sum(aV, p); if (lane < p) cp[wid * p + lane] = cV;
        const float g0 = warp_sum(ag); if (lane == 0) fp[D1 * nw + wid] = g0;
#pragma unroll
        for (int kk = 0; kk < D1; kk++) { const float v = warp_sum(dd[kk]); if (lane == 0) fp[kk * nw + wid] = v; }
        __syncthreads();
        if (e < p) { float v = 0; for (int w = 0; w < nw; w++) v += cp[w * p + e]; dVp[blt * p + e] = gv * v; }
        if (e == 0) {
            float tot = 0; for (int w = 0; w < nw; w++) tot += fp[D1 * nw + w];   /* dgain */
#pragma unroll
            for (int kk = 0; kk < D1; kk++) {
                float s = 0; for (int w = 0; w < nw; w++) s += fp[kk * nw + w];   /* ddec */
                const float g = s * dk[kk] * (-ek[kk]); GA[blt * D1 + kk] = g * gv; tot += g;
            }
            ddt[blt] = tot;
        }
    }
}

template <int D, bool DIRECT> static size_t fwd_shm(int N, int n, int p) {
    const int np = n * p, nw = np / 32, D1 = D + 1, AL = D * (D + 1) / 2;
    return sizeof(float) * ((DIRECT ? 0 : (size_t)N * np) + 4 * N + 2 * D * nw + (size_t)(D1 + AL) * nw * p + AL);
}
template <int D, bool DIRECT> static const char *fwd(int B, int L, int T, int N, int n, int p, float cap, int train, int bf16,
        void *h, const void *dt, const void *Bt, const void *Ct, const void *V, const void *Z, const void *s_lin,
        const void *bias, const void *ea, void *node_t, void *s_t, void *dec_t, void *R_t, void *rs,
        void *hprev, void *anode, void *as_t, void *aR_t, const void *lens, cudaStream_t stream) {
    const size_t shm = fwd_shm<D, DIRECT>(N, n, p);
    if (shm > 48 * 1024) cudaFuncSetAttribute(k_ssm_fwd<D, DIRECT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    k_ssm_fwd<D, DIRECT><<<B * T, n * p, shm, stream>>>(B, L, T, N, n, p, cap, train, bf16, (float *)h, (const float *)dt, (const float *)Bt,
        (const float *)Ct, (const float *)V, (const float *)Z, (const float *)s_lin, (const float *)bias, (const float *)ea,
        (int64_t *)node_t, (float *)s_t, (float *)dec_t, (float *)R_t, (float *)rs, hprev, (int64_t *)anode, (float *)as_t, (float *)aR_t, (const int64_t *)lens);
    cudaError_t r = cudaGetLastError(); return r == cudaSuccess ? 0 : cudaGetErrorString(r);
}
template <int D> static const char *bwd(int B, int L, int T, int N, int n, int p, int bf16,
        const void *node_t, const void *hprev, const void *dec_t, const void *dR, const void *Bt, const void *Ct, const void *V,
        const void *dt, const void *eaN, void *dBt, void *dCt, void *dVp, void *ddt, void *GA, const void *lens, cudaStream_t stream) {
    const int np = n * p, nw = np / 32;
    const size_t shm = sizeof(float) * ((size_t)N * np + 2 * nw * p + 2 * (D + 2) * nw);
    if (shm > 48 * 1024) cudaFuncSetAttribute(k_ssm_bwd<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    k_ssm_bwd<D><<<B * T, np, shm, stream>>>(B, L, T, N, n, p, bf16, (const int64_t *)node_t, hprev, (const float *)dec_t,
        (const float *)dR, (const float *)Bt, (const float *)Ct, (const float *)V, (const float *)dt, (const float *)eaN,
        (float *)dBt, (float *)dCt, (float *)dVp, (float *)ddt, (float *)GA, (const int64_t *)lens);
    cudaError_t r = cudaGetLastError(); return r == cudaSuccess ? 0 : cudaGetErrorString(r);
}

extern "C" {
int tss_scan_fits(int D, int n, int p) { return fits(D, n, p); }
const char *tss_scan_fwd(int B, int L, int T, int D, int N, int n, int p, float cap, int train, int bf16,
        void *h, const void *dt, const void *Bt, const void *Ct, const void *V, const void *Z, const void *s_lin,
        const void *bias, const void *ea, void *node_t, void *s_t, void *dec_t, void *R_t, void *rs,
        void *hprev, void *anode, void *as_t, void *aR_t, const void *lens, void *stream) {
    if (!fits(D, n, p)) return "tss_scan: tree or state shape not supported";
    const bool direct = L == 1 && !train;
#define F(DD) case DD: return direct ? fwd<DD, true>(B, L, T, N, n, p, cap, train, bf16, h, dt, Bt, Ct, V, Z, s_lin, bias, ea, node_t, s_t, dec_t, R_t, rs, hprev, anode, as_t, aR_t, lens, (cudaStream_t)stream) \
                                     : fwd<DD, false>(B, L, T, N, n, p, cap, train, bf16, h, dt, Bt, Ct, V, Z, s_lin, bias, ea, node_t, s_t, dec_t, R_t, rs, hprev, anode, as_t, aR_t, lens, (cudaStream_t)stream);
    switch (D) { F(1) F(2) F(3) F(4) F(5) F(6) F(7) }
#undef F
    return "tss_scan: depth not supported";
}
const char *tss_scan_bwd(int B, int L, int T, int D, int N, int n, int p, int bf16,
        const void *node_t, const void *hprev, const void *dec_t, const void *dR, const void *Bt, const void *Ct, const void *V,
        const void *dt, const void *eaN, void *dBt, void *dCt, void *dVp, void *ddt, void *GA, const void *lens, void *stream) {
    if (!fits(D, n, p)) return "tss_scan: tree or state shape not supported";
#define G(DD) case DD: return bwd<DD>(B, L, T, N, n, p, bf16, node_t, hprev, dec_t, dR, Bt, Ct, V, dt, eaN, dBt, dCt, dVp, ddt, GA, lens, (cudaStream_t)stream);
    switch (D) { G(1) G(2) G(3) G(4) G(5) G(6) G(7) }
#undef G
    return "tss_scan: depth not supported";
}
}
