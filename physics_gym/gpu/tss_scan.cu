/* tss_scan.cu: the SSM-tree layer's recurrence (nanotss_torch.py _ssm_fwd's time loop and _ssm_bwd's reverse-time
 * adjoints) as two fused kernels, for training on the GPU.
 *
 * In PyTorch each time step of the scan is some sixty small kernels (gathers, scatters, sums), so an update of 1024
 * sequences x 64 steps launched hundreds of thousands of them and the GPU waited on launches. Here one thread block
 * runs one (sequence, tree) pair through every time step, its tree's state rows (NODES x n x p floats) held in shared
 * memory, one thread per state element. The arithmetic is the PyTorch loop's: the decay exp(-(a dt)) in double
 * rounded to float, the same products; sums run in index order, so results agree to float rounding (a routing
 * decision can flip where a score sits on zero, as between the C and PyTorch builds). The model, its .tss files and
 * the CPU / browser forward pass are unchanged: this is training only.
 *
 * Plain C launchers on raw device pointers (as tssenv_kernels.cu), called through ctypes:
 *   nvcc -O3 --fmad=false -arch=sm_86 -shared -Xcompiler -fPIC -o libtss_scan.so tss_scan.cu
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define MAXD1 8     /* tree levels (D + 1) at most */
#define MAXAL 28    /* alternate-walk nodes D (D + 1) / 2 at most */

/* forward. Inputs [B, L, ...] row-major; h [B, TN, n, p] in and out (TN = T x N); per (b, t, tree) outputs:
 * node (global row in the layer), s, dec, R [p] per level; rs [p]; with train, the state rows read (hprev, bf16 or
 * float), and the alternate walks' rows, scores and R. */
extern "C" __global__ void k_ssm_fwd(int B, int L, int T, int D, int N, int n, int p, float cap, int train, int bf16,
        float *h, const float *dt, const float *Bt, const float *Ct, const float *V, const float *Z, const float *s_lin,
        const float *bias, const float *ea,
        int64_t *node_t, float *s_t, float *dec_t, float *R_t, float *rs,
        void *hprev, int64_t *anode, float *as_t, float *aR_t) {
    extern __shared__ float sh[];
    const int bt = blockIdx.x, b = bt / T, tr = bt % T, np = n * p, tid = threadIdx.x, i = tid / p, q = tid % p;
    const int D1 = D + 1, TN = T * N, AL = D * (D + 1) / 2, base_row = tr * N;
    float *S = sh;                        /* the tree's state: N x n x p */
    float *red = S + N * np;              /* n x p scratch for the R sums */
    float *Rv = red + np;                 /* p: this node's R */
    __shared__ int cur; __shared__ float sv, gdt, sdec; __shared__ int path[MAXD1]; __shared__ float pathS[MAXD1];
    float *hb = h + ((size_t)b * TN + base_row) * np;
    for (int k = tid; k < N * np; k += blockDim.x) S[k] = hb[k];
    __syncthreads();
    for (int t = 0; t < L; t++) {
        const size_t bl = (size_t)b * L + t, blt = bl * T + tr;
        const float dtt = dt[blt], Bi = tid < np ? Bt[blt * n + i] : 0, Ci = tid < np ? Ct[blt * n + i] : 0;
        const float Vq = tid < np ? V[bl * p + q] : 0, Zq = tid < np ? Z[bl * p + q] : 0;
        const float *sl = s_lin + bl * TN;
        float racc = 0;   /* rs: the sum of R over the path (thread q of row i == 0 keeps it) */
        if (tid == 0) cur = 0;
        __syncthreads();
        for (int kk = 0; kk < D1; kk++) {
            const int loc = cur, row = base_row + loc;
            if (tid == 0) { double e = exp(-(double)(ea[row] * dtt)); float d = (float)e; if (cap < 1 && d > cap) d = cap; sdec = d; }
            __syncthreads();
            const float dec = sdec; float hv = 0;
            if (tid < np) {
                const float ha = S[loc * np + tid];
                if (train) { const size_t o = ((blt * D1 + kk) * n + i) * p + q; if (bf16) ((__nv_bfloat16 *)hprev)[o] = __float2bfloat16(ha); else ((float *)hprev)[o] = ha; }
                hv = ha * dec; red[tid] = Ci * hv;
                S[loc * np + tid] = hv + (dtt * Bi) * Vq;
            }
            __syncthreads();
            if (tid < p) { float r = 0; for (int k = 0; k < n; k++) r += red[k * p + tid]; Rv[tid] = r; racc += r; R_t[(blt * D1 + kk) * p + tid] = r; }
            __syncthreads();
            if (tid == 0) {
                float z = 0; for (int k = 0; k < p; k++) z += Rv[k] * Z[bl * p + k];
                const float s = (sl[row] + bias[row]) + z;
                node_t[blt * D1 + kk] = row; s_t[blt * D1 + kk] = s; dec_t[blt * D1 + kk] = dec; path[kk] = loc; pathS[kk] = s;
                cur = 2 * loc + 1 + (s > 0);
            }
            __syncthreads();
        }
        if (tid < p) rs[blt * p + tid] = racc;
        if (train && D > 0) {   /* the alternate walks: from the sibling of each level's choice down to the leaves (rows only read) */
            int wbase = 0;
            for (int w = 0; w < D; w++) {
                if (tid == 0) cur = 2 * path[w] + 2 - (pathS[w] > 0);
                __syncthreads();
                for (int k = 0; k < D - w; k++) {
                    const int loc = cur, row = base_row + loc, ai = wbase + k;
                    if (tid == 0) sdec = (float)exp(-(double)(ea[row] * dtt));
                    if (tid < np) red[tid] = Ci * S[loc * np + tid];
                    __syncthreads();
                    if (tid < p) { float r = 0; for (int kq = 0; kq < n; kq++) r += red[kq * p + tid]; r *= sdec; Rv[tid] = r; aR_t[(blt * AL + ai) * p + tid] = r; }
                    __syncthreads();
                    if (tid == 0) {
                        float z = 0; for (int kq = 0; kq < p; kq++) z += Rv[kq] * Z[bl * p + kq];
                        const float s = (sl[row] + bias[row]) + z;
                        anode[blt * AL + ai] = row; as_t[blt * AL + ai] = s; cur = 2 * loc + 1 + (s > 0);
                    }
                    __syncthreads();
                }
                wbase += D - w;
            }
        }
    }
    for (int k = tid; k < N * np; k += blockDim.x) hb[k] = S[k];
}

/* backward: the recurrent adjoints in reverse time for one (sequence, tree), lam (the state adjoint) in shared memory.
 * dR [B, L, T, D1, p] = drs + ds Z (from the caller); outputs dBt, dCt [B, L, T, n], dVp [B, L, T, p] (this tree's share
 * of dV), ddt [B, L, T], GA [B, L, T, D1]. */
extern "C" __global__ void k_ssm_bwd(int B, int L, int T, int D, int N, int n, int p, int bf16,
        const int64_t *node_t, const void *hprev, const float *dec_t, const float *dR, const float *Bt, const float *Ct,
        const float *V, const float *dt, const float *eaN,
        float *dBt, float *dCt, float *dVp, float *ddt, float *GA) {
    extern __shared__ float sh[];
    const int bt = blockIdx.x, b = bt / T, tr = bt % T, np = n * p, tid = threadIdx.x, i = tid / p, q = tid % p;
    const int D1 = D + 1, base_row = tr * N;
    float *lam = sh;                      /* N x n x p */
    float *r1 = lam + N * np;             /* np scratch */
    float *laV = r1 + np;                 /* D1 x n */
    float *acc = laV + MAXD1 * 16;        /* small sums */
    for (int k = tid; k < N * np; k += blockDim.x) lam[k] = 0;
    __syncthreads();
    for (int t = L - 1; t >= 0; t--) {
        const size_t bl = (size_t)b * L + t, blt = bl * T + tr;
        const float gv = dt[blt], Bi = tid < np ? Bt[blt * n + i] : 0, Ci = tid < np ? Ct[blt * n + i] : 0, Vq = tid < np ? V[bl * p + q] : 0;
        float dB_i = 0, dC_i = 0, dV_q = 0, ddt_t = 0;
        for (int kk = 0; kk < D1; kk++) {
            const int loc = (int)(node_t[blt * D1 + kk] - base_row);
            const float dec = dec_t[blt * D1 + kk], dRq = tid < np ? dR[(blt * D1 + kk) * p + q] : 0, e = eaN[blt * D1 + kk];
            float la = 0, hp = 0;
            if (tid < np) {
                la = lam[loc * np + tid];
                const size_t o = ((blt * D1 + kk) * n + i) * p + q;
                hp = bf16 ? __bfloat162float(((const __nv_bfloat16 *)hprev)[o]) : ((const float *)hprev)[o];
                r1[tid] = la * Vq;
            }
            __syncthreads();
            if (tid < n) { float s = 0; for (int k = 0; k < p; k++) s += r1[tid * p + k]; laV[kk * 16 + tid] = s; }   /* laV[i] = sum_q la V */
            __syncthreads();
            if (tid < np) r1[tid] = la * Bi;
            __syncthreads();
            if (tid < p) { float s = 0; for (int k = 0; k < n; k++) s += r1[k * p + tid]; dV_q += s; }   /* sum_i la B */
            __syncthreads();
            if (tid < np) r1[tid] = hp * dRq;
            __syncthreads();
            if (tid < n) { float s = 0; for (int k = 0; k < p; k++) s += r1[tid * p + k]; dC_i += dec * s; dB_i += laV[kk * 16 + tid]; }
            if (tid == 0) { float s = 0; for (int k = 0; k < n; k++) s += laV[kk * 16 + k] * Bt[blt * n + k]; acc[0] = s; }   /* dgain */
            __syncthreads();
            float dht = 0;
            if (tid < np) { dht = la + Ci * dRq; r1[tid] = dht * hp; }
            __syncthreads();
            if (tid == 0) {
                float s = 0; for (int k = 0; k < np; k++) s += r1[k];   /* ddec */
                const float g = s * dec * (-e); ddt_t += acc[0] + g; GA[blt * D1 + kk] = g * gv;
            }
            if (tid < np) lam[loc * np + tid] = dec * dht;
            __syncthreads();
        }
        if (tid < n) { dBt[blt * n + tid] = gv * dB_i; dCt[blt * n + tid] = dC_i; }
        if (tid < p) dVp[blt * p + tid] = gv * dV_q;
        if (tid == 0) ddt[blt] = ddt_t;
        __syncthreads();
    }
}

extern "C" {
const char *tss_scan_fwd(int B, int L, int T, int D, int N, int n, int p, float cap, int train, int bf16,
        void *h, const void *dt, const void *Bt, const void *Ct, const void *V, const void *Z, const void *s_lin,
        const void *bias, const void *ea, void *node_t, void *s_t, void *dec_t, void *R_t, void *rs,
        void *hprev, void *anode, void *as_t, void *aR_t, void *stream) {
    if (D + 1 > MAXD1 || D * (D + 1) / 2 > MAXAL || n > 16 || n * p > 1024) return "tss_scan: tree or state too large";
    const int threads = ((n * p + 31) / 32) * 32; const size_t shm = sizeof(float) * ((size_t)N * n * p + n * p + p);
    if (shm > 48 * 1024) cudaFuncSetAttribute(k_ssm_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    k_ssm_fwd<<<B * T, threads, shm, (cudaStream_t)stream>>>(B, L, T, D, N, n, p, cap, train, bf16, (float *)h, (const float *)dt, (const float *)Bt,
        (const float *)Ct, (const float *)V, (const float *)Z, (const float *)s_lin, (const float *)bias, (const float *)ea,
        (int64_t *)node_t, (float *)s_t, (float *)dec_t, (float *)R_t, (float *)rs, hprev, (int64_t *)anode, (float *)as_t, (float *)aR_t);
    cudaError_t r = cudaGetLastError(); return r == cudaSuccess ? 0 : cudaGetErrorString(r);
}
const char *tss_scan_bwd(int B, int L, int T, int D, int N, int n, int p, int bf16,
        const void *node_t, const void *hprev, const void *dec_t, const void *dR, const void *Bt, const void *Ct, const void *V,
        const void *dt, const void *eaN, void *dBt, void *dCt, void *dVp, void *ddt, void *GA, void *stream) {
    if (D + 1 > MAXD1 || n > 16 || n * p > 1024) return "tss_scan: tree or state too large";
    const int threads = ((n * p + 31) / 32) * 32; const size_t shm = sizeof(float) * ((size_t)N * n * p + n * p + MAXD1 * 16 + 8);
    if (shm > 48 * 1024) cudaFuncSetAttribute(k_ssm_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    k_ssm_bwd<<<B * T, threads, shm, (cudaStream_t)stream>>>(B, L, T, D, N, n, p, bf16, (const int64_t *)node_t, hprev, (const float *)dec_t,
        (const float *)dR, (const float *)Bt, (const float *)Ct, (const float *)V, (const float *)dt, (const float *)eaN,
        (float *)dBt, (float *)dCt, (float *)dVp, (float *)ddt, (float *)GA);
    cudaError_t r = cudaGetLastError(); return r == cudaSuccess ? 0 : cudaGetErrorString(r);
}
}
