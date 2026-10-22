/* ssm_tree.c: a standalone byte-level language model built only from SSM-tree blocks, with its trainer.
 *
 * Extracted from nanotss_blt.c (branch icl-ttt), default byte-mode SSM layer: per-tree-group keys and
 * queries (GQA analogue), QK-norm and rotary on B/C, AdamW or blockwise Muon. Plain float32 throughout:
 * none of the ternary weights, int8 activations or AVX-512 kernels of the parent file, and no FFF blocks.
 *
 * Model
 *   x_0 = embed[byte]; then n_layer residual SSM-tree blocks; logits = embed . rmsnorm(x) (tied head).
 *
 * SSM-tree block (one position t, input x):
 *   u  = rmsnorm(x) * norm
 *   m  = sum_j conv[j] * u_{t-j}                       depthwise causal conv, K taps (tap 0 = identity at init)
 *   pr = proj . m  ->  B[G][n], C[G][n], V[p], Z[p], dtz[T]
 *   B,C <- L2-normalize per group (QK-norm), then rotate coordinate pairs by angle t*theta_k (rotary)
 *   for each of T binary trees (tree tr uses key/query group tr*G/T), walk D+1 nodes from the root:
 *     each node owns a state H[n][p] that persists across positions and changes only when visited:
 *       dt  = softplus(dtz[tr] + dt_bias[tr]),  dec = exp(-exp(A_log[node]) * dt)
 *       H  *= dec;  R = C^T H  (a p-vector read);  H += dt * B V^T  (a rank-1 write)
 *       s   = w_in[node] . m + bias[node] + R . Z
 *       y  += gelu(s) * w_out[node]
 *       go to child 2*node+1+(s>0)                     hard routing
 *     Racc[tr] += R
 *   x += y + U^T Racc                                  residual
 *
 * Routing is hard, so the gradient reaches only visited nodes; the router learns through gelu(s). The parent
 * file adds a sampled "routing credit" for the unvisited branch; it is left out here.
 *
 * Build:   cc -O3 -march=native -fopenmp ssm_tree.c -lm -o ssm_tree
 * Train:   ./ssm_tree --data input.txt --seconds 300 --threads 4 [--save m.bin] [--sample 400]
 * Sample:  ./ssm_tree --load m.bin --prompt "ROMEO:" --sample 400
 * Check:   cc -O2 -DDOUBLE ssm_tree.c -lm -o ssm_tree_dbl && ./ssm_tree_dbl --check
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef DOUBLE
typedef double real;
#else
typedef float real;
#endif

/* ------------------------------------------------------------------ config and parameters */
typedef struct { int d, n_layer, T, D, n, p, K, G, qknorm, rope; } Cfg;
#define NODES(c) ((1 << ((c)->D + 1)) - 1)
#define PDIM(c) (2 * (c)->n * (c)->G + 2 * (c)->p + (c)->T)
#define V 256

typedef struct { real *norm, *conv, *w_in, *bias, *w_out, *A_log, *proj, *dt_bias, *U; } Layer;
typedef struct { real *embed, *norm_f; Layer *ly; real *all; size_t size; } Params;

static size_t layer_size(const Cfg *c) {
    size_t d = c->d, TN = (size_t)c->T * NODES(c);
    return d + (size_t)c->K * d + TN * d + TN + TN * d + TN + (size_t)PDIM(c) * d + c->T + (size_t)c->T * c->p * d;
}
static void params_alloc(const Cfg *c, Params *P) {
    const size_t d = c->d, TN = (size_t)c->T * NODES(c);
    P->size = V * d + d + c->n_layer * layer_size(c);
    P->all = calloc(P->size, sizeof(real));
    P->ly = calloc(c->n_layer, sizeof(Layer));
    real *q = P->all;
#define TAKE(ptr, cnt) do { ptr = q; q += (cnt); } while (0)
    TAKE(P->embed, V * d);
    TAKE(P->norm_f, d);
    for (int l = 0; l < c->n_layer; l++) {
        Layer *L = P->ly + l;
        TAKE(L->norm, d); TAKE(L->conv, (size_t)c->K * d);
        TAKE(L->w_in, TN * d); TAKE(L->bias, TN); TAKE(L->w_out, TN * d); TAKE(L->A_log, TN);
        TAKE(L->proj, (size_t)PDIM(c) * d); TAKE(L->dt_bias, c->T); TAKE(L->U, (size_t)c->T * c->p * d);
    }
#undef TAKE
}

static uint64_t rng_state = 88172645463325252ULL;
static double urand(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (*s >> 11) * (1.0 / 9007199254740992.0); }
static double randn(void) { double a = urand(&rng_state) + 1e-12, b = urand(&rng_state); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
#define UNI(k) ((real)((2 * urand(&rng_state) - 1) * (k)))

static void params_init(const Cfg *c, Params *P) {
    const int d = c->d, T = c->T, TN = c->T * NODES(c);
    const real k_in = 1 / sqrt((real)d), k_out = 1 / sqrt((real)(T * (c->D + 1))), k_u = k_out / sqrt((real)c->p);
    for (size_t i = 0; i < (size_t)V * d; i++) P->embed[i] = (real)(0.02 * randn());
    for (int i = 0; i < d; i++) P->norm_f[i] = 1;
    for (int l = 0; l < c->n_layer; l++) {
        Layer *L = P->ly + l;
        for (int i = 0; i < d; i++) L->norm[i] = 1;
        for (int i = 0; i < c->K * d; i++) L->conv[i] = i < d;
        for (int r = 0; r < TN; r++) {
            for (int i = 0; i < d; i++) L->w_in[(size_t)r * d + i] = UNI(k_in);
            L->bias[r] = UNI(k_in);
            for (int i = 0; i < d; i++) L->w_out[(size_t)r * d + i] = UNI(k_out);
            L->A_log[r] = (real)log(1 + 15 * urand(&rng_state));
        }
        for (size_t i = 0; i < (size_t)PDIM(c) * d; i++) L->proj[i] = UNI(k_in);
        for (int t = 0; t < T; t++) {   /* dt log-uniform in [0.001, 0.1], stored as inverse softplus */
            double dt = exp(urand(&rng_state) * (log(0.1) - log(0.001)) + log(0.001));
            L->dt_bias[t] = (real)(dt + log(-expm1(-dt)));
        }
        for (size_t i = 0; i < (size_t)T * c->p * d; i++) L->U[i] = UNI(k_u);
    }
}

/* ------------------------------------------------------------------ small math */
static real dot(const real *a, const real *b, int n) { real s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static void axpy(real *y, real a, const real *x, int n) { for (int i = 0; i < n; i++) y[i] += a * x[i]; }
static real softplus(real x) { return x > 20 ? x : log1p(exp(x)); }
static real sigm(real x) { return 1 / (1 + exp(-x)); }
static real gelu(real x) { return (real)0.5 * x * (1 + erf(x * (real)0.70710678118654752440)); }
static real dgelu(real x) { return (real)0.5 * (1 + erf(x * (real)0.70710678118654752440)) + x * exp((real)-0.5 * x * x) * (real)0.39894228040143267794; }
static void rmsnorm_fwd(const real *x, const real *w, real *u, int d, real *r) {
    real s = 0; for (int i = 0; i < d; i++) s += x[i] * x[i];
    *r = 1 / sqrt(s / d + (real)1e-6);
    for (int i = 0; i < d; i++) u[i] = x[i] * *r * w[i];
}
static void rmsnorm_bwd(const real *x, const real *w, real r, const real *du, real *dx, real *dw, int d) {
    real s = 0; for (int i = 0; i < d; i++) s += w[i] * du[i] * x[i];
    const real k = r * r * r * s / d;
    for (int i = 0; i < d; i++) { dx[i] += r * w[i] * du[i] - x[i] * k; dw[i] += du[i] * x[i] * r; }
}

/* QK-norm + rotary on the B and C parts of one projection row (2*G vectors of length n). */
static void rope_angles(long pos, int n, real *cs, real *sn) {
    for (int k = 0; k + 1 < n; k += 2) { double a = pos * pow(10000., -(double)k / n); cs[k / 2] = (real)cos(a); sn[k / 2] = (real)sin(a); }
}
static void bc_prep(const Cfg *c, real *pt, long pos, real *inv) {
    real cs[32], sn[32]; const int n = c->n;
    if (c->rope) rope_angles(pos, n, cs, sn);
    for (int g = 0; g < 2 * c->G; g++) {
        real *v = pt + (size_t)g * n;
        if (c->qknorm) { real r = 1 / sqrt(dot(v, v, n) + (real)1e-6); inv[g] = r; for (int j = 0; j < n; j++) v[j] *= r; }
        if (c->rope) for (int k = 0; k + 1 < n; k += 2) { real a = v[k], b = v[k + 1]; v[k] = a * cs[k / 2] - b * sn[k / 2]; v[k + 1] = a * sn[k / 2] + b * cs[k / 2]; }
    }
}
/* dpt: gradient w.r.t. the prepared values (in place -> gradient w.r.t. the raw values); pt: prepared values. */
static void bc_prep_bwd(const Cfg *c, const real *pt, real *dpt, long pos, const real *inv) {
    real cs[32], sn[32]; const int n = c->n;
    if (c->rope) rope_angles(pos, n, cs, sn);
    for (int g = 0; g < 2 * c->G; g++) {
        const real *v = pt + (size_t)g * n; real *dv = dpt + (size_t)g * n, u[64];
        memcpy(u, v, n * sizeof(real));
        if (c->rope) for (int k = 0; k + 1 < n; k += 2) {
            real a = dv[k], b = dv[k + 1]; dv[k] = a * cs[k / 2] + b * sn[k / 2]; dv[k + 1] = -a * sn[k / 2] + b * cs[k / 2];
            real x = u[k], y = u[k + 1]; u[k] = x * cs[k / 2] + y * sn[k / 2]; u[k + 1] = -x * sn[k / 2] + y * cs[k / 2];
        }
        if (c->qknorm) { real r = inv[g], dd = dot(u, dv, n); for (int j = 0; j < n; j++) dv[j] = r * (dv[j] - u[j] * dd); }
    }
}

/* ------------------------------------------------------------------ recurrent state and tape */
typedef struct {            /* per layer: node states, conv history, stream position */
    real *H;                /* T*N*n*p */
    real *hist;             /* (K-1)*d past normalized inputs, newest first */
    long pos;
} LState;
typedef struct {            /* per layer, one sequence of Lmax positions */
    real *xin, *u, *rn, *m, *pp, *inv, *dt, *Racc;
    int *node; real *s, *dec, *hold, *R;   /* per visit v = (t*T + tr)*(D+1) + level */
} Tape;
typedef struct {
    int Lmax; LState *st; Tape *tp;
    real *x, *g, *lam, *dpp, *dm, *dRacc, *du;
    Params G;               /* this thread's gradient accumulator */
} Work;

static void state_reset(const Cfg *c, LState *s) {
    memset(s->H, 0, sizeof(real) * (size_t)c->T * NODES(c) * c->n * c->p);
    memset(s->hist, 0, sizeof(real) * (size_t)(c->K > 1 ? c->K - 1 : 1) * c->d);
    s->pos = 0;
}
static Work *work_new(const Cfg *c, int Lmax) {
    Work *w = calloc(1, sizeof(Work));
    const size_t d = c->d, L = Lmax, VIS = L * c->T * (c->D + 1), S = (size_t)c->T * NODES(c) * c->n * c->p;
    w->Lmax = Lmax;
    w->st = calloc(c->n_layer, sizeof(LState));
    w->tp = calloc(c->n_layer, sizeof(Tape));
    for (int l = 0; l < c->n_layer; l++) {
        w->st[l].H = calloc(S, sizeof(real)); w->st[l].hist = calloc((c->K > 1 ? c->K - 1 : 1) * d, sizeof(real));
        Tape *t = w->tp + l;
        t->xin = calloc(L * d, sizeof(real)); t->u = calloc(L * d, sizeof(real)); t->rn = calloc(L, sizeof(real));
        t->m = calloc(L * d, sizeof(real)); t->pp = calloc(L * PDIM(c), sizeof(real)); t->inv = calloc(L * 2 * c->G, sizeof(real));
        t->dt = calloc(L * c->T, sizeof(real)); t->Racc = calloc(L * c->T * c->p, sizeof(real));
        t->node = calloc(VIS, sizeof(int)); t->s = calloc(VIS, sizeof(real)); t->dec = calloc(VIS, sizeof(real));
        t->hold = calloc(VIS * c->n * c->p, sizeof(real)); t->R = calloc(VIS * c->p, sizeof(real));
    }
    w->x = calloc(L * d, sizeof(real)); w->g = calloc(L * d, sizeof(real)); w->lam = calloc(S, sizeof(real));
    w->dpp = calloc(L * PDIM(c), sizeof(real)); w->dm = calloc(L * d, sizeof(real));
    w->dRacc = calloc(L * c->T * c->p, sizeof(real)); w->du = calloc(L * d, sizeof(real));
    params_alloc(c, &w->G);
    return w;
}

/* ------------------------------------------------------------------ SSM-tree block, forward */
/* Runs L positions through layer l, updating x in place (residual) and the layer state st.
 * With a tape, records everything the backward pass needs (st must start at the tape's sequence start). */
static void layer_fwd(const Cfg *c, const Layer *P, int L, real *x, LState *st, Tape *tp) {
    const int d = c->d, T = c->T, D = c->D, n = c->n, pv = c->p, K = c->K, Pd = PDIM(c), N = NODES(c), G = c->G;
    real u[512], m[512], pp[1024], y[512], Racc[64 * 16], inv[64], R[64];
    for (int t = 0; t < L; t++) {
        real *xt = x + (size_t)t * d, rn;
        rmsnorm_fwd(xt, P->norm, u, d, &rn);
        for (int i = 0; i < d; i++) m[i] = P->conv[i] * u[i];
        for (int j = 1; j < K; j++) { const real *past = st->hist + (size_t)(j - 1) * d; for (int i = 0; i < d; i++) m[i] += P->conv[(size_t)j * d + i] * past[i]; }
        for (int r = 0; r < Pd; r++) pp[r] = dot(P->proj + (size_t)r * d, m, d);
        bc_prep(c, pp, st->pos, inv);
        const real *B0 = pp, *C0 = pp + G * n, *Vv = pp + 2 * G * n, *Z = Vv + pv, *dtz = Z + pv;
        if (tp) {
            memcpy(tp->xin + (size_t)t * d, xt, d * sizeof(real)); memcpy(tp->u + (size_t)t * d, u, d * sizeof(real)); tp->rn[t] = rn;
            memcpy(tp->m + (size_t)t * d, m, d * sizeof(real)); memcpy(tp->pp + (size_t)t * Pd, pp, Pd * sizeof(real));
            memcpy(tp->inv + (size_t)t * 2 * G, inv, 2 * G * sizeof(real));
        }
        memset(y, 0, d * sizeof(real)); memset(Racc, 0, (size_t)T * pv * sizeof(real));
        for (int tr = 0; tr < T; tr++) {
            const real dt = softplus(dtz[tr] + P->dt_bias[tr]);
            const real *B = B0 + (size_t)(tr * G / T) * n, *C = C0 + (size_t)(tr * G / T) * n;
            if (tp) tp->dt[(size_t)t * T + tr] = dt;
            int node = 0;
            for (int k = 0; k <= D; k++) {
                const int row = tr * N + node;
                real *H = st->H + (size_t)row * n * pv;
                const real dec = exp(-exp(P->A_log[row]) * dt);
                const size_t v = ((size_t)t * T + tr) * (D + 1) + k;
                if (tp) memcpy(tp->hold + v * n * pv, H, (size_t)n * pv * sizeof(real));
                for (int q = 0; q < pv; q++) R[q] = 0;
                for (int j = 0; j < n; j++) for (int q = 0; q < pv; q++) { H[j * pv + q] *= dec; R[q] += C[j] * H[j * pv + q]; }
                for (int j = 0; j < n; j++) for (int q = 0; q < pv; q++) H[j * pv + q] += dt * B[j] * Vv[q];
                const real s = dot(P->w_in + (size_t)row * d, m, d) + P->bias[row] + dot(R, Z, pv);
                axpy(y, gelu(s), P->w_out + (size_t)row * d, d);
                for (int q = 0; q < pv; q++) Racc[tr * pv + q] += R[q];
                if (tp) { tp->node[v] = row; tp->s[v] = s; tp->dec[v] = dec; memcpy(tp->R + v * pv, R, pv * sizeof(real)); }
                node = 2 * node + 1 + (s > 0);
            }
        }
        for (int r = 0; r < T * pv; r++) axpy(y, Racc[r], P->U + (size_t)r * d, d);
        if (tp) memcpy(tp->Racc + (size_t)t * T * pv, Racc, (size_t)T * pv * sizeof(real));
        for (int i = 0; i < d; i++) xt[i] += y[i];
        if (K > 1) { memmove(st->hist + d, st->hist, sizeof(real) * (size_t)(K - 2) * d); memcpy(st->hist, u, d * sizeof(real)); }
        st->pos++;
    }
}

/* ------------------------------------------------------------------ SSM-tree block, backward */
/* g: on entry dL/d(layer output), on exit dL/d(layer input). Assumes the sequence started from zero state. */
static void layer_bwd(const Cfg *c, const Layer *P, Layer *Gd, int L, real *g, const Tape *tp, Work *w) {
    const int d = c->d, T = c->T, D = c->D, n = c->n, pv = c->p, K = c->K, Pd = PDIM(c), N = NODES(c), G = c->G;
    real *lam = w->lam, *dpp = w->dpp, *dm = w->dm, *dRacc = w->dRacc, *du = w->du;
    memset(lam, 0, sizeof(real) * (size_t)T * N * n * pv);
    memset(dpp, 0, sizeof(real) * (size_t)L * Pd); memset(dm, 0, sizeof(real) * (size_t)L * d); memset(du, 0, sizeof(real) * (size_t)L * d);
    /* y = ... + U^T Racc */
    for (int t = 0; t < L; t++) for (int r = 0; r < T * pv; r++) {
        dRacc[(size_t)t * T * pv + r] = dot(P->U + (size_t)r * d, g + (size_t)t * d, d);
        axpy(Gd->U + (size_t)r * d, tp->Racc[(size_t)t * T * pv + r], g + (size_t)t * d, d);
    }
    real dh1[64 * 16];
    for (int t = L - 1; t >= 0; t--) {
        const real *gy = g + (size_t)t * d, *m = tp->m + (size_t)t * d, *pp = tp->pp + (size_t)t * Pd;
        real *dpt = dpp + (size_t)t * Pd, *dmt = dm + (size_t)t * d;
        const real *Vv = pp + 2 * G * n, *Z = Vv + pv;
        real *dV = dpt + 2 * G * n, *dZ = dV + pv, *ddtz = dZ + pv;
        for (int tr = 0; tr < T; tr++) {
            const real dt = tp->dt[(size_t)t * T + tr];
            const real *B = pp + (size_t)(tr * G / T) * n, *C = pp + G * n + (size_t)(tr * G / T) * n;
            real *dB = dpt + (size_t)(tr * G / T) * n, *dC = dpt + G * n + (size_t)(tr * G / T) * n;
            real ddt = 0;
            for (int k = D; k >= 0; k--) {
                const size_t v = ((size_t)t * T + tr) * (D + 1) + k;
                const int row = tp->node[v];
                const real s = tp->s[v], dec = tp->dec[v], *R = tp->R + v * pv, *hold = tp->hold + v * n * pv;
                /* y += gelu(s) w_out */
                const real ds = dgelu(s) * dot(P->w_out + (size_t)row * d, gy, d);
                axpy(Gd->w_out + (size_t)row * d, gelu(s), gy, d);
                /* s = w_in.m + bias + R.Z ; Racc += R */
                axpy(Gd->w_in + (size_t)row * d, ds, m, d); axpy(dmt, ds, P->w_in + (size_t)row * d, d);
                Gd->bias[row] += ds;
                real dR[64];
                for (int q = 0; q < pv; q++) { dR[q] = ds * Z[q] + dRacc[(size_t)t * T * pv + tr * pv + q]; dZ[q] += ds * R[q]; }
                /* state: h1 = dec*hold; R = C^T h1; hnew = h1 + dt B V^T ; lam = dL/dhnew */
                real *lr = lam + (size_t)row * n * pv, ddec = 0;
                for (int j = 0; j < n; j++) {
                    real bv = 0, dcj = 0;
                    for (int q = 0; q < pv; q++) {
                        const real l = lr[j * pv + q], h1 = dec * hold[j * pv + q];
                        bv += l * Vv[q];
                        dV[q] += dt * l * B[j];
                        dcj += dR[q] * h1;
                        dh1[j * pv + q] = l + C[j] * dR[q];
                        ddec += dh1[j * pv + q] * hold[j * pv + q];
                    }
                    ddt += bv * B[j];
                    dB[j] += dt * bv;
                    dC[j] += dcj;
                }
                for (int i = 0; i < n * pv; i++) lr[i] = dec * dh1[i];
                /* dec = exp(-a dt), a = exp(A_log) */
                const real a = exp(P->A_log[row]);
                Gd->A_log[row] += ddec * dec * (-a * dt);
                ddt += ddec * dec * (-a);
            }
            const real dz = ddt * sigm(pp[2 * G * n + 2 * pv + tr] + P->dt_bias[tr]);
            ddtz[tr] += dz; Gd->dt_bias[tr] += dz;
        }
    }
    for (int t = 0; t < L; t++) {
        real *dpt = dpp + (size_t)t * Pd;
        bc_prep_bwd(c, tp->pp + (size_t)t * Pd, dpt, t, tp->inv + (size_t)t * 2 * G);
        for (int r = 0; r < Pd; r++) { axpy(Gd->proj + (size_t)r * d, dpt[r], tp->m + (size_t)t * d, d); axpy(dm + (size_t)t * d, dpt[r], P->proj + (size_t)r * d, d); }
    }
    /* m_t = sum_j conv[j] * u_{t-j} */
    for (int t = 0; t < L; t++) for (int j = 0; j < K && j <= t; j++) {
        const real *ut = tp->u + (size_t)(t - j) * d, *dmt = dm + (size_t)t * d;
        for (int i = 0; i < d; i++) { Gd->conv[(size_t)j * d + i] += dmt[i] * ut[i]; du[(size_t)(t - j) * d + i] += P->conv[(size_t)j * d + i] * dmt[i]; }
    }
    for (int t = 0; t < L; t++) rmsnorm_bwd(tp->xin + (size_t)t * d, P->norm, tp->rn[t], du + (size_t)t * d, g + (size_t)t * d, Gd->norm, d);
}

/* ------------------------------------------------------------------ head */
static real logits_of(const Cfg *c, const Params *P, const real *x, real *lg) {
    real u[512], r, mx = -INFINITY;
    rmsnorm_fwd(x, P->norm_f, u, c->d, &r);
    for (int v = 0; v < V; v++) { lg[v] = dot(P->embed + (size_t)v * c->d, u, c->d); mx = lg[v] > mx ? lg[v] : mx; }
    return mx;
}
/* Cross-entropy at one position; accumulates dL/dx into gx and head gradients into Gd. */
static real head_train(const Cfg *c, const Params *P, Params *Gd, const real *x, int y, real *gx) {
    const int d = c->d; real u[512], du[512], r, lg[V], mx = -INFINITY; double z = 0;
    rmsnorm_fwd(x, P->norm_f, u, d, &r);
    for (int v = 0; v < V; v++) { lg[v] = dot(P->embed + (size_t)v * d, u, d); mx = lg[v] > mx ? lg[v] : mx; }
    for (int v = 0; v < V; v++) z += exp(lg[v] - mx);
    memset(du, 0, d * sizeof(real));
    for (int v = 0; v < V; v++) {
        const real pr = (real)(exp(lg[v] - mx) / z) - (v == y);
        axpy(Gd->embed + (size_t)v * d, pr, u, d); axpy(du, pr, P->embed + (size_t)v * d, d);
    }
    rmsnorm_bwd(x, P->norm_f, r, du, gx, Gd->norm_f, d);
    return (real)(log(z) + mx - lg[y]);
}

/* ------------------------------------------------------------------ one sequence: forward + backward */
/* labels[t] < 0 are not scored. Returns summed loss (nats), sets *ns. With train=0 only scores. */
static double run_seq(const Cfg *c, const Params *P, const int *tok, const int *lab, int L, Work *w, int *ns, int train) {
    const int d = c->d; real *x = w->x, lg[V];
    for (int t = 0; t < L; t++) memcpy(x + (size_t)t * d, P->embed + (size_t)tok[t] * d, d * sizeof(real));
    for (int l = 0; l < c->n_layer; l++) { state_reset(c, w->st + l); layer_fwd(c, P->ly + l, L, x, w->st + l, train ? w->tp + l : NULL); }
    double loss = 0; *ns = 0;
    if (!train) {
        for (int t = 0; t < L; t++) if (lab[t] >= 0) {
            real mx = logits_of(c, P, x + (size_t)t * d, lg); double z = 0;
            for (int v = 0; v < V; v++) z += exp(lg[v] - mx);
            loss += log(z) + mx - lg[lab[t]]; (*ns)++;
        }
        return loss;
    }
    memset(w->g, 0, sizeof(real) * (size_t)L * d);
    for (int t = 0; t < L; t++) if (lab[t] >= 0) { loss += head_train(c, P, &w->G, x + (size_t)t * d, lab[t], w->g + (size_t)t * d); (*ns)++; }
    for (int l = c->n_layer - 1; l >= 0; l--) layer_bwd(c, P->ly + l, w->G.ly + l, L, w->g, w->tp + l, w);
    for (int t = 0; t < L; t++) axpy(w->G.embed + (size_t)tok[t] * d, 1, w->g + (size_t)t * d, d);
    return loss;
}

/* ------------------------------------------------------------------ optimizers */
typedef struct { real lr, beta1, beta2, eps, wd, clip; long step; int muon; } Hyper;
static void adamw(real *p, const real *g, real *m, real *v, size_t n, const Hyper *h, real scale) {
    const real c1 = 1 - pow(h->beta1, h->step), c2 = 1 - pow(h->beta2, h->step);
    for (size_t i = 0; i < n; i++) {
        const real gi = g[i] * scale;
        m[i] = h->beta1 * m[i] + (1 - h->beta1) * gi; v[i] = h->beta2 * v[i] + (1 - h->beta2) * gi * gi;
        p[i] -= h->lr * (h->wd * p[i] + (m[i] / c1) / (sqrt(v[i] / c2) + h->eps));
    }
}
/* Blockwise Muon on a matrix of `rows` x d: blocks of 16 rows, Nesterov momentum 0.95, 5 Newton-Schulz
 * steps on the block; rows with zero gradient (unvisited tree nodes) are skipped. */
static void muon(real *p, const real *g, real *mom, int rows, int d, const Hyper *h, real scale) {
    enum { BR = 16 }; real X[BR * 512], Y[BR * 512], A[BR * BR], Bm[BR * BR]; int map[BR];
    for (int r0 = 0; r0 < rows; r0 += BR) {
        int r = 0; double norm = 0;
        for (int i = r0; i < r0 + BR && i < rows; i++) {
            const real *gi = g + (size_t)i * d; int hit = 0; for (int k = 0; k < d; k++) if (gi[k] != 0) { hit = 1; break; }
            if (!hit) continue;
            map[r] = i;
            for (int k = 0; k < d; k++) {
                const size_t o = (size_t)i * d + k; const real gg = gi[k] * scale;
                mom[o] = (real).95 * mom[o] + (real).05 * gg;
                X[r * d + k] = (real).95 * mom[o] + (real).05 * gg; norm += (double)X[r * d + k] * X[r * d + k];
            }
            r++;
        }
        if (!r) continue;
        const real inv = (real)(1 / (sqrt(norm) + 1e-7)); for (int i = 0; i < r * d; i++) X[i] *= inv;
        for (int it = 0; it < 5; it++) {
            for (int i = 0; i < r; i++) for (int k = 0; k <= i; k++) A[i * r + k] = A[k * r + i] = dot(X + i * d, X + k * d, d);
            for (int i = 0; i < r; i++) for (int k = 0; k < r; k++) Bm[i * r + k] = (real)-4.775 * A[i * r + k] + (real)2.0315 * dot(A + i * r, A + k * r, r);
            for (int i = 0; i < r; i++) { for (int k = 0; k < d; k++) Y[i * d + k] = (real)3.4445 * X[i * d + k]; for (int k = 0; k < r; k++) axpy(Y + i * d, Bm[i * r + k], X + k * d, d); }
            memcpy(X, Y, (size_t)r * d * sizeof(real));
        }
        const real lr = h->lr * 3 * (real)(.2 * sqrt(fmax(r, d)));
        for (int i = 0; i < r; i++) { real *pi = p + (size_t)map[i] * d; for (int k = 0; k < d; k++) pi[k] -= h->lr * h->wd * pi[k] + lr * X[i * d + k]; }
    }
}
typedef struct { Params m, v; } OptState;
static void opt_step(const Cfg *c, Params *P, const Params *Gs, OptState *o, const Hyper *h, real scale) {
    if (!h->muon) { adamw(P->all, Gs->all, o->m.all, o->v.all, P->size, h, scale); return; }
    /* Muon for the matrices (tree-node rows, proj, U); AdamW for everything else. */
    const int d = c->d, TN = c->T * NODES(c);
    adamw(P->embed, Gs->embed, o->m.embed, o->v.embed, (size_t)V * d, h, scale);
    adamw(P->norm_f, Gs->norm_f, o->m.norm_f, o->v.norm_f, d, h, scale);
    for (int l = 0; l < c->n_layer; l++) {
        Layer *p = P->ly + l, *m = o->m.ly + l, *v = o->v.ly + l; const Layer *g = Gs->ly + l;
        muon(p->w_in, g->w_in, m->w_in, TN, d, h, scale); muon(p->w_out, g->w_out, m->w_out, TN, d, h, scale);
        muon(p->proj, g->proj, m->proj, PDIM(c), d, h, scale); muon(p->U, g->U, m->U, c->T * c->p, d, h, scale);
        adamw(p->norm, g->norm, m->norm, v->norm, d, h, scale); adamw(p->conv, g->conv, m->conv, v->conv, (size_t)c->K * d, h, scale);
        adamw(p->bias, g->bias, m->bias, v->bias, TN, h, scale); adamw(p->A_log, g->A_log, m->A_log, v->A_log, TN, h, scale);
        adamw(p->dt_bias, g->dt_bias, m->dt_bias, v->dt_bias, c->T, h, scale);
    }
}

/* ------------------------------------------------------------------ data, eval, sampling, I/O */
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static unsigned char *read_file(const char *path, long *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f);
    unsigned char *b = malloc(*n); if (fread(b, 1, *n, f) != (size_t)*n) { fclose(f); free(b); return NULL; }
    fclose(f); return b;
}
/* Validation BPB over consecutive windows of L bytes; the first `burn` positions of each window are context only. */
static double eval_bpb(const Cfg *c, const Params *P, const unsigned char *raw, long a, long b, int L, int burn, Work **ws, int threads) {
    long nw = (b - a - 1) / L; double loss = 0; long tot = 0;
#pragma omp parallel for num_threads(threads) reduction(+:loss, tot) schedule(static)
    for (long w = 0; w < nw; w++) {
#ifdef _OPENMP
        int th = omp_get_thread_num();
#else
        int th = 0;
#endif
        int tok[4096], lab[4096], ns;
        for (int i = 0; i < L; i++) { tok[i] = raw[a + w * L + i]; lab[i] = i < burn ? -1 : raw[a + w * L + i + 1]; }
        loss += run_seq(c, P, tok, lab, L, ws[th], &ns, 0); tot += ns;
    }
    return loss / tot / log(2.);
}
static void sample(const Cfg *c, const Params *P, Work *w, const char *prompt, int count, real temp, uint64_t seed) {
    real *x = w->x, lg[V];
    for (int l = 0; l < c->n_layer; l++) state_reset(c, w->st + l);
    const char *pr = prompt && *prompt ? prompt : "\n"; int cur = (unsigned char)pr[0]; fputs(pr, stdout);
    for (int i = 1, total = (int)strlen(pr) + count; i < total; i++) {
        memcpy(x, P->embed + (size_t)cur * c->d, c->d * sizeof(real));
        for (int l = 0; l < c->n_layer; l++) layer_fwd(c, P->ly + l, 1, x, w->st + l, NULL);
        if (i < (int)strlen(pr)) { cur = (unsigned char)pr[i]; continue; }
        real mx = logits_of(c, P, x, lg); double z = 0, pv[V];
        for (int v = 0; v < V; v++) { pv[v] = exp((lg[v] - mx) / temp); z += pv[v]; }
        double r = urand(&seed) * z; cur = V - 1;
        for (int v = 0; v < V; v++) { r -= pv[v]; if (r <= 0) { cur = v; break; } }
        putchar(cur);
    }
    putchar('\n');
}
static int save(const char *path, const Cfg *c, const Params *P) {
    FILE *f = fopen(path, "wb"); if (!f) return 1;
    int ok = fwrite("SSMT", 1, 4, f) == 4 && fwrite(c, sizeof(*c), 1, f) == 1;
    for (size_t i = 0; ok && i < P->size; i++) { float v = (float)P->all[i]; ok = fwrite(&v, 4, 1, f) == 1; }
    fclose(f); return !ok;
}
static int load(const char *path, Cfg *c, Params *P) {
    FILE *f = fopen(path, "rb"); char mg[4]; if (!f) return 1;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "SSMT", 4) || fread(c, sizeof(*c), 1, f) != 1) { fclose(f); return 1; }
    params_alloc(c, P);
    for (size_t i = 0; i < P->size; i++) { float v; if (fread(&v, 4, 1, f) != 1) { fclose(f); return 1; } P->all[i] = v; }
    fclose(f); return 0;
}

/* ------------------------------------------------------------------ gradient check */
/* Central differences on random parameters of a tiny model (build with -DDOUBLE). Hard routing makes the
 * loss piecewise smooth, so a perturbation that flips a route is reported and skipped. */
static int check(void) {
    Cfg c = {.d = 16, .n_layer = 2, .T = 2, .D = 2, .n = 4, .p = 4, .K = 3, .G = 2, .qknorm = 1, .rope = 1};
    const int L = 12; Params P; params_alloc(&c, &P); params_init(&c, &P);
    for (size_t i = 0; i < P.size; i++) P.all[i] += (real)(0.05 * randn());   /* move off the symmetric init */
    for (int l = 0; l < c.n_layer; l++) for (int t = 0; t < c.T; t++) P.ly[l].dt_bias[t] = (real)0.3;
    Work *w = work_new(&c, L); int tok[64], lab[64], ns;
    for (int t = 0; t < L; t++) { tok[t] = (int)(urand(&rng_state) * V); lab[t] = (int)(urand(&rng_state) * V); }
    memset(w->G.all, 0, w->G.size * sizeof(real));
    run_seq(&c, &P, tok, lab, L, w, &ns, 1);
    double worst = 0; int tested = 0, flips = 0; Work *w2 = work_new(&c, L);
    const size_t nv = (size_t)L * c.T * (c.D + 1); int *routes = malloc(sizeof(int) * nv * c.n_layer);
    for (int trial = 0; trial < 400; trial++) {
        size_t i = (size_t)(urand(&rng_state) * P.size); const real e = (real)1e-5, old = P.all[i];
        P.all[i] = old + e; double lp = run_seq(&c, &P, tok, lab, L, w2, &ns, 1);
        for (int l = 0; l < c.n_layer; l++) memcpy(routes + l * nv, w2->tp[l].node, sizeof(int) * nv);
        P.all[i] = old - e; double lm = run_seq(&c, &P, tok, lab, L, w2, &ns, 1);
        int flip = 0; for (int l = 0; l < c.n_layer; l++) flip |= memcmp(routes + l * nv, w2->tp[l].node, sizeof(int) * nv) != 0;
        P.all[i] = old;
        if (flip) { flips++; continue; }
        const double num = (lp - lm) / (2 * e), ana = w->G.all[i], err = fabs(num - ana) / fmax(1e-6, fabs(num) + fabs(ana));
        if (fabs(num) + fabs(ana) > 1e-7) { tested++; if (err > worst) worst = err; }
    }
    printf("gradient check: %d parameters compared, %d skipped for route flips, worst relative error %.3g\n", tested, flips, worst);
    return worst < 1e-4 ? 0 : 1;
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    Cfg c = {.d = 128, .n_layer = 4, .T = 4, .D = 7, .n = 16, .p = 16, .K = 3, .G = 4, .qknorm = 1, .rope = 1};
    const char *data = NULL, *out = NULL, *in = NULL, *prompt = "\n";
    int threads = 4, B = 16, L = 256, burn = 64, nsample = 0, seed = 0; long steps = 0;
    double seconds = 300, lr = 0.003; real temp = 0.8; int use_muon = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i]; const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define OPT(name, dst, conv) if (!strcmp(a, name) && v) { dst = conv(v); i++; continue; }
        OPT("--data", data, (const char *)) OPT("--save", out, (const char *)) OPT("--load", in, (const char *)) OPT("--prompt", prompt, (const char *))
        OPT("--threads", threads, atoi) OPT("--batch", B, atoi) OPT("--seq", L, atoi) OPT("--burn", burn, atoi) OPT("--seed", seed, atoi)
        OPT("--seconds", seconds, atof) OPT("--steps", steps, atol) OPT("--lr", lr, atof) OPT("--temperature", temp, atof) OPT("--sample", nsample, atoi)
        OPT("--width", c.d, atoi) OPT("--layers", c.n_layer, atoi) OPT("--trees", c.T, atoi) OPT("--depth", c.D, atoi) OPT("--state-n", c.n, atoi)
        OPT("--state-p", c.p, atoi) OPT("--conv", c.K, atoi) OPT("--groups", c.G, atoi) OPT("--qknorm", c.qknorm, atoi) OPT("--rope", c.rope, atoi)
        OPT("--muon", use_muon, atoi)
#undef OPT
        if (!strcmp(a, "--check")) return check();
        fprintf(stderr, "unknown or incomplete argument %s\n", a); return 2;
    }
    rng_state ^= (uint64_t)(seed + 1) * 0x9E3779B97F4A7C15ULL;
    Params P;
    if (in) { if (load(in, &c, &P)) { fprintf(stderr, "cannot load %s\n", in); return 1; } }
    if (c.d > 512 || c.T * c.p > 1024 || c.p > 64 || c.n * c.p > 1024 || c.n > 64 || PDIM(&c) > 1024 || c.K < 1 || c.G < 1 || c.T % c.G || L > 4096) { fprintf(stderr, "configuration out of bounds\n"); return 2; }
    if (!in) { params_alloc(&c, &P); params_init(&c, &P); }
    Work **ws = calloc(threads, sizeof(Work *)); for (int t = 0; t < threads; t++) ws[t] = work_new(&c, L);
    printf("{\"event\":\"config\",\"parameters\":%zu,\"d\":%d,\"layers\":%d,\"trees\":%d,\"depth\":%d,\"state\":\"%dx%d\",\"groups\":%d,\"qknorm\":%d,\"rope\":%d,\"optimizer\":\"%s\"}\n",
           P.size, c.d, c.n_layer, c.T, c.D, c.n, c.p, c.G, c.qknorm, c.rope, use_muon ? "muon+adamw" : "adamw");
    if (!data) { if (nsample > 0) sample(&c, &P, ws[0], prompt, nsample, temp, (uint64_t)seed * 7 + 1); return 0; }

    long n; unsigned char *raw = read_file(data, &n); if (!raw) { fprintf(stderr, "cannot read %s\n", data); return 1; }
    const long split = (long)(n * .9); if (split < L + 2 || n - split < L + 2) { fprintf(stderr, "data too short\n"); return 2; }
    OptState o; params_alloc(&c, &o.m); params_alloc(&c, &o.v); Params Gs; params_alloc(&c, &Gs);
    Hyper h = {.beta1 = .9, .beta2 = .98, .eps = 1e-8, .wd = .01, .clip = 1, .muon = use_muon};
    int *bt = malloc(sizeof(int) * (size_t)B * L), *by = malloc(sizeof(int) * (size_t)B * L);
    uint64_t drng = ((uint64_t)seed + 1) * 0xD1B54A32D192ED03ULL;
    double t0 = now(), elapsed = 0, max_step = 0, loss = 0, nextlog = 60; long step = 0;
    while (steps ? step < steps : elapsed + 1.25 * max_step < seconds) {
        step++; h.step = step;
        const double prog = steps ? (double)(step - 1) / fmax(1, steps - 1) : fmin(1, elapsed / seconds);
        const double warm = steps ? fmin(1, step / 100.) : fmin(1, (elapsed + .1) / fmin(10, seconds * .1));
        h.lr = (real)(lr * warm * (.1 + .9 * .5 * (1 + cos(3.141592653589793 * prog))));
        for (int b = 0; b < B; b++) { long at = (long)(urand(&drng) * (split - L - 1)); for (int i = 0; i < L; i++) { bt[b * L + i] = raw[at + i]; by[b * L + i] = raw[at + i + 1]; } }
        const double st = now(); double lsum = 0; long nsc = 0;
#pragma omp parallel for num_threads(threads) reduction(+:lsum, nsc) schedule(static)
        for (int b = 0; b < B; b++) {
#ifdef _OPENMP
            int th = omp_get_thread_num();
#else
            int th = 0;
#endif
            int ns; lsum += run_seq(&c, &P, bt + (size_t)b * L, by + (size_t)b * L, L, ws[th], &ns, 1); nsc += ns;
        }
        /* reduce thread gradients, mean over scored bytes, clip by global norm */
        double gn = 0;
#pragma omp parallel for num_threads(threads) reduction(+:gn) schedule(static)
        for (size_t i = 0; i < Gs.size; i++) { real s = 0; for (int t = 0; t < threads; t++) { s += ws[t]->G.all[i]; ws[t]->G.all[i] = 0; } Gs.all[i] = s; gn += (double)s * s; }
        real scale = (real)(1. / nsc); gn = sqrt(gn) * scale; if (gn > h.clip) scale *= (real)(h.clip / (gn + 1e-6));
        opt_step(&c, &P, &Gs, &o, &h, scale);
        loss = lsum / nsc;
        const double sd = now() - st; if (sd > max_step) max_step = sd;
        elapsed = now() - t0;
        if (elapsed >= nextlog) { printf("{\"event\":\"progress\",\"step\":%ld,\"seconds\":%.1f,\"train_bpb\":%.4f}\n", step, elapsed, loss / log(2.)); fflush(stdout); nextlog += 60; }
    }
    const double val = eval_bpb(&c, &P, raw, split, n, L, burn, ws, threads);
    printf("{\"event\":\"eval\",\"step\":%ld,\"train_seconds\":%.2f,\"bytes_per_second\":%.0f,\"train_bpb\":%.4f,\"val_bpb\":%.4f}\n",
           step, elapsed, (double)step * B * L / elapsed, loss / log(2.), val);
    if (out && save(out, &c, &P)) { fprintf(stderr, "cannot save %s\n", out); return 1; }
    if (nsample > 0) sample(&c, &P, ws[0], prompt, nsample, temp, (uint64_t)seed * 7 + 1);
    return 0;
}
