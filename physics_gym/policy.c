#include "policy.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define UNFOLD 6
const char *pol_name(int t) { return t == POL_MLP ? "mlp" : t == POL_SSM ? "ssm" : "ltc"; }

static int ssm_nodes(const PolCfg *c) { return (1 << (c->depth + 1)) - 1; }
static int ssm_pdim(const PolCfg *c) { return 2 * c->n * c->trees + 2 * c->p + c->trees; }   /* one key/query group per tree */
static int ssm_layer_size(const PolCfg *c) {
    const int d = c->hidden, TN = c->trees * ssm_nodes(c);
    return d /*norm*/ + TN * (2 * d + 2) /*w_in,w_out,bias,A_log*/ + ssm_pdim(c) * d + c->trees + c->trees * c->p * d;
}

int pol_nparams(const PolCfg *c) {
    const int I = c->in, O = c->out, H = c->hidden;
    if (c->type == POL_MLP) return I * H + H + H * H + H + H * O + O;
    if (c->type == POL_LTC) return H * I + H * H + H * 3 /*b, tau, A*/ + O * H + O;
    return I * H + H + c->layers * ssm_layer_size(c) + H /*final norm*/ + O * H + O;
}

static double urand(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (*s >> 11) * (1.0 / 9007199254740992.0); }
static float uni(uint64_t *s, double k) { return (float)((2 * urand(s) - 1) * k); }

void pol_init(const PolCfg *c, float *th, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 7; const int I = c->in, O = c->out, H = c->hidden; float *q = th;
    if (c->type == POL_MLP) {
        for (int i = 0; i < I * H; i++) *q++ = uni(&s, 1 / sqrt(I)); for (int i = 0; i < H; i++) *q++ = 0;
        for (int i = 0; i < H * H; i++) *q++ = uni(&s, 1 / sqrt(H)); for (int i = 0; i < H; i++) *q++ = 0;
        for (int i = 0; i < H * O; i++) *q++ = uni(&s, 0.1 / sqrt(H)); for (int i = 0; i < O; i++) *q++ = 0;
        return;
    }
    if (c->type == POL_LTC) {
        for (int i = 0; i < H * I; i++) *q++ = uni(&s, 1 / sqrt(I));
        for (int i = 0; i < H * H; i++) *q++ = uni(&s, 1 / sqrt(H));
        for (int i = 0; i < H; i++) *q++ = 0;                                   /* b */
        for (int i = 0; i < H; i++) *q++ = (float)log(0.05 + 0.5 * urand(&s));  /* log tau (s) */
        for (int i = 0; i < H; i++) *q++ = uni(&s, 1);                          /* A: reversal potential */
        for (int i = 0; i < O * H; i++) *q++ = uni(&s, 0.1 / sqrt(H)); for (int i = 0; i < O; i++) *q++ = 0;
        return;
    }
    const int d = H, TN = c->trees * ssm_nodes(c), Pd = ssm_pdim(c);
    for (int i = 0; i < I * d; i++) *q++ = uni(&s, 1 / sqrt(I)); for (int i = 0; i < d; i++) *q++ = 0;
    for (int l = 0; l < c->layers; l++) {
        for (int i = 0; i < d; i++) *q++ = 1;                                   /* norm */
        for (int i = 0; i < TN * d; i++) *q++ = uni(&s, 1 / sqrt(d));           /* w_in */
        for (int i = 0; i < TN * d; i++) *q++ = uni(&s, 1 / sqrt(c->trees * (c->depth + 1.0)));  /* w_out */
        for (int i = 0; i < TN; i++) *q++ = uni(&s, 1 / sqrt(d));               /* bias */
        for (int i = 0; i < TN; i++) *q++ = (float)log(1 + 15 * urand(&s));     /* A_log */
        for (int i = 0; i < Pd * d; i++) *q++ = uni(&s, 1 / sqrt(d));           /* proj */
        for (int i = 0; i < c->trees; i++) { double dt = exp(urand(&s) * (log(0.5) - log(0.02)) + log(0.02)); *q++ = (float)(dt + log(-expm1(-dt))); }
        for (int i = 0; i < c->trees * c->p * d; i++) *q++ = uni(&s, 0.3 / sqrt(c->p * c->trees));   /* U */
    }
    for (int i = 0; i < d; i++) *q++ = 1;
    for (int i = 0; i < O * d; i++) *q++ = 0;   /* zero output: training starts from "hold the reference" */
    for (int i = 0; i < O; i++) *q++ = 0;
}

PolState *pol_state_new(const PolCfg *c) {
    PolState *s = calloc(1, sizeof(PolState));
    size_t nh = c->type == POL_SSM ? (size_t)c->layers * c->trees * ssm_nodes(c) * c->n * c->p : c->type == POL_LTC ? c->hidden : 1;
    s->h = calloc(nh, sizeof(float)); s->x = calloc(1024, sizeof(float));
    return s;
}
void pol_reset(const PolCfg *c, PolState *s) {
    size_t nh = c->type == POL_SSM ? (size_t)c->layers * c->trees * ssm_nodes(c) * c->n * c->p : c->type == POL_LTC ? c->hidden : 1;
    memset(s->h, 0, nh * sizeof(float));
}

static float dotf(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static float sigm(float x) { return 1 / (1 + expf(-x)); }
static float softplusf(float x) { return x > 20 ? x : log1pf(expf(x)); }
static float geluf(float x) { return 0.5f * x * (1 + erff(x * 0.70710678f)); }
static void rmsn(const float *x, const float *w, float *u, int d) { float s = 0; for (int i = 0; i < d; i++) s += x[i] * x[i]; float r = 1 / sqrtf(s / d + 1e-6f); for (int i = 0; i < d; i++) u[i] = x[i] * r * w[i]; }

void pol_act(const PolCfg *c, const float *th, PolState *st, const float *o, float *a) {
    const int I = c->in, O = c->out, H = c->hidden; const float *q = th;
    float x[256], y[256], u[256];
    if (c->type == POL_MLP) {
        const float *W1 = q, *b1 = W1 + I * H, *W2 = b1 + H, *b2 = W2 + H * H, *W3 = b2 + H, *b3 = W3 + H * O;
        for (int i = 0; i < H; i++) x[i] = tanhf(dotf(W1 + i * I, o, I) + b1[i]);
        for (int i = 0; i < H; i++) y[i] = tanhf(dotf(W2 + i * H, x, H) + b2[i]);
        for (int k = 0; k < O; k++) a[k] = tanhf(dotf(W3 + k * H, y, H) + b3[k]);
        return;
    }
    if (c->type == POL_LTC) {
        const float *Wu = q, *Wx = Wu + H * I, *b = Wx + H * H, *ltau = b + H, *A = ltau + H, *Wo = A + H, *bo = Wo + O * H;
        float uin[256], *h = st->h; const float dt = 1.0f / (30 * UNFOLD);
        for (int i = 0; i < H; i++) uin[i] = dotf(Wu + i * I, o, I) + b[i];
        for (int k = 0; k < UNFOLD; k++) {
            for (int i = 0; i < H; i++) {
                const float f = sigm(dotf(Wx + i * H, h, H) + uin[i]), inv_tau = expf(-ltau[i]);
                y[i] = (h[i] + dt * f * A[i]) / (1 + dt * (inv_tau + f));
            }
            memcpy(h, y, H * sizeof(float));
        }
        for (int k = 0; k < O; k++) a[k] = tanhf(dotf(Wo + k * H, h, H) + bo[k]);
        return;
    }
    /* SSM-tree */
    const int d = H, T = c->trees, D = c->depth, n = c->n, pv = c->p, N = ssm_nodes(c), TN = T * N, Pd = ssm_pdim(c);
    const float *Win = q, *bin = Win + I * d; q = bin + d;
    for (int i = 0; i < d; i++) x[i] = dotf(Win + i * I, o, I) + bin[i];
    float pp[512], R[64], Racc[512];
    for (int l = 0; l < c->layers; l++) {
        const float *norm = q, *w_in = norm + d, *w_out = w_in + TN * d, *bias = w_out + TN * d, *A_log = bias + TN, *proj = A_log + TN,
                    *dt_bias = proj + Pd * d, *U = dt_bias + T;
        q = U + T * pv * d;
        float *Hs = st->h + (size_t)l * TN * n * pv;
        rmsn(x, norm, u, d);
        for (int r = 0; r < Pd; r++) pp[r] = dotf(proj + r * d, u, d);
        for (int g = 0; g < 2 * T; g++) { float *v = pp + g * n, ss = dotf(v, v, n), rr = 1 / sqrtf(ss + 1e-6f); for (int j = 0; j < n; j++) v[j] *= rr; }   /* QK-norm */
        const float *Vv = pp + 2 * T * n, *Z = Vv + pv, *dtz = Z + pv;
        memset(y, 0, d * sizeof(float)); memset(Racc, 0, T * pv * sizeof(float));
        for (int tr = 0; tr < T; tr++) {
            const float dt = softplusf(dtz[tr] + dt_bias[tr]), *B = pp + tr * n, *C = pp + T * n + tr * n;
            int node = 0;
            for (int k = 0; k <= D; k++) {
                const int row = tr * N + node; float *Hn = Hs + (size_t)row * n * pv; const float dec = expf(-expf(A_log[row]) * dt);
                for (int qq = 0; qq < pv; qq++) R[qq] = 0;
                for (int j = 0; j < n; j++) for (int qq = 0; qq < pv; qq++) { Hn[j * pv + qq] *= dec; R[qq] += C[j] * Hn[j * pv + qq]; }
                for (int j = 0; j < n; j++) for (int qq = 0; qq < pv; qq++) Hn[j * pv + qq] += dt * B[j] * Vv[qq];
                const float s = dotf(w_in + row * d, u, d) + bias[row] + dotf(R, Z, pv), g = geluf(s);
                for (int i = 0; i < d; i++) y[i] += g * w_out[row * d + i];
                for (int qq = 0; qq < pv; qq++) Racc[tr * pv + qq] += R[qq];
                node = 2 * node + 1 + (s > 0);
            }
        }
        for (int r = 0; r < T * pv; r++) for (int i = 0; i < d; i++) y[i] += Racc[r] * U[r * d + i];
        for (int i = 0; i < d; i++) x[i] += y[i];
    }
    const float *nf = q, *Wo = nf + d, *bo = Wo + O * d;
    rmsn(x, nf, u, d);
    for (int k = 0; k < O; k++) a[k] = tanhf(dotf(Wo + k * d, u, d) + bo[k]);
}
