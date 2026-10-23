/* ppo.c: PPO with backpropagation and Muon for the SSM-tree body controller.
 *
 * The policy network is the same SSM-tree controller as policy.c (input projection, SSM-tree blocks, final norm,
 * tanh head), but trained by gradient: its blocks are standalone/ssm_tree.c's layer forward/backward (gradient-
 * checked), and its matrices are updated by that file's blockwise Muon, everything else by AdamW.
 *
 *   policy:  mu = tanh(Wo rmsnorm(SSM-tree(Win obs_norm + bin)) + bo), a ~ N(mu, exp(log_std)^2), per joint dof
 *   critic:  V(obs) = MLP 216 -> 128 -> 128 -> 1 (tanh), AdamW
 *   rollout: NENV environments x HOR steps per iteration on all threads, the recurrent state carried across
 *            iterations; GAE(gamma, lambda); a fall ends an episode with value 0, the time limit bootstraps
 *   update:  EPOCHS passes over the chunks, MB minibatches each; the clipped surrogate is backpropagated through
 *            time over each chunk (split where an episode ended), starting from the state stored at its start
 *
 *   ./ppo --task walk --assist 0.07 --load walk.pol --seconds 600 --out walk_ppo.pol
 *   ./ppo --check      (finite-difference check of the policy gradient; build with -DDOUBLE)
 * The output is a policy.c .pol file (the width-1 convolution is folded into the weights), so the viewer runs it.
 */
#include "env.h"
#include "policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define main ssm_tree_unused_main
#include "../standalone/ssm_tree.c"
#undef main

/* ------------------------------------------------------------------ the policy network */
typedef struct {
    Cfg c;                 /* SSM-tree core (K = 1, G = T, qknorm, no rope) */
    Params core;           /* core.embed unused; core.norm_f is the final norm */
    real *Win, *bin, *Wo, *bo, *log_std;   /* input projection, head, exploration */
    real *all; size_t size;                /* Win..log_std contiguous */
} Net;
static void net_alloc(Net *N, const Cfg *c) {
    N->c = *c; params_alloc(c, &N->core);
    const int d = c->d; N->size = (size_t)d * OBS_DIM + d + (size_t)ACT_DIM * d + ACT_DIM + ACT_DIM;
    N->all = calloc(N->size, sizeof(real)); real *q = N->all;
    N->Win = q; q += (size_t)d * OBS_DIM; N->bin = q; q += d; N->Wo = q; q += (size_t)ACT_DIM * d; N->bo = q; q += ACT_DIM; N->log_std = q;
}
static void net_zero(Net *N) { memset(N->core.all, 0, N->core.size * sizeof(real)); memset(N->all, 0, N->size * sizeof(real)); }

/* Load a policy.c SSM .pol (same architecture; conv = 1) into a Net. */
static int net_load_pol(Net *N, const char *path, float *mean, float *sd, double log_std0) {
    FILE *f = fopen(path, "rb"); PolCfg pc; if (!f || fread(&pc, sizeof pc, 1, f) != 1 || pc.type != POL_SSM) { if (f) fclose(f); return 1; }
    int P = pol_nparams(&pc); float *th = malloc(P * sizeof(float));
    if (fread(th, 4, P, f) != (size_t)P || fread(mean, 4, OBS_DIM, f) != OBS_DIM || fread(sd, 4, OBS_DIM, f) != OBS_DIM) { fclose(f); return 1; }
    fclose(f);
    Cfg c = {.d = pc.hidden, .n_layer = pc.layers, .T = pc.trees, .D = pc.depth, .n = pc.n, .p = pc.p, .K = 1, .G = pc.trees, .qknorm = 1, .rope = 0};
    net_alloc(N, &c); const int d = c.d, TN = c.T * NODES(&c), Pd = PDIM(&c); const float *q = th;
    for (int i = 0; i < d * OBS_DIM; i++) N->Win[i] = *q++; for (int i = 0; i < d; i++) N->bin[i] = *q++;
    for (int l = 0; l < c.n_layer; l++) {
        Layer *L = N->core.ly + l;
        for (int i = 0; i < d; i++) L->norm[i] = *q++;
        for (int i = 0; i < d; i++) L->conv[i] = 1;
        for (int i = 0; i < TN * d; i++) L->w_in[i] = *q++; for (int i = 0; i < TN * d; i++) L->w_out[i] = *q++;
        for (int i = 0; i < TN; i++) L->bias[i] = *q++; for (int i = 0; i < TN; i++) L->A_log[i] = *q++;
        for (int i = 0; i < Pd * d; i++) L->proj[i] = *q++; for (int i = 0; i < c.T; i++) L->dt_bias[i] = *q++;
        for (int i = 0; i < c.T * c.p * d; i++) L->U[i] = *q++;
    }
    for (int i = 0; i < d; i++) N->core.norm_f[i] = *q++;
    for (int i = 0; i < ACT_DIM * d; i++) N->Wo[i] = *q++; for (int i = 0; i < ACT_DIM; i++) N->bo[i] = *q++;
    for (int k = 0; k < ACT_DIM; k++) N->log_std[k] = (real)log_std0;
    free(th); return 0;
}
/* Write the Net as a policy.c .pol: the width-1 conv scales the inputs of w_in and proj, so it is folded there. */
static int net_save_pol(const Net *N, const char *path, const float *mean, const float *sd) {
    const Cfg *c = &N->c; const int d = c->d, TN = c->T * NODES(c), Pd = PDIM(c);
    PolCfg pc = {.type = POL_SSM, .in = OBS_DIM, .out = ACT_DIM, .hidden = d, .layers = c->n_layer, .trees = c->T, .depth = c->D, .n = c->n, .p = c->p};
    FILE *f = fopen(path, "wb"); if (!f) return 1;
    fwrite(&pc, sizeof pc, 1, f);
#define PUT(x) do { float v_ = (float)(x); fwrite(&v_, 4, 1, f); } while (0)
    for (int i = 0; i < d * OBS_DIM; i++) PUT(N->Win[i]); for (int i = 0; i < d; i++) PUT(N->bin[i]);
    for (int l = 0; l < c->n_layer; l++) {
        const Layer *L = N->core.ly + l;
        for (int i = 0; i < d; i++) PUT(L->norm[i]);
        for (int r = 0; r < TN; r++) for (int i = 0; i < d; i++) PUT(L->w_in[(size_t)r * d + i] * L->conv[i]);
        for (int i = 0; i < TN * d; i++) PUT(L->w_out[i]);
        for (int i = 0; i < TN; i++) PUT(L->bias[i]); for (int i = 0; i < TN; i++) PUT(L->A_log[i]);
        for (int r = 0; r < Pd; r++) for (int i = 0; i < d; i++) PUT(L->proj[(size_t)r * d + i] * L->conv[i]);
        for (int i = 0; i < c->T; i++) PUT(L->dt_bias[i]); for (int i = 0; i < c->T * c->p * d; i++) PUT(L->U[i]);
    }
    for (int i = 0; i < d; i++) PUT(N->core.norm_f[i]);
    for (int i = 0; i < ACT_DIM * d; i++) PUT(N->Wo[i]); for (int i = 0; i < ACT_DIM; i++) PUT(N->bo[i]);
#undef PUT
    fwrite(mean, 4, OBS_DIM, f); fwrite(sd, 4, OBS_DIM, f); fclose(f); return 0;
}

/* Per-sequence workspace: ssm_tree.c's Work (tapes, layer states, its own gradient accumulator for the core)
 * plus the input/head activations and the gradient of the input/head parameters. */
typedef struct { Work *w; real *u, *rn, *mu; Net G; } Seq;
static Seq *seq_new(const Net *N, int Lmax) {
    Seq *s = calloc(1, sizeof(Seq)); s->w = work_new(&N->c, Lmax);
    s->u = calloc((size_t)Lmax * N->c.d, sizeof(real)); s->rn = calloc(Lmax, sizeof(real)); s->mu = calloc((size_t)Lmax * ACT_DIM, sizeof(real));
    s->G.c = N->c; s->G.core = s->w->G; s->G.size = N->size; s->G.all = calloc(N->size, sizeof(real));
    real *q = s->G.all; const int d = N->c.d;
    s->G.Win = q; q += (size_t)d * OBS_DIM; s->G.bin = q; q += d; s->G.Wo = q; q += (size_t)ACT_DIM * d; s->G.bo = q; q += ACT_DIM; s->G.log_std = q;
    return s;
}
static size_t state_floats(const Cfg *c) { return (size_t)c->T * NODES(c) * c->n * c->p; }

/* One step of the policy (no tape): obs -> mean action, updating the recurrent states st[l]. */
static void net_step(const Net *N, const float *obs, LState *st, real *x, real *mu) {
    const int d = N->c.d; real u[512], r;
    for (int i = 0; i < d; i++) { real s = N->bin[i]; const real *w = N->Win + (size_t)i * OBS_DIM; for (int j = 0; j < OBS_DIM; j++) s += w[j] * obs[j]; x[i] = s; }
    for (int l = 0; l < N->c.n_layer; l++) layer_fwd(&N->c, N->core.ly + l, 1, x, st + l, NULL);
    rmsnorm_fwd(x, N->core.norm_f, u, d, &r);
    for (int k = 0; k < ACT_DIM; k++) mu[k] = tanh(N->bo[k] + dot(N->Wo + (size_t)k * d, u, d));
}

/* PPO loss over one sequence of L steps (obs normalised, actions, old log-probs, advantages), starting from the
 * recurrent states H0 (NULL = zero). Accumulates gradients into s->G (core in s->w->G) and returns the summed
 * surrogate loss; clipped is the clip range; ent the entropy bonus per dof. */
static double seq_loss(const Net *N, Seq *s, const float *obs, const float *act, const float *logp_old, const float *adv,
                       int L, const real *H0, double clip, double ent, int *nclip) {
    const Cfg *c = &N->c; const int d = c->d; Work *w = s->w; real *x = w->x; const size_t S = state_floats(c);
    for (int t = 0; t < L; t++) {
        real *xt = x + (size_t)t * d; const float *o = obs + (size_t)t * OBS_DIM;
        for (int i = 0; i < d; i++) { real a = N->bin[i]; const real *wr = N->Win + (size_t)i * OBS_DIM; for (int j = 0; j < OBS_DIM; j++) a += wr[j] * o[j]; xt[i] = a; }
    }
    for (int l = 0; l < c->n_layer; l++) {
        state_reset(c, w->st + l); if (H0) memcpy(w->st[l].H, H0 + (size_t)l * S, S * sizeof(real));
        layer_fwd(c, N->core.ly + l, L, x, w->st + l, w->tp + l);
    }
    real *g = w->g; memset(g, 0, sizeof(real) * (size_t)L * d); double loss = 0;
    for (int t = 0; t < L; t++) {
        real *u = s->u + (size_t)t * d, *mu = s->mu + (size_t)t * ACT_DIM; rmsnorm_fwd(x + (size_t)t * d, N->core.norm_f, u, d, s->rn + t);
        double lp = 0; const float *a = act + (size_t)t * ACT_DIM;
        for (int k = 0; k < ACT_DIM; k++) {
            mu[k] = tanh(N->bo[k] + dot(N->Wo + (size_t)k * d, u, d));
            const double sg = exp(N->log_std[k]), z = (a[k] - mu[k]) / sg; lp += -0.5 * z * z - N->log_std[k] - 0.9189385332046727;
        }
        const double ratio = exp(lp - logp_old[t]), A = adv[t];
        const double s1 = ratio * A, s2 = (ratio > 1 + clip ? 1 + clip : ratio < 1 - clip ? 1 - clip : ratio) * A;
        loss += -(s1 < s2 ? s1 : s2);
        const int active = s1 <= s2 || (ratio > 1 - clip && ratio < 1 + clip);   /* gradient flows through the unclipped term */
        if (!active && nclip) (*nclip)++;
        const double dlp = active ? -A * ratio : 0;   /* d loss / d log-prob */
        real du[512]; memset(du, 0, d * sizeof(real));
        for (int k = 0; k < ACT_DIM; k++) {
            const double sg = exp(N->log_std[k]), z = (a[k] - mu[k]) / sg;
            s->G.log_std[k] += (real)(dlp * (z * z - 1) - ent);
            const real dz = (real)(dlp * (z / sg) * (1 - mu[k] * mu[k]));
            s->G.bo[k] += dz; axpy(s->G.Wo + (size_t)k * d, dz, u, d); axpy(du, dz, N->Wo + (size_t)k * d, d);
        }
        rmsnorm_bwd(x + (size_t)t * d, N->core.norm_f, s->rn[t], du, g + (size_t)t * d, w->G.norm_f, d);
    }
    for (int l = c->n_layer - 1; l >= 0; l--) layer_bwd(c, N->core.ly + l, w->G.ly + l, L, g, w->tp + l, w);
    for (int t = 0; t < L; t++) {
        const real *gt = g + (size_t)t * d; const float *o = obs + (size_t)t * OBS_DIM;
        for (int i = 0; i < d; i++) { s->G.bin[i] += gt[i]; real *gw = s->G.Win + (size_t)i * OBS_DIM; for (int j = 0; j < OBS_DIM; j++) gw[j] += gt[i] * o[j]; }
    }
    return loss;
}

/* ------------------------------------------------------------------ ternary weights (quantisation-aware, STE) */
/* As the parent model: matrices run forward as {-1, 0, +1} x an absmean scale per group of 32 weights along each
 * row (BitNet b1.58), while the latent float weights take the gradient unchanged (straight-through) and Muon
 * updates them. Norms, biases, decay rates, steps and the exploration spread stay float. */
#define TGROUP 32
static void ternarize(real *w, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) for (size_t g0 = 0; g0 < cols; g0 += TGROUP) {
        real *x = w + r * cols + g0; const size_t n = cols - g0 < TGROUP ? cols - g0 : TGROUP; double a = 0;
        for (size_t i = 0; i < n; i++) a += fabs(x[i]);
        const real sc = (real)(a / n + 1e-8);
        for (size_t i = 0; i < n; i++) { const real q = x[i] / sc; x[i] = (q > 0.5 ? 1 : q < -0.5 ? -1 : 0) * sc; }
    }
}
/* dst = src with every matrix ternarised (dst allocated with the same Cfg). */
static void net_quantize(const Net *src, Net *dst) {
    const Cfg *c = &src->c; const int d = c->d, TN = c->T * NODES(c);
    memcpy(dst->core.all, src->core.all, src->core.size * sizeof(real)); memcpy(dst->all, src->all, src->size * sizeof(real));
    for (int l = 0; l < c->n_layer; l++) {
        Layer *L = dst->core.ly + l;
        ternarize(L->w_in, TN, d); ternarize(L->w_out, TN, d); ternarize(L->proj, PDIM(c), d); ternarize(L->U, (size_t)c->T * c->p, d);
    }
    ternarize(dst->Win, d, OBS_DIM); ternarize(dst->Wo, ACT_DIM, d);
}

/* ------------------------------------------------------------------ critic: MLP OBS -> H -> H -> 1 */
#define CH 128
typedef struct { real *W1, *b1, *W2, *b2, *W3, *b3, *all; size_t size; } Critic;
static void critic_alloc(Critic *C) {
    C->size = (size_t)CH * OBS_DIM + CH + (size_t)CH * CH + CH + CH + 1; C->all = calloc(C->size, sizeof(real)); real *q = C->all;
    C->W1 = q; q += (size_t)CH * OBS_DIM; C->b1 = q; q += CH; C->W2 = q; q += (size_t)CH * CH; C->b2 = q; q += CH; C->W3 = q; q += CH; C->b3 = q;
}
static void critic_init(Critic *C, uint64_t s) {
    for (size_t i = 0; i < (size_t)CH * OBS_DIM; i++) C->W1[i] = (real)((2 * urand(&s) - 1) / sqrt(OBS_DIM));
    for (size_t i = 0; i < (size_t)CH * CH; i++) C->W2[i] = (real)((2 * urand(&s) - 1) / sqrt(CH));
    for (int i = 0; i < CH; i++) C->W3[i] = (real)((2 * urand(&s) - 1) * 0.1 / sqrt(CH));
}
static real critic_fwd(const Critic *C, const float *o, real *h1, real *h2) {
    for (int i = 0; i < CH; i++) { real a = C->b1[i]; const real *w = C->W1 + (size_t)i * OBS_DIM; for (int j = 0; j < OBS_DIM; j++) a += w[j] * o[j]; h1[i] = tanh(a); }
    for (int i = 0; i < CH; i++) h2[i] = tanh(C->b2[i] + dot(C->W2 + (size_t)i * CH, h1, CH));
    return C->b3[0] + dot(C->W3, h2, CH);
}
static void critic_bwd(const Critic *C, Critic *G, const float *o, const real *h1, const real *h2, real dv) {
    real d2[CH], d1[CH]; memset(d1, 0, sizeof d1);
    G->b3[0] += dv; axpy(G->W3, dv, h2, CH);
    for (int i = 0; i < CH; i++) d2[i] = dv * C->W3[i] * (1 - h2[i] * h2[i]);
    for (int i = 0; i < CH; i++) { G->b2[i] += d2[i]; axpy(G->W2 + (size_t)i * CH, d2[i], h1, CH); axpy(d1, d2[i], C->W2 + (size_t)i * CH, CH); }
    for (int i = 0; i < CH; i++) { const real g = d1[i] * (1 - h1[i] * h1[i]); G->b1[i] += g; real *w = G->W1 + (size_t)i * OBS_DIM; for (int j = 0; j < OBS_DIM; j++) w[j] += g * o[j]; }
}

/* ------------------------------------------------------------------ optimiser: Muon for matrices, AdamW for the rest */
typedef struct { real *m, *v; } Mom;
static Mom mom_new(size_t n) { Mom m = {calloc(n, sizeof(real)), calloc(n, sizeof(real))}; return m; }
static void net_update(Net *N, const Net *G, Mom *mc, Mom *mh, const Hyper *h, real scale) {
    const Cfg *c = &N->c; const int d = c->d, TN = c->T * NODES(c);
    Hyper hm = *h;
    for (int l = 0; l < c->n_layer; l++) {
        Layer *p = N->core.ly + l; const Layer *g = G->core.ly + l;
        #define AT(base, ptr) ((base) + ((ptr) - N->core.all))
        muon(p->w_in, g->w_in, AT(mc->m, p->w_in), TN, d, &hm, scale); muon(p->w_out, g->w_out, AT(mc->m, p->w_out), TN, d, &hm, scale);
        muon(p->proj, g->proj, AT(mc->m, p->proj), PDIM(c), d, &hm, scale); muon(p->U, g->U, AT(mc->m, p->U), c->T * c->p, d, &hm, scale);
        adamw(p->norm, g->norm, AT(mc->m, p->norm), AT(mc->v, p->norm), d, h, scale);
        adamw(p->conv, g->conv, AT(mc->m, p->conv), AT(mc->v, p->conv), (size_t)c->K * d, h, scale);
        adamw(p->bias, g->bias, AT(mc->m, p->bias), AT(mc->v, p->bias), TN, h, scale);
        adamw(p->A_log, g->A_log, AT(mc->m, p->A_log), AT(mc->v, p->A_log), TN, h, scale);
        adamw(p->dt_bias, g->dt_bias, AT(mc->m, p->dt_bias), AT(mc->v, p->dt_bias), c->T, h, scale);
    }
    adamw(N->core.norm_f, G->core.norm_f, AT(mc->m, N->core.norm_f), AT(mc->v, N->core.norm_f), d, h, scale);
    #undef AT
    #define AH(base, ptr) ((base) + ((ptr) - N->all))
    muon(N->Win, G->Win, AH(mh->m, N->Win), d, OBS_DIM, &hm, scale);
    muon(N->Wo, G->Wo, AH(mh->m, N->Wo), ACT_DIM, d, &hm, scale);
    adamw(N->bin, G->bin, AH(mh->m, N->bin), AH(mh->v, N->bin), d, h, scale);
    adamw(N->bo, G->bo, AH(mh->m, N->bo), AH(mh->v, N->bo), ACT_DIM, h, scale);
    Hyper hs = *h; hs.wd = 0;
    adamw(N->log_std, G->log_std, AH(mh->m, N->log_std), AH(mh->v, N->log_std), ACT_DIM, &hs, scale);
    #undef AH
    for (int k = 0; k < ACT_DIM; k++) { if (N->log_std[k] < -3) N->log_std[k] = -3; if (N->log_std[k] > 0) N->log_std[k] = 0; }
}

/* ------------------------------------------------------------------ gradient check */
static int ppo_check(void) {
    Cfg c = {.d = 16, .n_layer = 2, .T = 2, .D = 2, .n = 4, .p = 4, .K = 1, .G = 2, .qknorm = 1, .rope = 0};
    Net N; net_alloc(&N, &c); params_init(&c, &N.core);
    uint64_t s = 99; for (size_t i = 0; i < N.size; i++) N.all[i] = (real)((2 * urand(&s) - 1) * 0.2);
    for (int k = 0; k < ACT_DIM; k++) N.log_std[k] = (real)-0.7;
    for (int l = 0; l < c.n_layer; l++) for (int t = 0; t < c.T; t++) N.core.ly[l].dt_bias[t] = (real)0.3;
    const int L = 8; float *obs = malloc(sizeof(float) * L * OBS_DIM), act[8 * ACT_DIM], lpo[8], adv[8];
    for (int i = 0; i < L * OBS_DIM; i++) obs[i] = (float)(2 * urand(&s) - 1);
    for (int i = 0; i < L * ACT_DIM; i++) act[i] = (float)(2 * urand(&s) - 1) * 0.5f;
    for (int t = 0; t < L; t++) { lpo[t] = (float)(-20 + 4 * urand(&s)); adv[t] = (float)(2 * urand(&s) - 1); }
    real *H0 = calloc(state_floats(&c) * c.n_layer, sizeof(real)); for (size_t i = 0; i < state_floats(&c) * c.n_layer; i++) H0[i] = (real)(0.1 * (2 * urand(&s) - 1));
    Seq *q = seq_new(&N, L);
    memset(q->w->G.all, 0, q->w->G.size * sizeof(real)); memset(q->G.all, 0, q->G.size * sizeof(real));
    seq_loss(&N, q, obs, act, lpo, adv, L, H0, 1e9, 0.01, NULL);   /* no clipping, so the loss is smooth */
    double worst = 0; int tested = 0;
    for (int trial = 0; trial < 300; trial++) {
        const int which = trial % 2; real *p, *gp; size_t n = which ? N.core.size : N.size, i = (size_t)(urand(&s) * n);
        if (which) { p = N.core.all + i; gp = q->w->G.all + i; } else { p = N.all + i; gp = q->G.all + i; }
        const real old = *p, e = (real)1e-5; Seq *r = seq_new(&N, L);
        *p = old + e; double lp = seq_loss(&N, r, obs, act, lpo, adv, L, H0, 1e9, 0.01, NULL);
        *p = old - e; double lm = seq_loss(&N, r, obs, act, lpo, adv, L, H0, 1e9, 0.01, NULL);
        *p = old;
        double num = (lp - lm) / (2 * e), ana = *gp;
        if (!which && p >= N.log_std) num -= 0;   /* the entropy bonus enters the gradient only (constant in the loss check) */
        if (!which && p >= N.log_std) ana += 0.01;
        if (fabs(num) + fabs(ana) < 1e-7) continue;
        double err = fabs(num - ana) / (fabs(num) + fabs(ana)); tested++; if (err > worst) worst = err;
    }
    printf("ppo gradient check: %d parameters, worst relative error %.3g\n", tested, worst);
    return worst < 1e-4 ? 0 : 1;
}

/* ------------------------------------------------------------------ training */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double gauss(uint64_t *s) { double a = urand(s) + 1e-12, b = urand(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

int main(int argc, char **argv) {
    const char *out = "ppo.pol", *in = NULL, *task = "balance", *mref = "motions_ref.bin";
    double seconds = 600, lr = 3e-4, vlr = 1e-3, gamma = 0.95, lam = 0.95, clipr = 0.2, ent = 0.0, log_std0 = -1.6;
    int threads = 4, nenv = 64, hor = 64, epochs = 4, mb = 4, seed = 0, ternary = 0;
    TaskCfg tc = {.push_max = 0, .push_every = 2, .steps = 300, .init_noise = 0.05};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--check")) return ppo_check();
#define OPT(nm, dst, conv) if (!strcmp(a, nm) && v) { dst = conv(v); i++; continue; }
        OPT("--out", out, (const char *)) OPT("--load", in, (const char *)) OPT("--task", task, (const char *)) OPT("--mref", mref, (const char *))
        OPT("--seconds", seconds, atof) OPT("--lr", lr, atof) OPT("--vlr", vlr, atof) OPT("--gamma", gamma, atof) OPT("--lambda", lam, atof)
        OPT("--clip", clipr, atof) OPT("--ent", ent, atof) OPT("--log-std", log_std0, atof) OPT("--threads", threads, atoi) OPT("--envs", nenv, atoi)
        OPT("--horizon", hor, atoi) OPT("--epochs", epochs, atoi) OPT("--minibatches", mb, atoi) OPT("--seed", seed, atoi) OPT("--ternary", ternary, atoi)
        OPT("--push", tc.push_max, atof) OPT("--steps", tc.steps, atoi) OPT("--assist", tc.assist, atof) OPT("--sway", tc.sway, atof) OPT("--arm-tone", tc.arm_tone, atof)
#undef OPT
        fprintf(stderr, "unknown argument %s\n", a); return 2;
    }
    Motions M;
    if (strcmp(task, "balance")) { if (mref_load(mref, &M) || !(tc.clip = mref_find(&M, task))) { fprintf(stderr, "no clip %s in %s\n", task, mref); return 1; } tc.task = TASK_IMITATE; tc.init_noise = 0; }
    float mean[OBS_DIM], sd[OBS_DIM]; Net N;
    if (in) { if (net_load_pol(&N, in, mean, sd, log_std0)) { fprintf(stderr, "cannot load %s (an SSM .pol)\n", in); return 1; } }
    else {
        Cfg c = {.d = 32, .n_layer = 2, .T = 2, .D = 3, .n = 8, .p = 8, .K = 1, .G = 2, .qknorm = 1, .rope = 0};
        net_alloc(&N, &c); params_init(&c, &N.core); uint64_t s = (uint64_t)seed * 31 + 7;
        for (size_t i = 0; i < (size_t)c.d * OBS_DIM; i++) N.Win[i] = (real)((2 * urand(&s) - 1) / sqrt(OBS_DIM));
        for (int k = 0; k < ACT_DIM; k++) N.log_std[k] = (real)log_std0;   /* Wo, bo zero: start from holding the reference */
        for (int i = 0; i < OBS_DIM; i++) { mean[i] = 0; sd[i] = 1; }
    }
    const Cfg *c = &N.c; const int d = c->d; const size_t S = state_floats(c) * c->n_layer;
    Critic C, CG; critic_alloc(&C); critic_alloc(&CG); critic_init(&C, seed + 5); Mom cm = mom_new(C.size);
    Mom mc = mom_new(N.core.size), mh = mom_new(N.size);
    /* rollout storage */
    Env *E = calloc(nenv, sizeof(Env)); LState *st = calloc((size_t)nenv * c->n_layer, sizeof(LState));
    for (int e = 0; e < nenv * c->n_layer; e++) { st[e].H = calloc(state_floats(c), sizeof(real)); st[e].hist = calloc(d, sizeof(real)); }
    float *O = malloc(sizeof(float) * (size_t)nenv * (hor + 1) * OBS_DIM), *A = malloc(sizeof(float) * (size_t)nenv * hor * ACT_DIM);
    float *LP = malloc(sizeof(float) * nenv * hor), *VAL = malloc(sizeof(float) * nenv * (hor + 1)), *R = malloc(sizeof(float) * nenv * hor);
    float *ADV = malloc(sizeof(float) * nenv * hor), *RET = malloc(sizeof(float) * nenv * hor);
    int *DONE = calloc((size_t)nenv * hor, sizeof(int)), *FELL = calloc((size_t)nenv * hor, sizeof(int));
    float *VTERM = calloc((size_t)nenv * hor, sizeof(float));   /* value of the last observation of an episode cut by the time limit */
    real *H0 = malloc(sizeof(real) * (size_t)nenv * S);
    double *osum = calloc(OBS_DIM, sizeof(double)), *osq = calloc(OBS_DIM, sizeof(double)), on = 0;
    if (in) { on = 20000; for (int i = 0; i < OBS_DIM; i++) { osum[i] = (double)mean[i] * on; osq[i] = ((double)sd[i] * sd[i] + (double)mean[i] * mean[i]) * on; } }   /* the loaded statistics as a prior */
    uint64_t erng = (uint64_t)seed * 7919 + 1; long episodes = 0; double ep_ret_sum = 0, ep_len_sum = 0; int ep_count = 0;
    double *ep_ret = calloc(nenv, sizeof(double));
    for (int e = 0; e < nenv; e++) env_reset(&E[e], &tc, erng + e * 1000003ULL), episodes++;
    Seq **sq = malloc(threads * sizeof(Seq *)); for (int t = 0; t < threads; t++) sq[t] = seq_new(&N, hor);
    Net Gsum; net_alloc(&Gsum, c);
    /* Fw: the network that runs (ternarised copy of the latent weights when --ternary 1, else the latent net) */
    Net Q; net_alloc(&Q, c); Net *Fw = ternary ? &Q : &N; if (ternary) net_quantize(&N, &Q);
    printf("{\"event\":\"config\",\"task\":\"%s\",\"weights\":\"%s\",\"optimizer\":\"muon+adamw\",\"params\":%zu,\"envs\":%d,\"horizon\":%d,\"lr\":%g,\"log_std\":%g}\n",
           task, ternary ? "ternary (STE)" : "float", N.size + N.core.size - (size_t)V * d, nenv, hor, lr, log_std0);
    Hyper hp = {.lr = (real)lr, .beta1 = .9, .beta2 = .999, .eps = 1e-8, .wd = 0.0, .clip = 1}, hv = hp; hv.lr = (real)vlr;
    double t0 = now_s(); long it = 0, steps_total = 0; double best = -1;
    while (now_s() - t0 < seconds) {
        it++;
        /* ---- rollout ---- */
        for (int e = 0; e < nenv; e++) for (int l = 0; l < c->n_layer; l++) memcpy(H0 + (size_t)e * S + (size_t)l * state_floats(c), st[e * c->n_layer + l].H, state_floats(c) * sizeof(real));
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (int e = 0; e < nenv; e++) {
            real x[512], mu[ACT_DIM], h1[CH], h2[CH]; float o[OBS_DIM], a[ACT_DIM]; uint64_t rs = erng ^ ((uint64_t)it << 20) ^ (uint64_t)e * 0x9E3779B97F4A7C15ULL;
            for (int t = 0; t <= hor; t++) {
                float *on_ = O + ((size_t)e * (hor + 1) + t) * OBS_DIM; env_obs(&E[e], o);
                for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; on_[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
                VAL[e * (hor + 1) + t] = (float)critic_fwd(&C, on_, h1, h2);
                if (t == hor) break;
                net_step(Fw, on_, st + e * c->n_layer, x, mu);
                double lp = 0; float *at = A + ((size_t)e * hor + t) * ACT_DIM;
                for (int k = 0; k < ACT_DIM; k++) { const double sg = exp(N.log_std[k]), z = gauss(&rs); at[k] = (float)(mu[k] + sg * z); lp += -0.5 * z * z - N.log_std[k] - 0.9189385332046727; a[k] = at[k] > 1 ? 1 : at[k] < -1 ? -1 : at[k]; }
                LP[e * hor + t] = (float)lp;
                const double r = env_step(&E[e], a); R[e * hor + t] = (float)r;
#pragma omp atomic
                ep_ret[e] += r;
                const int fell = E[e].done && E[e].step < E[e].cfg.steps;
                DONE[e * hor + t] = E[e].done; FELL[e * hor + t] = fell;
                if (E[e].done && !fell) {
                    float ot[OBS_DIM]; env_obs(&E[e], o);
                    for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; ot[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
                    VTERM[e * hor + t] = (float)critic_fwd(&C, ot, h1, h2);
                }
                if (E[e].done) {
#pragma omp critical
                    { ep_ret_sum += ep_ret[e]; ep_len_sum += E[e].step; ep_count++; episodes++; }
                    ep_ret[e] = 0;
                    env_reset(&E[e], &tc, erng + (uint64_t)episodes * 7777 + e);
                    for (int l = 0; l < c->n_layer; l++) state_reset(c, st + e * c->n_layer + l);
                }
            }
        }
        steps_total += (long)nenv * hor;
        /* observation statistics (applied from the next iteration on) */
        for (int e = 0; e < nenv; e++) for (int t = 0; t < hor; t += 4) {
            const float *on_ = O + ((size_t)e * (hor + 1) + t) * OBS_DIM;
            for (int i = 0; i < OBS_DIM; i++) { double raw = on_[i] * sd[i] + mean[i]; osum[i] += raw; osq[i] += raw * raw; } on++;
        }
        /* ---- GAE ---- */
        double asum = 0, asq = 0;
        for (int e = 0; e < nenv; e++) {
            double g = 0;
            for (int t = hor - 1; t >= 0; t--) {
                const int k = e * hor + t;
                const double vnext = FELL[k] ? 0 : DONE[k] ? VTERM[k] : VAL[e * (hor + 1) + t + 1];   /* a fall is terminal; the time limit bootstraps */
                const double delta = R[k] + gamma * vnext - VAL[e * (hor + 1) + t];
                g = delta + (DONE[k] ? 0 : gamma * lam * g);
                ADV[k] = (float)g; RET[k] = (float)(g + VAL[e * (hor + 1) + t]); asum += g; asq += g * g;
            }
        }
        const double am = asum / (nenv * hor), asd = sqrt(fmax(asq / (nenv * hor) - am * am, 1e-8));
        for (int k = 0; k < nenv * hor; k++) ADV[k] = (float)((ADV[k] - am) / asd);
        /* ---- update ---- */
        int nclip = 0; double surr = 0, vloss = 0;
        int *order = malloc(nenv * sizeof(int)); for (int e = 0; e < nenv; e++) order[e] = e;
        for (int ep = 0; ep < epochs; ep++) {
            for (int e = nenv - 1; e > 0; e--) { int j = (int)(urand(&erng) * (e + 1)); int tmp = order[e]; order[e] = order[j]; order[j] = tmp; }
            for (int b = 0; b < mb; b++) {
                const int e0 = b * nenv / mb, e1 = (b + 1) * nenv / mb; long nsamp = 0;
                for (int t = 0; t < threads; t++) { memset(sq[t]->w->G.all, 0, sq[t]->w->G.size * sizeof(real)); memset(sq[t]->G.all, 0, sq[t]->G.size * sizeof(real)); }
#pragma omp parallel for num_threads(threads) schedule(dynamic) reduction(+:surr, nclip, nsamp)
                for (int ii = e0; ii < e1; ii++) {
#ifdef _OPENMP
                    const int th = omp_get_thread_num();
#else
                    const int th = 0;
#endif
                    const int e = order[ii]; int t0_ = 0;
                    while (t0_ < hor) {   /* one segment per episode piece in this chunk */
                        int t1 = t0_; while (t1 < hor - 1 && !DONE[e * hor + t1]) t1++;
                        const int L = t1 - t0_ + 1; int nc = 0;
                        surr += seq_loss(Fw, sq[th], O + ((size_t)e * (hor + 1) + t0_) * OBS_DIM, A + ((size_t)e * hor + t0_) * ACT_DIM,
                                         LP + e * hor + t0_, ADV + e * hor + t0_, L, t0_ == 0 ? H0 + (size_t)e * S : NULL, clipr, ent, &nc);
                        nclip += nc; nsamp += L; t0_ = t1 + 1;
                    }
                }
                /* reduce and step the policy */
                memset(Gsum.core.all, 0, Gsum.core.size * sizeof(real)); memset(Gsum.all, 0, Gsum.size * sizeof(real));
                for (int t = 0; t < threads; t++) {
                    for (size_t i = 0; i < Gsum.core.size; i++) Gsum.core.all[i] += sq[t]->w->G.all[i];
                    for (size_t i = 0; i < Gsum.size; i++) Gsum.all[i] += sq[t]->G.all[i];
                }
                double gn = 0; for (size_t i = 0; i < Gsum.core.size; i++) gn += (double)Gsum.core.all[i] * Gsum.core.all[i]; for (size_t i = 0; i < Gsum.size; i++) gn += (double)Gsum.all[i] * Gsum.all[i];
                real scale = (real)(1.0 / nsamp); gn = sqrt(gn) * scale; if (gn > 1) scale *= (real)(1 / gn);
                hp.step++; net_update(&N, &Gsum, &mc, &mh, &hp, scale);   /* the latent weights take the (straight-through) gradient */
                if (ternary) net_quantize(&N, &Q);
                /* critic on the same samples */
                memset(CG.all, 0, CG.size * sizeof(real)); real h1[CH], h2[CH];
                for (int ii = e0; ii < e1; ii++) { const int e = order[ii];
                    for (int t = 0; t < hor; t++) { const float *on_ = O + ((size_t)e * (hor + 1) + t) * OBS_DIM; const real v = critic_fwd(&C, on_, h1, h2), dv = v - RET[e * hor + t];
                        vloss += dv * dv; critic_bwd(&C, &CG, on_, h1, h2, dv); } }
                hv.step++; adamw(C.all, CG.all, cm.m, cm.v, C.size, &hv, (real)(1.0 / ((e1 - e0) * hor)));
            }
        }
        free(order);
        if (on > 10) for (int i = 0; i < OBS_DIM; i++) { const double mu_ = osum[i] / on, var = osq[i] / on - mu_ * mu_; mean[i] = (float)mu_; sd[i] = (float)sqrt(fmax(var, 1e-2)); }
        const double rew = ep_count ? ep_ret_sum / ep_count : 0, len = ep_count ? ep_len_sum / ep_count / CTRL_HZ : 0;
        double sgm = 0; for (int k = 0; k < ACT_DIM; k++) sgm += exp(N.log_std[k]) / ACT_DIM;
        if (it % 5 == 1 || now_s() - t0 >= seconds) {
            printf("{\"event\":\"iter\",\"iter\":%ld,\"seconds\":%.0f,\"steps\":%ld,\"episode_return\":%.1f,\"episode_s\":%.2f,\"episodes\":%d,\"clip_frac\":%.3f,\"value_loss\":%.3f,\"std\":%.3f}\n",
                   it, now_s() - t0, steps_total, rew, len, ep_count, (double)nclip / ((double)epochs * nenv * hor), vloss / ((double)epochs * nenv * hor), sgm);
            fflush(stdout);
            if (ep_count >= 8 && rew > best) { best = rew; net_save_pol(Fw, out, mean, sd); }
            ep_ret_sum = ep_len_sum = 0; ep_count = 0;
        }
    }
    if (best < 0) net_save_pol(Fw, out, mean, sd);
    printf("{\"event\":\"done\",\"iterations\":%ld,\"steps\":%ld,\"best_episode_return\":%.1f}\n", it, steps_total, best);
    return 0;
}
