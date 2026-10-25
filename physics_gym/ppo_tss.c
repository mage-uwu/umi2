/* ppo_tss.c: PPO for the body controller on the nanotss model itself (a fork of the language model's stack).
 *
 * The policy is nanotss_blt.c's byte-mode network (SSM-tree and FFF blocks, ternary weights with int8
 * activations and AVX-512 kernels, sampled routing credit, straight-through gradients), with its byte
 * embedding and softmax head swapped for an observation projection and a Gaussian action head:
 *   x_t = Win obs_t + bin  ->  nanotss layers  ->  mu = tanh(Wo rmsnorm(x) + bo),  a ~ N(mu, exp(log_std)^2)
 * Rollouts stream through gen_core (the inference path); training runs each chunk through layer_fwd /
 * layer_bwd from the recurrent state saved at its start (carry_h / carry_hist), batched by the unmodified
 * tss_train_step through its sequence hook, so the reduction, clipping, blockwise Muon and requantisation are
 * the language model's own. Win / Wo / bo / log_std are trained with Adam; a small MLP critic estimates values.
 *
 *   ./ppo_tss --task walk --assist 0.3 --seconds 600 --out walk     (writes walk.tss and walk.head)
 *   ./ppo_tss --task goto --mref data/clean/cmu_gait.bin --room rooms_nav.json --load data/clean/warm/aware --assist 0.05 --assist-auto
 *   ./ppo_tss --task getup_imitate --style data/cmu_jump.bin:fjump1,fjump2 --room data/jump_bank_standing.json --load data/clean/warm/jump_stick --eval 64
 * Tasks: balance | crawl | getup | goto | getup_imitate | a clip name of --mref (imitation), as gpu/ppo_gpu.py's; --room a room
 * bank (or one room.json); --eval N runs a policy instead of training it. Heads with fewer inputs (older policies) are widened.
 */
#include "env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define main nanotss_main
#include "../nanotss_blt.c"
#undef main

/* ------------------------------------------------------------------ input projection and action head (float) */
typedef struct { real *Win, *bin, *Wo, *bo, *log_std, *all; size_t size; } Ends;
static void ends_alloc(Ends *E, int d) {
    E->size = (size_t)d * OBS_DIM + d + (size_t)ACT_DIM * d + ACT_DIM + ACT_DIM; E->all = calloc(E->size, sizeof(real)); real *q = E->all;
    E->Win = q; q += (size_t)d * OBS_DIM; E->bin = q; q += d; E->Wo = q; q += (size_t)ACT_DIM * d; E->bo = q; q += ACT_DIM; E->log_std = q;
}

/* ------------------------------------------------------------------ critic: MLP OBS -> 128 -> 128 -> 1 */
#define CH 128
typedef struct { real *W1, *b1, *W2, *b2, *W3, *b3, *all; size_t size; } Critic;
static void critic_alloc(Critic *C) {
    C->size = (size_t)CH * OBS_DIM + CH + (size_t)CH * CH + CH + CH + 1; C->all = calloc(C->size, sizeof(real)); real *q = C->all;
    C->W1 = q; q += (size_t)CH * OBS_DIM; C->b1 = q; q += CH; C->W2 = q; q += (size_t)CH * CH; C->b2 = q; q += CH; C->W3 = q; q += CH; C->b3 = q;
}
static double ur(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (*s >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(uint64_t *s) { double a = ur(s) + 1e-12, b = ur(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
static void critic_init(Critic *C, uint64_t s) {
    for (size_t i = 0; i < (size_t)CH * OBS_DIM; i++) C->W1[i] = (real)((2 * ur(&s) - 1) / sqrt(OBS_DIM));
    for (size_t i = 0; i < (size_t)CH * CH; i++) C->W2[i] = (real)((2 * ur(&s) - 1) / sqrt(CH));
    for (int i = 0; i < CH; i++) C->W3[i] = (real)((2 * ur(&s) - 1) * 0.1 / sqrt(CH));
}
static real critic_fwd(const Critic *C, const float *o, real *h1, real *h2) {
    for (int i = 0; i < CH; i++) h1[i] = tanh(C->b1[i] + dotr(C->W1 + (size_t)i * OBS_DIM, o, OBS_DIM));
    for (int i = 0; i < CH; i++) h2[i] = tanh(C->b2[i] + dotr(C->W2 + (size_t)i * CH, h1, CH));
    return C->b3[0] + dotr(C->W3, h2, CH);
}
static void critic_bwd(const Critic *C, Critic *G, const float *o, const real *h1, const real *h2, real dv) {
    real d2[CH], d1[CH]; memset(d1, 0, sizeof d1);
    G->b3[0] += dv; axpy(G->W3, dv, h2, CH);
    for (int i = 0; i < CH; i++) d2[i] = dv * C->W3[i] * (1 - h2[i] * h2[i]);
    for (int i = 0; i < CH; i++) { G->b2[i] += d2[i]; axpy(G->W2 + (size_t)i * CH, d2[i], h1, CH); axpy(d1, d2[i], C->W2 + (size_t)i * CH, CH); }
    for (int i = 0; i < CH; i++) { const real gg = d1[i] * (1 - h1[i] * h1[i]); G->b1[i] += gg; axpy(G->W1 + (size_t)i * OBS_DIM, gg, o, OBS_DIM); }
}

/* ------------------------------------------------------------------ the batch the sequence hook trains on */
typedef struct { int e, t0, L; } Segment;
typedef struct {
    const Ends *ends; Ends *eg;            /* eg: one gradient accumulator per thread */
    const Segment *seg; int hor;
    const float *O, *A, *LP, *ADV;         /* O: (hor+1) observations per env, normalised */
    const real *H0, *HH0;                  /* recurrent state at each chunk start: SSM states, conv history */
    real *hscratch, *hhscratch;            /* per-thread copies (carry_post writes the end state back) */
    size_t S_all, HH_all;
    double clip, ent;
    long clipped;
    int first; double lpdiff; long lpn;    /* first epoch: rollout vs training log-prob, which must agree */
} Batch;

/* PPO loss for one segment, through the nanotss layers (hooked into tss_train_step). */
static real ppo_seq(void *user, int b, const Cfg *c, const Params *P, const Params *Q, Params *G, unsigned char *hit, int *ns, Work *w, const Codes *k) {
    Batch *B = user; const Segment sg = B->seg[b]; const int d = c->d, L = sg.L, nl = c->n_layer, N = NODES(c), T = c->T;
#ifdef _OPENMP
    const int th = omp_get_thread_num();
#else
    const int th = 0;
#endif
    const Ends *E = B->ends; Ends *eg = B->eg + th;
    w->k = k;
    if (sg.t0 == 0) {   /* the chunk's saved state; later segments start a fresh episode from zero */
        real *hs = B->hscratch + (size_t)th * B->S_all, *hh = B->hhscratch + (size_t)th * B->HH_all;
        memcpy(hs, B->H0 + (size_t)sg.e * B->S_all, B->S_all * sizeof(real)); memcpy(hh, B->HH0 + (size_t)sg.e * B->HH_all, B->HH_all * sizeof(real));
        w->carry_h = hs; w->carry_hist = hh;
    } else { w->carry_h = NULL; w->carry_hist = NULL; }
    real *x = w->x, *h = w->h; Tape *tp = w->tp;
    const float *obs = B->O + ((size_t)sg.e * (B->hor + 1) + sg.t0) * OBS_DIM;
    for (int t = 0; t < L; t++) {
        const float *o = obs + (size_t)t * OBS_DIM;
        for (int i = 0; i < d; i++) x[(size_t)t * d + i] = E->bin[i] + dotr(E->Win + (size_t)i * OBS_DIM, o, OBS_DIM);
    }
    for (int l = 0; l < nl; l++) {
        if (!(l & 1)) carry_pre(c, w, l, h);
        layer_fwd(c, P, Q, l, L, x, h, tp + l, w);
        if (!(l & 1)) carry_post(c, w, l, L, h);
    }
    real *g = w->g; memset(g, 0, sizeof(real) * (size_t)L * d); double loss = 0;
    const float *act = B->A + ((size_t)sg.e * B->hor + sg.t0) * ACT_DIM, *lpo = B->LP + (size_t)sg.e * B->hor + sg.t0, *adv = B->ADV + (size_t)sg.e * B->hor + sg.t0;
    real u[512], du[512], mu[ACT_DIM];
    for (int t = 0; t < L; t++) {
        real r; rmsnorm_fwd(x + (size_t)t * d, P->norm_f, u, d, &r);
        double lp = 0; const float *a = act + (size_t)t * ACT_DIM;
        for (int kk = 0; kk < ACT_DIM; kk++) {
            mu[kk] = tanh(E->bo[kk] + dotr(E->Wo + (size_t)kk * d, u, d));
            const double sgm = exp(E->log_std[kk]), z = (a[kk] - mu[kk]) / sgm; lp += -0.5 * z * z - E->log_std[kk] - 0.9189385332046727;
        }
        if (B->first) {
#pragma omp atomic
            B->lpdiff += fabs(lp - lpo[t]);
#pragma omp atomic
            B->lpn++;
        }
        const double ratio = exp(lp - lpo[t]), Aa = adv[t], cl = B->clip;
        const double s1 = ratio * Aa, s2 = (ratio > 1 + cl ? 1 + cl : ratio < 1 - cl ? 1 - cl : ratio) * Aa;
        loss += -(s1 < s2 ? s1 : s2);
        const int active = s1 <= s2 || (ratio > 1 - cl && ratio < 1 + cl);
        if (!active) {
#pragma omp atomic
            B->clipped++;
        }
        const double dlp = active ? -Aa * ratio : 0;
        memset(du, 0, d * sizeof(real));
        for (int kk = 0; kk < ACT_DIM; kk++) {
            const double sgm = exp(E->log_std[kk]), z = (a[kk] - mu[kk]) / sgm;
            eg->log_std[kk] += (real)(dlp * (z * z - 1) - B->ent);
            const real dz = (real)(dlp * (z / sgm) * (1 - mu[kk] * mu[kk]));
            eg->bo[kk] += dz; axpy(eg->Wo + (size_t)kk * d, dz, u, d); axpy(du, dz, E->Wo + (size_t)kk * d, d);
        }
        rmsnorm_bwd(x + (size_t)t * d, P->norm_f, r, du, g + (size_t)t * d, G->norm_f, d);
    }
    real *lam = w->lam;
    for (int l = nl - 1; l >= 0; l--) {
        if (!(l & 1)) memset(lam, 0, sizeof(real) * (size_t)T * N * c->n * c->p);
        layer_bwd(c, P, Q, G, l, L, g, lam, hit + node_off(c, l), tp + l, w);
    }
    for (int t = 0; t < L; t++) {
        const real *gt = g + (size_t)t * d; const float *o = obs + (size_t)t * OBS_DIM;
        for (int i = 0; i < d; i++) { eg->bin[i] += gt[i]; axpy(eg->Win + (size_t)i * OBS_DIM, gt[i], o, OBS_DIM); }
    }
    w->carry_h = w->carry_hist = NULL;
    *ns = L;
    return (real)loss;
}

/* One policy step on the inference path: obs -> mean action, advancing the streaming state g. */
static void policy_step(Model *m, const Ends *E, Gen *g, const float *obs, real *mu) {
    const int d = m->c.d; real x0[512], u[512], r;
    for (int i = 0; i < d; i++) x0[i] = E->bin[i] + dotr(E->Win + (size_t)i * OBS_DIM, obs, OBS_DIM);
    gen_core(m, g, x0, NULL);
    rmsnorm_fwd(g->x, m->P.norm_f, u, d, &r);
    for (int k = 0; k < ACT_DIM; k++) mu[k] = tanh(E->bo[k] + dotr(E->Wo + (size_t)k * d, u, d));
}
/* Saved state of a Gen in carry_h / carry_hist layout: SSM states of the even layers, conv history of all. */
static void gen_snapshot(const Cfg *c, const Gen *g, real *hs, real *hh) {
    const size_t S = (size_t)c->T * NODES(c) * c->n * c->p;
    for (int l = 0; l < c->n_layer; l += 2) memcpy(hs + (size_t)(l / 2) * S, g->h + node_off(c, l) * c->n * c->p, S * sizeof(real));
    memcpy(hh, g->hist, (size_t)c->n_layer * (c->K > 1 ? c->K - 1 : 1) * c->d * sizeof(real));
}

/* a saved head (save_head, or ppo_gpu.py's): an older one, with fewer inputs (the newer come last) or outputs, is
 * widened with zero weights, so the policy acts as it did; the critic too when it was saved; 0 when read */
static int load_head(const char *path, Ends *E, float *mean, float *sd, int d, Critic *C, int *had_critic) {
    FILE *f = fopen(path, "rb"); char mg[4]; int dims[3]; *had_critic = 0; if (!f) return 1;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "TSPH", 4) || fread(dims, sizeof dims, 1, f) != 1 || dims[0] != d || dims[1] > OBS_DIM || dims[2] > ACT_DIM) { fclose(f); return 1; }
    const int ho = dims[1], ha = dims[2]; real *tmp = malloc(sizeof(real) * (size_t)CH * ho); int ok = 1;
    memset(E->all, 0, E->size * sizeof(real));
    ok &= fread(tmp, sizeof(real), (size_t)d * ho, f) == (size_t)d * ho;
    for (int i = 0; i < d; i++) memcpy(E->Win + (size_t)i * OBS_DIM, tmp + (size_t)i * ho, sizeof(real) * ho);
    ok &= fread(E->bin, sizeof(real), d, f) == (size_t)d; ok &= fread(E->Wo, sizeof(real), (size_t)ha * d, f) == (size_t)ha * d;
    ok &= fread(E->bo, sizeof(real), ha, f) == (size_t)ha; ok &= fread(E->log_std, sizeof(real), ha, f) == (size_t)ha;
    for (int i = 0; i < OBS_DIM; i++) { mean[i] = 0; sd[i] = 1; }
    ok &= fread(mean, 4, ho, f) == (size_t)ho; ok &= fread(sd, 4, ho, f) == (size_t)ho;
    if (ok && fread(tmp, sizeof(real), (size_t)CH * ho, f) == (size_t)CH * ho) {   /* the critic, when saved with it */
        memset(C->W1, 0, sizeof(real) * (size_t)CH * OBS_DIM);
        for (int i = 0; i < CH; i++) memcpy(C->W1 + (size_t)i * OBS_DIM, tmp + (size_t)i * ho, sizeof(real) * ho);
        const size_t rest = C->size - (size_t)CH * OBS_DIM; *had_critic = fread(C->b1, sizeof(real), rest, f) == rest;
    }
    free(tmp); fclose(f); return !ok;
}
static int save_head(const char *path, const Ends *E, const float *mean, const float *sd, int d, const Critic *C) {
    FILE *f = fopen(path, "wb"); if (!f) return 1;
    fwrite("TSPH", 1, 4, f); int dims[3] = {d, OBS_DIM, ACT_DIM}; fwrite(dims, sizeof dims, 1, f);
    fwrite(E->all, sizeof(real), E->size, f); fwrite(mean, 4, OBS_DIM, f); fwrite(sd, 4, OBS_DIM, f); fwrite(C->all, sizeof(real), C->size, f); return fclose(f);
}

int main(int argc, char **argv) {
    const char *load_from = NULL, *out = "tsspolicy", *task = "balance", *mref = "data/motions_ref.bin";
    double seconds = 600, lr = 3e-4, hlr = 3e-4, vlr = 1e-3, gamma = 0.95, lam = 0.95, clipr = 0.2, ent = 0, log_std0 = -1.6;
    int threads = 4, nenv = 64, hor = 64, epochs = 4, mb = 4, seed = 0, d = 64, depth = 4, trees = 4;
    TaskCfg tc = {.push_every = 2, .steps = 300, .init_noise = 0.05}; int steps_set = 0, assist_auto = 0, eval_n = 0; const char *style_arg = NULL, *room = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
#define OPT(nm, dst, conv) if (!strcmp(a, nm) && v) { dst = conv(v); i++; continue; }
        OPT("--out", out, (const char *)) OPT("--load", load_from, (const char *)) OPT("--task", task, (const char *)) OPT("--mref", mref, (const char *))
        OPT("--seconds", seconds, atof) OPT("--lr", lr, atof) OPT("--head-lr", hlr, atof) OPT("--vlr", vlr, atof) OPT("--clip", clipr, atof)
        OPT("--log-std", log_std0, atof) OPT("--threads", threads, atoi) OPT("--envs", nenv, atoi) OPT("--horizon", hor, atoi)
        OPT("--epochs", epochs, atoi) OPT("--minibatches", mb, atoi) OPT("--seed", seed, atoi) OPT("--width", d, atoi) OPT("--depth", depth, atoi) OPT("--trees", trees, atoi)
        OPT("--push", tc.push_max, atof) OPT("--act-mul", tc.act_mul, atof) OPT("--style", style_arg, (const char *)) OPT("--style-start", tc.style_start, atof) if (!strcmp(a, "--steps") && v) { tc.steps = atoi(v); steps_set = 1; i++; continue; } OPT("--assist", tc.assist, atof) OPT("--sway", tc.sway, atof) OPT("--arm-tone", tc.arm_tone, atof)
        OPT("--room", room, (const char *)) OPT("--goal-tops", tc.goal_tops, atof) OPT("--goal-cone", tc.goal_cone, atof) OPT("--free-heading", tc.free_heading, atoi) OPT("--eval", eval_n, atoi)
        if (!strcmp(a, "--assist-auto")) { assist_auto = 1; continue; }
#undef OPT
        fprintf(stderr, "unknown argument %s\n", a); return 2;
    }
    /* the task, as gpu/tssenv.c pool_new and gpu_env.py set it up (the same flags give the same tasks) */
    static Motions M, SM;
    if (style_arg) {   /* --style file.bin:clip1,clip2,...: a motion style prior, or (getup_imitate) the clip set to track */
        char buf[1024]; snprintf(buf, sizeof buf, "%s", style_arg); char *colon = strchr(buf, ':'); if (!colon) { fprintf(stderr, "--style file:clips\n"); return 2; }
        *colon = 0; if (mref_load(buf, &SM)) { fprintf(stderr, "cannot load %s\n", buf); return 1; }
        for (char *tok = strtok(colon + 1, ","); tok && tc.nstyle < 8; tok = strtok(NULL, ",")) {
            const Clip *cl = mref_find(&SM, tok); if (!cl) { fprintf(stderr, "no clip %s\n", tok); return 1; }
            env_style_prepare(cl); tc.style[tc.nstyle++] = cl; }
    }
    if (!strcmp(task, "crawl") || !strcmp(task, "getup")) {   /* the floor tasks: no clip, wider joint ranges */
        tc.task = !strcmp(task, "crawl") ? TASK_CRAWL : TASK_GETUP; if (!tc.act_mul) tc.act_mul = 2.5; if (!steps_set) tc.steps = tc.task == TASK_CRAWL ? 300 : 240;
    } else if (!strcmp(task, "goto")) {   /* walking to goals: the walk clip's cycle is the phase input and the gait prior */
        tc.task = TASK_GOTO; if (!steps_set) tc.steps = 600;
        if (mref_load(mref, &M) || !(tc.clip = mref_find(&M, "walk"))) { fprintf(stderr, "no clip walk in %s\n", mref); return 1; }
        if (!tc.nstyle) { env_style_prepare(tc.clip); tc.style[tc.nstyle++] = tc.clip; }
    } else if (!strcmp(task, "getup_imitate")) {   /* tracking a clip set (get-ups; in a room, a jump course: env.c course mode) */
        if (!tc.nstyle) { fprintf(stderr, "getup_imitate: --style file:clips\n"); return 2; }
        tc.task = TASK_IMITATE; tc.init_noise = 0; if (!tc.act_mul) tc.act_mul = 2.5; if (!steps_set) tc.steps = 240; tc.clip = tc.style[0];
        if (tc.style_start > 0) env_fall_bank();   /* the matched starts' falls */
    } else if (strcmp(task, "balance")) {   /* imitation of one clip */
        if (mref_load(mref, &M) || !(tc.clip = mref_find(&M, task))) { fprintf(stderr, "no clip %s in %s\n", task, mref); return 1; }
        tc.task = TASK_IMITATE; tc.init_noise = 0; tc.nstyle = 0;
    }
    if (room && *room) { const int nr = room_load_bank(room); if (nr < 0) { fprintf(stderr, "cannot read %s\n", room); return 1; } }
    /* the language model's defaults: pure SSM-tree + FFF, per-tree keys/queries, QK-norm, Muon; no rotary
     * (positions would differ between rollout and training chunks) */
    ssm_groups = trees; ssm_qknorm = 1; ssm_rope = 0; n_loops = 1;
    Cfg gc = {.vocab = 256, .d = d, .n_layer = 4, .T = trees, .D = depth, .n = 16, .p = 16, .K = 3, .group = d, .walks = 2, .temp = 1, .fw_mask = 0, .fff_gate = 1};
    Model *m;
    if (load_from) {   /* continue from a saved policy (curriculum stages): its stack, ends and observation statistics */
        char p1[1024]; snprintf(p1, sizeof p1, "%s.tss", load_from);
        if (!(m = load(p1, threads, hor))) { fprintf(stderr, "cannot load %s\n", p1); return 1; }
        d = m->c.d;
    } else { m = model_new(&gc, threads, hor); g_rng = (uint64_t)seed * 7919 + 88172645463325252ULL; params_init(&gc, &m->P); model_requantize(m); }
    byte_muon = 1; byte_muon_model = m;
    const Cfg *c = &m->c;
    Ends EN; ends_alloc(&EN, d); Ends *EG = calloc(threads, sizeof(Ends)); for (int t = 0; t < threads; t++) ends_alloc(&EG[t], d);
    { uint64_t s = (uint64_t)seed * 31 + 7; for (size_t i = 0; i < (size_t)d * OBS_DIM; i++) EN.Win[i] = (real)((2 * ur(&s) - 1) / sqrt(OBS_DIM)); }
    for (int k = 0; k < ACT_DIM; k++) EN.log_std[k] = (real)log_std0;   /* Wo, bo zero: training starts from holding the reference */
    real *em = calloc(EN.size, sizeof(real)), *ev = calloc(EN.size, sizeof(real)), *gsum = calloc(EN.size, sizeof(real));
    Critic C, CG; critic_alloc(&C); critic_alloc(&CG); Critic *CGT = calloc(threads, sizeof(Critic)); for (int t = 1; t < threads; t++) critic_alloc(&CGT[t]); critic_init(&C, seed + 5); real *cm = calloc(C.size, sizeof(real)), *cv = calloc(C.size, sizeof(real));
    float mean[OBS_DIM], sd[OBS_DIM]; for (int i = 0; i < OBS_DIM; i++) { mean[i] = 0; sd[i] = 1; }
    double *osum = calloc(OBS_DIM, sizeof(double)), *osq = calloc(OBS_DIM, sizeof(double)), on = 0;
    if (load_from) {
        char p2[1024]; snprintf(p2, sizeof p2, "%s.head", load_from); int had_critic;
        if (load_head(p2, &EN, mean, sd, d, &C, &had_critic)) { fprintf(stderr, "cannot load %s\n", p2); return 1; }
        if (!had_critic) critic_init(&C, seed + 5);
        /* the loaded statistics count as a prior worth 1000 samples, so normalisation does not jump */
        on = 1000; for (int i = 0; i < OBS_DIM; i++) { osum[i] = mean[i] * on; osq[i] = ((double)sd[i] * sd[i] + (double)mean[i] * mean[i]) * on; }
        for (int k = 0; k < ACT_DIM; k++) EN.log_std[k] = (real)log_std0;
    }
    if (eval_n > 0) {   /* --eval N: N episodes of the loaded policy with its mean actions (no hand-up unless --assist), then a summary */
        if (!load_from) { fprintf(stderr, "--eval needs --load\n"); return 2; }
        Env *EV = calloc(eval_n, sizeof(Env)); Gen **G = calloc(eval_n, sizeof(Gen *)); double *ret = calloc(eval_n, sizeof(double)); int *len = calloc(eval_n, sizeof(int)), *fell = calloc(eval_n, sizeof(int)), *goals = calloc(eval_n, sizeof(int));
        for (int e = 0; e < eval_n; e++) { env_reset(&EV[e], &tc, (uint64_t)seed * 7919 + 12345 + e * 1000003ULL); G[e] = gen_new(c); }
        const double te = wall();
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (int g0 = 0; g0 < eval_n; g0 += 8) {
            const int gn = eval_n - g0 < 8 ? eval_n - g0 : 8; float o[OBS_DIM], on_[OBS_DIM], a[8][ACT_DIM]; const float *ap[8]; Env *ep[8]; double rw[8]; real mu[ACT_DIM]; int live[8], alive = gn;
            for (int j = 0; j < gn; j++) { ap[j] = a[j]; ep[j] = &EV[g0 + j]; live[j] = 1; }
            while (alive) {
                for (int j = 0; j < gn; j++) {
                    if (!live[j]) { memset(a[j], 0, sizeof a[j]); continue; }
                    env_obs(ep[j], o); for (int i = 0; i < OBS_DIM; i++) { float zz = (o[i] - mean[i]) / sd[i]; on_[i] = zz > 5 ? 5 : zz < -5 ? -5 : zz; }
                    policy_step(m, &EN, G[g0 + j], on_, mu); for (int k = 0; k < ACT_DIM; k++) a[j][k] = mu[k] > 1 ? 1 : mu[k] < -1 ? -1 : (float)mu[k];
                }
                env_step_many(ep, ap, rw, gn);
                for (int j = 0; j < gn; j++) if (live[j]) {
                    const int e = g0 + j; ret[e] += rw[j];
                    if (EV[e].done) { live[j] = 0; alive--; len[e] = EV[e].step; fell[e] = EV[e].step < EV[e].cfg.steps; goals[e] = EV[e].goals; }
                }
            }
        }
        double rs = 0, ls = 0; int fs = 0; long gs = 0; for (int e = 0; e < eval_n; e++) { rs += ret[e]; ls += len[e]; fs += fell[e]; gs += goals[e]; }
        printf("{\"event\":\"eval\",\"task\":\"%s\",\"policy\":\"%s\",\"episodes\":%d,\"episode_return\":%.1f,\"episode_s\":%.2f,\"ended_early\":%d,\"goals_reached\":%ld,\"assist\":%.3f,\"seconds\":%.1f}\n",
               task, load_from, eval_n, rs / eval_n, ls / eval_n / CTRL_HZ, fs, gs, tc.assist, wall() - te);
        return 0;
    }
    /* rollout state */
    const size_t S_all = (size_t)(c->n_layer / 2) * c->T * NODES(c) * c->n * c->p, HH_all = (size_t)c->n_layer * (c->K - 1) * d;
    Env *ENV = calloc(nenv, sizeof(Env)); Gen **gen = calloc(nenv, sizeof(Gen *));
    float *O = malloc(sizeof(float) * (size_t)nenv * (hor + 1) * OBS_DIM), *A = malloc(sizeof(float) * (size_t)nenv * hor * ACT_DIM);
    float *LP = malloc(sizeof(float) * nenv * hor), *VAL = malloc(sizeof(float) * nenv * (hor + 1)), *R = malloc(sizeof(float) * nenv * hor);
    float *ADV = malloc(sizeof(float) * nenv * hor), *RET = malloc(sizeof(float) * nenv * hor), *VTERM = calloc((size_t)nenv * hor, sizeof(float));
    int *DONE = calloc((size_t)nenv * hor, sizeof(int)), *FELL = calloc((size_t)nenv * hor, sizeof(int));
    real *H0 = malloc(sizeof(real) * nenv * S_all), *HH0 = malloc(sizeof(real) * nenv * HH_all);
    Segment *segs = malloc(sizeof(Segment) * (size_t)nenv * hor);
    Batch B = {.ends = &EN, .eg = EG, .hor = hor, .O = O, .A = A, .LP = LP, .ADV = ADV, .H0 = H0, .HH0 = HH0, .S_all = S_all, .HH_all = HH_all,
               .hscratch = malloc(sizeof(real) * threads * S_all), .hhscratch = malloc(sizeof(real) * threads * HH_all), .clip = clipr, .ent = ent};
    seq_hook = ppo_seq; seq_hook_user = &B;
    uint64_t erng = (uint64_t)seed * 7919 + 1; long episodes = 0, ep_goals = 0; double ep_ret_sum = 0, ep_len_sum = 0; int ep_count = 0;
    double cur_assist = tc.assist, best_assist = 1e9;   /* --assist-auto: the hand-up now, and the lowest a checkpoint was saved at */
    double *ep_ret = calloc(nenv, sizeof(double));
    for (int e = 0; e < nenv; e++) { env_reset(&ENV[e], &tc, erng + e * 1000003ULL); gen[e] = gen_new(c); episodes++; }
    size_t z[N_FIELDS]; sizes(c, z); size_t core = 0; for (int f = 0; f < N_FIELDS; f++) core += z[f]; core -= (size_t)c->vocab * d;
    printf("{\"event\":\"config\",\"task\":\"%s\",\"model\":\"nanotss byte stack (ssm-tree + fff, ternary STE, avx-512)\",\"optimizer\":\"tss_train_step: blockwise muon + adam\",\"core_params\":%zu,\"io_params\":%zu,\"width\":%d,\"trees\":%d,\"depth\":%d,\"envs\":%d,\"horizon\":%d}\n",
           task, core, EN.size, d, c->T, c->D, nenv, hor);
    Hyper hp = {.lr = (real)lr, .beta1 = .9, .beta2 = .999, .eps = 1e-8, .weight_decay = 0, .clip = 1}; long hstep = 0, cstep = 0;
    double t0 = wall(), best = -1, t_roll = 0, t_crit = 0; long it = 0, steps_total = 0;
    while (wall() - t0 < seconds) {
        it++;
        for (int e = 0; e < nenv; e++) gen_snapshot(c, gen[e], H0 + (size_t)e * S_all, HH0 + (size_t)e * HH_all);
        /* rollout: each thread steps groups of 8 environments, their physics together in vector lanes */
        const double tr0 = wall();
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (int g0 = 0; g0 < nenv; g0 += 8) {
            const int gn = nenv - g0 < 8 ? nenv - g0 : 8;
            real mu[ACT_DIM], h1[CH], h2[CH]; float o[OBS_DIM], a[8][ACT_DIM]; const float *ap[8]; Env *ep[8]; double rw[8]; uint64_t rs[8];
            for (int j = 0; j < gn; j++) { const int e = g0 + j; rs[j] = erng ^ ((uint64_t)it << 20) ^ (uint64_t)e * 0x9E3779B97F4A7C15ULL; ap[j] = a[j]; ep[j] = &ENV[e]; }
            for (int t = 0; t <= hor; t++) {
                for (int j = 0; j < gn; j++) {
                    const int e = g0 + j;
                    float *on_ = O + ((size_t)e * (hor + 1) + t) * OBS_DIM; env_obs(&ENV[e], o);
                    for (int i = 0; i < OBS_DIM; i++) { float zz = (o[i] - mean[i]) / sd[i]; on_[i] = zz > 5 ? 5 : zz < -5 ? -5 : zz; }
                    VAL[e * (hor + 1) + t] = (float)critic_fwd(&C, on_, h1, h2);
                    if (t == hor) continue;
                    policy_step(m, &EN, gen[e], on_, mu);
                    double lp = 0; float *at = A + ((size_t)e * hor + t) * ACT_DIM;
                    for (int k = 0; k < ACT_DIM; k++) { const double sg = exp(EN.log_std[k]), zz = gauss(&rs[j]); at[k] = (float)(mu[k] + sg * zz); lp += -0.5 * zz * zz - EN.log_std[k] - 0.9189385332046727; a[j][k] = at[k] > 1 ? 1 : at[k] < -1 ? -1 : at[k]; }
                    LP[e * hor + t] = (float)lp;
                }
                if (t == hor) break;
                env_step_many(ep, ap, rw, gn);
                for (int j = 0; j < gn; j++) {
                    const int e = g0 + j;
                    R[e * hor + t] = (float)rw[j]; ep_ret[e] += rw[j];
                    const int fell = ENV[e].done && ENV[e].step < ENV[e].cfg.steps;
                    DONE[e * hor + t] = ENV[e].done; FELL[e * hor + t] = fell;
                    if (ENV[e].done && !fell) {
                        float ot[OBS_DIM]; env_obs(&ENV[e], o);
                        for (int i = 0; i < OBS_DIM; i++) { float zz = (o[i] - mean[i]) / sd[i]; ot[i] = zz > 5 ? 5 : zz < -5 ? -5 : zz; }
                        VTERM[e * hor + t] = (float)critic_fwd(&C, ot, h1, h2);
                    }
                    if (ENV[e].done) {
                        long ep_id;
#pragma omp critical
                        { ep_ret_sum += ep_ret[e]; ep_len_sum += ENV[e].step; ep_count++; ep_goals += ENV[e].goals; ep_id = ++episodes; }
                        ep_ret[e] = 0; env_reset(&ENV[e], &tc, erng + (uint64_t)ep_id * 7777 + e); ENV[e].cfg.assist = cur_assist; gen_reset(c, gen[e]);
                    }
                }
            }
        }
        t_roll += wall() - tr0;
        steps_total += (long)nenv * hor;
        for (int e = 0; e < nenv; e++) for (int t = 0; t < hor; t += 4) {
            const float *on_ = O + ((size_t)e * (hor + 1) + t) * OBS_DIM;
            for (int i = 0; i < OBS_DIM; i++) { double raw = on_[i] * sd[i] + mean[i]; osum[i] += raw; osq[i] += raw * raw; } on++;
        }
        /* GAE: a fall is terminal, the time limit bootstraps */
        double asum = 0, asq = 0;
        for (int e = 0; e < nenv; e++) {
            double gg = 0;
            for (int t = hor - 1; t >= 0; t--) {
                const int k = e * hor + t;
                const double vnext = FELL[k] ? 0 : DONE[k] ? VTERM[k] : VAL[e * (hor + 1) + t + 1];
                const double delta = R[k] + gamma * vnext - VAL[e * (hor + 1) + t];
                gg = delta + (DONE[k] ? 0 : gamma * lam * gg);
                ADV[k] = (float)gg; RET[k] = (float)(gg + VAL[e * (hor + 1) + t]); asum += gg; asq += gg * gg;
            }
        }
        const double am = asum / (nenv * hor), asd = sqrt(fmax(asq / (nenv * hor) - am * am, 1e-8));
        for (int k = 0; k < nenv * hor; k++) ADV[k] = (float)((ADV[k] - am) / asd);
        /* update: minibatches of whole chunks, cut into segments at episode ends */
        int *order = malloc(nenv * sizeof(int)); for (int e = 0; e < nenv; e++) order[e] = e;
        B.clipped = 0; B.lpdiff = 0; B.lpn = 0; double vloss = 0;
        for (int ep = 0; ep < epochs; ep++) {
            for (int e = nenv - 1; e > 0; e--) { int j = (int)(ur(&erng) * (e + 1)); int tmp = order[e]; order[e] = order[j]; order[j] = tmp; }
            for (int bb = 0; bb < mb; bb++) {
                B.first = ep == 0 && bb == 0;
                const int e0 = bb * nenv / mb, e1 = (bb + 1) * nenv / mb; int nseg = 0;
                for (int ii = e0; ii < e1; ii++) { const int e = order[ii]; int ta = 0;
                    while (ta < hor) { int tb = ta; while (tb < hor - 1 && !DONE[e * hor + tb]) tb++; segs[nseg++] = (Segment){e, ta, tb - ta + 1}; ta = tb + 1; } }
                B.seg = segs;
                for (int t = 0; t < threads; t++) memset(EG[t].all, 0, EG[t].size * sizeof(real));
                hp.step = ++hstep;
                tss_train_step(c, &m->P, &m->Q, &m->G, m->hit, &m->M, &m->V, m->Gt, m->hits, m->ws, threads, NULL, NULL, nseg, hor, &hp, m->k);
                /* input projection and head: Adam on the mean gradient per scored step */
                long nsamp = 0; for (int s = 0; s < nseg; s++) nsamp += segs[s].L;
                memset(gsum, 0, EN.size * sizeof(real)); for (int t = 0; t < threads; t++) for (size_t i = 0; i < EN.size; i++) gsum[i] += EG[t].all[i];
                Hyper hh = hp; hh.lr = (real)hlr; const real c1 = 1 - pow(hh.beta1, hstep), c2 = 1 - pow(hh.beta2, hstep);
                adam(EN.all, gsum, em, ev, EN.size, &hh, (real)(1.0 / nsamp), c1, c2);
                for (int k = 0; k < ACT_DIM; k++) { if (EN.log_std[k] < -3) EN.log_std[k] = -3; if (EN.log_std[k] > 0) EN.log_std[k] = 0; }
                /* critic */
                const double tc0 = wall();
                /* over threads, each into its own gradient, then summed */
#pragma omp parallel num_threads(threads) reduction(+:vloss)
                {
#ifdef _OPENMP
                    const int th = omp_get_thread_num();
#else
                    const int th = 0;
#endif
                    Critic *gt = th ? &CGT[th] : &CG; real h1[CH], h2[CH];
                    memset(gt->all, 0, gt->size * sizeof(real));
#pragma omp for schedule(static)
                    for (int q = 0; q < (e1 - e0) * hor; q++) { const int e = order[e0 + q / hor], t = q % hor;
                        const float *on_ = O + ((size_t)e * (hor + 1) + t) * OBS_DIM; const real vv = critic_fwd(&C, on_, h1, h2), dv = vv - RET[e * hor + t];
                        vloss += dv * dv; critic_bwd(&C, gt, on_, h1, h2, dv); }
                }
                for (int t = 1; t < threads; t++) for (size_t i = 0; i < CG.size; i++) CG.all[i] += CGT[t].all[i];
                Hyper hv = hp; hv.lr = (real)vlr; cstep++; adam(C.all, CG.all, cm, cv, C.size, &hv, (real)(1.0 / ((e1 - e0) * hor)), 1 - pow(.9, cstep), 1 - pow(.999, cstep));
                t_crit += wall() - tc0;
            }
        }
        free(order);
        if (on > 10) for (int i = 0; i < OBS_DIM; i++) { const double mu_ = osum[i] / on, var = osq[i] / on - mu_ * mu_; mean[i] = (float)mu_; sd[i] = (float)sqrt(fmax(var, 1e-2)); }
        const double rew = ep_count ? ep_ret_sum / ep_count : 0, len = ep_count ? ep_len_sum / ep_count / CTRL_HZ : 0;
        double sgm = 0; for (int k = 0; k < ACT_DIM; k++) sgm += exp(EN.log_std[k]) / ACT_DIM;
        if (it % 5 == 1 || wall() - t0 >= seconds) {
            printf("{\"event\":\"iter\",\"iter\":%ld,\"seconds\":%.0f,\"steps\":%ld,\"episode_return\":%.1f,\"episode_s\":%.2f,\"episodes\":%d,\"goals_reached\":%ld,\"clip_frac\":%.3f,\"value_loss\":%.3f,\"std\":%.3f,\"lp_gap\":%.2e,\"rollout_share\":%.2f,\"critic_share\":%.2f}\n",
                   it, wall() - t0, steps_total, rew, len, ep_count, ep_goals, (double)B.clipped / ((double)epochs * nenv * hor), vloss / ((double)epochs * nenv * hor), sgm, B.lpn ? B.lpdiff / B.lpn : 0, t_roll / (wall() - t0), t_crit / (wall() - t0));
            fflush(stdout);
            /* the checkpoint: the best return (with --assist-auto, at the lowest hand-up reached) */
            if (ep_count >= 8 && (assist_auto ? cur_assist < best_assist - 1e-9 || (fabs(cur_assist - best_assist) < 1e-9 && rew > best) : rew > best)) {
                best = rew; best_assist = cur_assist; char p1[1024], p2[1024]; snprintf(p1, sizeof p1, "%s.tss", out); snprintf(p2, sizeof p2, "%s.head", out);
                save(m, p1); save_head(p2, &EN, mean, sd, d, &C);
            }
            if (assist_auto) {   /* as ppo_gpu.py: down 15% (at least 0.005) while episodes last over 60% of their length, up 0.02 (to --assist) under 30% */
                const double frac = ep_count ? len * CTRL_HZ / tc.steps : 0, down = cur_assist - fmax(0.005, 0.15 * cur_assist);
                const double nw = frac > 0.6 ? (down > 0.004 ? down : 0) : frac < 0.3 ? fmin(tc.assist, cur_assist + 0.02) : cur_assist;
                if (nw != cur_assist) { cur_assist = nw; for (int e = 0; e < nenv; e++) ENV[e].cfg.assist = cur_assist; }
                printf("{\"event\":\"assist\",\"iter\":%ld,\"survival\":%.2f,\"assist\":%.3f,\"checkpoint_assist\":%.3f}\n", it, frac, cur_assist, best_assist); fflush(stdout);
            }
            ep_ret_sum = ep_len_sum = 0; ep_count = 0; ep_goals = 0;
        }
    }
    printf("{\"event\":\"done\",\"iterations\":%ld,\"steps\":%ld,\"best_episode_return\":%.1f}\n", it, steps_total, best);
    return 0;
}
