/* tssenv.c: the body environments (env.c, XPBD physics 8 worlds per AVX-512 group) as a shared library for
 * ppo_gpu.py, plus the routing-credit random stream for nanotss_torch.py.
 *
 *   gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm
 *
 * Task setup follows ppo_tss.c's argument handling, so the same flags give the same tasks.
 */
#include "../env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { TaskCfg tc; Motions M, SM; Env *E; int n; uint64_t erng; long episodes; double *ret; } Pool;

void tss_dims(int *obs, int *act, int *hz) { *obs = OBS_DIM; *act = ACT_DIM; *hz = CTRL_HZ; }

/* urand() after tss_work_seed(seed), n values per seed (nanotss_blt.c) */
void tss_urand(const uint64_t *seeds, int B, int n, float *out) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        uint64_t s = seeds[b] * 0x9E3779B97F4A7C15ULL + 1;
        for (int i = 0; i < n; i++) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            out[(size_t)b * n + i] = (float)((s * 2685821657736338717ULL) >> 11) * (float)(1.0 / 9007199254740992.0);
        }
    }
}

/* task: balance | crawl | getup | goto | goto_free | getup_imitate | a clip name in mref (imitation). style: "file.bin:clip1,clip2" or NULL.
 * Zero (or negative for steps) means ppo_tss.c's default. */
Pool *pool_new(int n, const char *task, const char *mref, const char *style, double assist, double push, double act_mul,
               int steps, double style_start, double sway, double arm_tone, int seed, double goal_cone, double goal_tops, int free_heading,
               int chain, double chain_yaw, int fixed_start, double strength, double start0, double drift, double tail, int assist_pos_only) {
    Pool *P = calloc(1, sizeof(Pool)); TaskCfg *tc = &P->tc; tc->chain = chain; tc->chain_yaw = chain_yaw; tc->fixed_start = fixed_start;
    tc->strength = strength; tc->start0 = start0; tc->drift = drift; tc->tail = tail; tc->assist_pos_only = assist_pos_only;
    tc->push_every = 2; tc->steps = 300; tc->init_noise = 0.05;
    tc->goal_cone = goal_cone; tc->goal_tops = goal_tops; tc->free_heading = free_heading; tc->assist = assist; tc->push_max = push; tc->act_mul = act_mul; tc->style_start = style_start; tc->sway = sway; tc->arm_tone = arm_tone;
    if (!strcmp(task, "crawl") || !strcmp(task, "getup")) {
        tc->task = !strcmp(task, "crawl") ? TASK_CRAWL : TASK_GETUP; if (!tc->act_mul) tc->act_mul = 2.5; tc->steps = tc->task == TASK_CRAWL ? 300 : 240;
        if (style && *style) {
            char buf[1024]; snprintf(buf, sizeof buf, "%s", style); char *colon = strchr(buf, ':'); if (!colon) { fprintf(stderr, "--style file:clips\n"); return NULL; }
            *colon = 0; if (mref_load(buf, &P->SM)) { fprintf(stderr, "cannot load %s\n", buf); return NULL; }
            for (char *tok = strtok(colon + 1, ","); tok && tc->nstyle < MAX_STYLE; tok = strtok(NULL, ",")) {
                const Clip *cl = mref_find(&P->SM, tok); if (!cl) { fprintf(stderr, "no clip %s\n", tok); return NULL; }
                env_style_prepare(cl); tc->style[tc->nstyle++] = cl; }
        }
    } else if (!strcmp(task, "goto")) {   /* physics_gym: the walk clip's cycle is the phase input; walking poses are the style */
        tc->task = TASK_GOTO; tc->steps = 600;
        if (mref_load(mref, &P->M) || !(tc->clip = mref_find(&P->M, "walk"))) { fprintf(stderr, "no clip walk in %s\n", mref); return NULL; }
        if (style && *style) {
            char buf[1024]; snprintf(buf, sizeof buf, "%s", style); char *colon = strchr(buf, ':'); if (!colon) { fprintf(stderr, "--style file:clips\n"); return NULL; }
            *colon = 0; if (mref_load(buf, &P->SM)) { fprintf(stderr, "cannot load %s\n", buf); return NULL; }
            for (char *tok = strtok(colon + 1, ","); tok && tc->nstyle < MAX_STYLE; tok = strtok(NULL, ",")) {
                const Clip *cl = mref_find(&P->SM, tok); if (!cl) { fprintf(stderr, "no clip %s\n", tok); return NULL; }
                env_style_prepare(cl); tc->style[tc->nstyle++] = cl; }
        } else { env_style_prepare(tc->clip); tc->style[tc->nstyle++] = tc->clip; }
    } else if (!strcmp(task, "goto_free")) {   /* go-to with no gait prior (env.h): the motion capture judges, it does not lead */
        tc->task = TASK_GOTO; tc->steps = 600; if (!tc->act_mul) tc->act_mul = 1.5;
    } else if (!strcmp(task, "getup_imitate")) {   /* imitation over a clip set (env.h TaskCfg.style): the get-up clips in style */
        tc->task = TASK_IMITATE; tc->init_noise = 0; if (!tc->act_mul) tc->act_mul = 2.5; tc->steps = 240;
        char buf[1024]; snprintf(buf, sizeof buf, "%s", style ? style : ""); char *colon = strchr(buf, ':'); if (!colon) { fprintf(stderr, "getup_imitate: --style file:clips\n"); return NULL; }
        *colon = 0; if (mref_load(buf, &P->SM)) { fprintf(stderr, "cannot load %s\n", buf); return NULL; }
        for (char *tok = strtok(colon + 1, ","); tok && tc->nstyle < MAX_STYLE; tok = strtok(NULL, ",")) {
            const Clip *cl = mref_find(&P->SM, tok); if (!cl) { fprintf(stderr, "no clip %s\n", tok); return NULL; }
            env_style_prepare(cl); tc->style[tc->nstyle++] = cl; }
        tc->clip = tc->style[0];
        if (tc->style_start > 0) env_fall_bank();
    } else if (strcmp(task, "balance")) {
        if (mref_load(mref, &P->M) || !(tc->clip = mref_find(&P->M, task))) { fprintf(stderr, "no clip %s in %s\n", task, mref); return NULL; }
        tc->task = TASK_IMITATE; tc->init_noise = 0;
    }
    if (steps > 0) tc->steps = steps;
    P->n = n; P->E = calloc(n, sizeof(Env)); P->ret = calloc(n, sizeof(double));
    P->erng = (uint64_t)seed * 7919 + 1;
    for (int e = 0; e < n; e++) { env_reset(&P->E[e], tc, P->erng + e * 1000003ULL); P->episodes++; }
    return P;
}

int pool_steps(const Pool *P) { return P->tc.steps; }
void pool_set_starts(const void *s, int n) { env_set_starts((const FallState *)s, n); }   /* before pool_new: the start bank (env.h START_N) */
int tss_start_size(void) { return (int)sizeof(FallState); }

/* A clip of motion capture (world rotations frames x NB x 4 and root frames x 3, at CTRL_HZ) as the discriminator sees it:
 * for each frame but the first and last, the body block of the observation (env_body_feats) with velocities by central
 * difference; and (starts, if not NULL) the same moments as start states, pelvis over the origin. Returns the rows
 * written (frames - 2). */
int amp_clip(const float *root, const float *q, int frames, float *feat, void *starts) {
    if (frames < 3) return 0;
    const double dt = 1.0 / CTRL_HZ;
#pragma omp parallel for schedule(static)
    for (int f = 1; f < frames - 1; f++) {
        World W[3];
        for (int k = 0; k < 3; k++) {
            const int fr = f - 1 + k; Quat wq[NB], rel[NB];
            for (int i = 0; i < NB; i++) { const float *v = q + ((size_t)fr * NB + i) * 4; Quat x = {v[0], v[1], v[2], v[3]}; wq[i] = qnorm(x); }
            rel[0] = wq[0]; for (int i = 1; i < NB; i++) rel[i] = qmul(qconj(wq[BODY_DEFS[i].parent]), wq[i]);
            world_init(W + k, 45, 20, dt);
            world_pose(W + k, v3(root[fr * 3], root[fr * 3 + 1], root[fr * 3 + 2]), rel[0], rel, v3(0, 0, 0));
        }
        World *M = W + 1;
        for (int i = 0; i < NB; i++) {
            Body *b = M->b + i; b->v = vscale(vsub(W[2].b[i].x, W[0].b[i].x), 0.5 / dt);
            b->w = vscale(qlog(qmul(W[2].b[i].q, qconj(W[0].b[i].q))), 0.5 / dt);
        }
        env_body_feats(M, 0, feat + (size_t)(f - 1) * AMP_FEAT);
        if (starts) {
            FallState *s = (FallState *)starts + (f - 1); const Vec p0 = M->b[PELVIS].x;
            for (int i = 0; i < NB; i++) { const Body *b = M->b + i; s->x[i] = v3(b->x.x - p0.x, b->x.y, b->x.z - p0.z); s->q[i] = b->q; s->v[i] = b->v; s->w[i] = b->w; }
        }
    }
    return frames - 2;
}

void pool_obs(const Pool *P, float *out) {
#pragma omp parallel for schedule(static)
    for (int e = 0; e < P->n; e++) env_obs(&P->E[e], out + (size_t)e * OBS_DIM);
}

/* Step every environment with act[n*ACT_DIM] (already clipped to [-1, 1]). Writes the reward, done (fall or time
 * limit), fell, the raw observation reached on a time-limit end (for bootstrapping), and for finished episodes
 * their return and length (ep_len 0 otherwise). Finished environments are reset, as ppo_tss.c resets them. */
void pool_step(Pool *P, const float *act, float *rew, int *done, int *fell, float *term_obs, float *ep_ret, int *ep_len) {
    const int n = P->n;
#pragma omp parallel for schedule(dynamic)
    for (int g0 = 0; g0 < n; g0 += 8) {
        const int gn = n - g0 < 8 ? n - g0 : 8; Env *ep[8]; const float *ap[8]; double rw[8];
        for (int j = 0; j < gn; j++) { ep[j] = &P->E[g0 + j]; ap[j] = act + (size_t)(g0 + j) * ACT_DIM; }
        env_step_many(ep, ap, rw, gn);
        for (int j = 0; j < gn; j++) {
            const int e = g0 + j; Env *E = &P->E[e];
            rew[e] = (float)rw[j]; P->ret[e] += rw[j];
            done[e] = E->done; fell[e] = E->done && E->step < E->cfg.steps; ep_len[e] = 0;
            if (E->done && !fell[e]) env_obs(E, term_obs + (size_t)e * OBS_DIM);
            if (E->done) { ep_ret[e] = (float)P->ret[e]; ep_len[e] = E->step; }
        }
    }
    for (int e = 0; e < n; e++) if (P->E[e].done) {   /* in order, so the seeds do not depend on thread timing */
        P->ret[e] = 0; P->episodes++; env_reset(&P->E[e], &P->tc, P->erng + (uint64_t)P->episodes * 7777 + e);
    }
}

/* debugging the GPU build: an environment's bytes, the constants as tss_consts lays them out, the physics alone */
int pool_env_size(void) { return (int)sizeof(Env); }
void pool_env_copy(const Pool *P, int e, void *out) { memcpy(out, &P->E[e], sizeof(Env)); }
void pool_env_set(Pool *P, int e, const void *in) { memcpy(&P->E[e], in, sizeof(Env)); }
void pool_world_step(Pool *P, int e) { world_step(&P->E[e].W); }
void tss_consts_cpu(double *o) {
    int k = 0;
    for (int i = 0; i < NB; i++) { const BodyDef *d = BODY_DEFS + i; o[k++] = d->mass_frac; o[k++] = d->kp; o[k++] = d->kd; o[k++] = d->hinge; o[k++] = d->parent;
        o[k++] = d->anchor.x; o[k++] = d->anchor.y; o[k++] = d->anchor.z; o[k++] = d->center.x; o[k++] = d->center.y; o[k++] = d->center.z; o[k++] = d->half.x; o[k++] = d->half.y; o[k++] = d->half.z; }
    for (int i = 0; i < NB; i++) o[k++] = ACT_SCALES[i];
}
/* environment e placed limp 1.2 m above the top of room box j, tumbling a little (room parity, room_drop) */
void pool_drop(Pool *P, int e, int j) {
    double v[ROOM_MAX * ROOM_REC]; extern int room_get(double *); const int n = room_get(v); if (j < 0 || j >= n) return;
    const double *b = v + ROOM_REC * j; Quat q = {b[6], b[7], b[8], b[9]}; const double top = b[1] + qrot(q, v3(0, b[4], 0)).y;
    World *W = &P->E[e].W; Quat rel[NB]; stand_pose(rel); W->kp_scale = 0.05;
    world_pose(W, v3(b[0], top + 1.2, b[2]), qexp(v3(1.2, 0.3 + 0.1 * e, 0)), rel, v3(0, 0, 0));
    for (int i = 0; i < NB; i++) W->target[i] = rel[i];
}
int pool_room(const Pool *P, int e) { return P->E[e].W.room; }
void tss_env_offsets(int *jumps, int *room) { *jumps = (int)offsetof(Env, jumps); *room = (int)(offsetof(Env, W) + offsetof(World, room)); }   /* fields read from a batch of Envs */   /* the room of the bank environment e is in */
void pool_set_assist(Pool *P, double a) { P->tc.assist = a; for (int e = 0; e < P->n; e++) P->E[e].cfg.assist = a; }   /* the hand-up assist, from the next step on, in every environment */
void pool_pelvis(const Pool *P, int e, double *out) { const Vec x = P->E[e].W.b[PELVIS].x; out[0] = x.x; out[1] = x.y; out[2] = x.z; }   /* where her pelvis is (world) */
void pool_mtau(const Pool *P, int e, double *out) { memcpy(out, P->E[e].W.mtau, sizeof(double) * NB); }   /* each joint's muscle torque, last substep (N m) */
void pool_free(Pool *P) { free(P->E); free(P->ret); free(P); }
