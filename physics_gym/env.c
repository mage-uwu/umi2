#include "env.h"
#include <stdio.h>
#include <stdlib.h>

BHD double env_rand(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (*s >> 11) * (1.0 / 9007199254740992.0); }
static BHD double env_randn(uint64_t *s) { double a = env_rand(s) + 1e-12, b = env_rand(s); return sqrt(-2 * b_log(a)) * b_cos(6.283185307179586 * b); }

/* action range per joint (rad per unit action), indexed by the child body */
BCONST double ACT_SCALES[NB] = {0, 0.5, 0.5, 0.4, 0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 0.6, 1.0, 1.2, 0.6};

BHD void stand_pose(Quat *rel) {   /* the shader page's training pose: arms relaxed 68 degrees down from the T-pose */
    for (int i = 0; i < NB; i++) rel[i] = QID;
    const double a = 68 * M_PI / 180;
    rel[L_UARM] = qexp(v3(0, 0, -a)); rel[R_UARM] = qexp(v3(0, 0, a));
}

BHD void sway_pose(Quat *rel, double t, double amp) {
    stand_pose(rel);
    if (amp <= 0) return;
    const double k = amp / 0.012;   /* idle motion scales with the sway */
    const double shift = 0.035 * k * b_sin(2 * M_PI * 0.07 * t + 0.5);   /* weight from leg to leg */
    const double fb = amp * b_sin(2 * M_PI * 0.23 * t), lr = 0.6 * amp * b_sin(2 * M_PI * 0.17 * t + 1.0) + shift;
    const double br = 0.03 * k * (0.5 + 0.5 * b_sin(2 * M_PI * 0.25 * t));   /* breathing: 0 (out) .. 0.03 (in) */
    const double yaw = k * (0.15 * b_sin(2 * M_PI * 0.045 * t) + 0.06 * b_sin(2 * M_PI * 0.11 * t + 2.0));
    const double nod = k * (0.04 * b_sin(2 * M_PI * 0.08 * t + 1.0) - 0.02);
    const Quat ankle = qexp(v3(fb, 0, lr)), hip = qexp(v3(-0.4 * fb, 0, -0.4 * lr)), spine = qexp(v3(0.25 * fb, 0, 0.25 * lr));
    rel[L_FOOT] = qmul(rel[L_FOOT], ankle); rel[R_FOOT] = qmul(rel[R_FOOT], ankle);
    rel[L_THIGH] = qmul(rel[L_THIGH], hip); rel[R_THIGH] = qmul(rel[R_THIGH], hip);
    /* the unloaded leg softens at the knee as the weight shifts off it */
    rel[L_SHIN] = qmul(rel[L_SHIN], qexp(v3(0.12 * k * fmax(0, -shift / 0.035), 0, 0)));
    rel[R_SHIN] = qmul(rel[R_SHIN], qexp(v3(0.12 * k * fmax(0, shift / 0.035), 0, 0)));
    rel[SPINE] = qmul(rel[SPINE], spine);
    rel[CHEST] = qmul(rel[CHEST], qexp(v3(-br, 0, 0)));                 /* breathing: the chest lifts */
    rel[HEAD] = qmul(rel[HEAD], qexp(v3(nod + 0.5 * br, yaw, 0)));      /* the gaze drifts */
    /* relaxed arms: shoulders rise a little on the in-breath, elbows softly bent */
    rel[L_UARM] = qmul(rel[L_UARM], qexp(v3(0, 0, 0.5 * br))); rel[R_UARM] = qmul(rel[R_UARM], qexp(v3(0, 0, -0.5 * br)));
    rel[L_FARM] = qmul(rel[L_FARM], qexp(v3(0, -0.25 * k, 0))); rel[R_FARM] = qmul(rel[R_FARM], qexp(v3(0, 0.25 * k, 0)));
}

static BHD Quat arc(Vec a, Vec b) {   /* shortest rotation taking unit a to unit b */
    Vec c = vcross(a, b); Quat q = {1 + vdot(a, b), c.x, c.y, c.z}; return qnorm(q);
}
BHD void crawl_pose(Quat *rel, Quat *root_q) {
    for (int i = 0; i < NB; i++) rel[i] = QID;
    /* torso sloping up toward the shoulders (her arms are longer than her thighs, as everyone's are), thighs and
     * arms vertical, shins along the floor */
    const double tilt = M_PI / 2 - 0.75;
    *root_q = qexp(v3(tilt, 0, 0));
    const Vec down = qrot(qconj(*root_q), v3(0, -1, 0));   /* straight down, in the torso's frame */
    rel[L_THIGH] = rel[R_THIGH] = arc(v3(0, -1, 0), down);
    rel[L_SHIN] = rel[R_SHIN] = qexp(v3(M_PI / 2, 0, 0));
    rel[L_FOOT] = rel[R_FOOT] = qexp(v3(0.6, 0, 0));
    rel[L_UARM] = arc(v3(1, 0, 0), down); rel[R_UARM] = arc(v3(-1, 0, 0), down);
    rel[HEAD] = qexp(v3(-tilt + 0.25, 0, 0));   /* face ahead, a little down */
}

static BHD void goto_reset(Env *E);
static BHD Vec hinge_axis(int h) { return h == 1 ? v3(1, 0, 0) : h == 2 ? v3(0, 1, 0) : v3(0, 0, 1); }

BHD double env_phase(const Env *E) { return E->cfg.task == TASK_IMITATE || (E->cfg.task == TASK_GOTO && E->cfg.clip) ? E->t0 + E->W.t : 0; }

/* Reference at clip time t: joint rotations, body world rotations and root, the root carried across loops. */
static BHD void ref_at(const Env *E, double t, Quat *rel, Quat *world, Vec *root) {
    const Clip *c = E->cfg.clip; const double T = (c->frames - 1) / c->fps;
    mref_sample(c, t, root, world, rel);
    double cycles = floor(t / T); Vec loop = vsub(c->root[c->frames - 1], c->root[0]); loop.y = 0;
    *root = vadd(vadd(*root, vscale(loop, cycles)), E->clip_shift);
}

static BHD void settle_to_ground(World *W) {
    double lo = 1e9; for (int i = 0; i < NB; i++) { double y = world_lowest(W, i); if (y < lo) lo = y; }
    for (int i = 0; i < NB; i++) { W->b[i].x.y -= lo; W->b[i].xprev = W->b[i].x; }
}
/* Start states on the floor: on all fours (with noise), or where a limp fall from standing left her. */
static BHD void floor_reset(Env *E) {
    World *W = &E->W; Quat rel[NB], rq;
    if (E->cfg.task == TASK_GETUP && E->cfg.nstyle && env_rand(&E->rng) < E->cfg.style_start) {
        /* start in the pose of a random moment of a person getting up (reference state initialisation): from
         * lying to nearly up, so every stage of rising gets practised; the heading is random */
        const Clip *c = E->cfg.style[(int)(env_rand(&E->rng) * E->cfg.nstyle) % E->cfg.nstyle];
        Quat wq[NB]; Vec root; mref_sample(c, env_rand(&E->rng) * (c->frames - 1) / c->fps, &root, wq, rel);
        rq = qmul(qexp(v3(0, env_rand(&E->rng) * 2 * M_PI, 0)), wq[PELVIS]);
        world_pose(W, v3(0, 1, 0), rq, rel, v3(0, 0, 0));
        for (int i = 0; i < NB; i++) W->target[i] = rel[i];
        settle_to_ground(W);
        for (int s = 0; s < 3; s++) world_step(W);
        W->t = 0; stand_pose(E->ref); return;
    }
    const int fall = E->cfg.task == TASK_GETUP && env_rand(&E->rng) < 0.6;
    crawl_pose(rel, &rq);
    if (fall) stand_pose(rel), rq = QID;
    for (int i = 1; i < NB; i++) {
        /* drawn z, y, x: the order gcc evaluated the three calls in when they were arguments of v3 (an order C leaves
         * open, and nvcc picks differently), so the CPU episodes stay as they were and the GPU build matches them */
        const double nz = env_randn(&E->rng), ny = env_randn(&E->rng), nx = env_randn(&E->rng); Vec n = v3(nx, ny, nz);
        if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
        rel[i] = qmul(rel[i], qexp(vscale(n, 0.15)));
    }
    rq = qmul(qexp(v3(0, env_rand(&E->rng) * 2 * M_PI, 0)), rq);   /* any heading */
    world_pose(W, v3(0, 1, 0), rq, rel, v3(0, 0, 0));
    for (int i = 0; i < NB; i++) W->target[i] = rel[i];
    settle_to_ground(W);
    if (fall) {   /* go limp, get shoved, and settle wherever she lands */
        const double a = env_rand(&E->rng) * 2 * M_PI, sp = 1 + 2 * env_rand(&E->rng);
        for (int i = 0; i < NB; i++) W->b[i].v = vscale(v3(b_cos(a), 0, b_sin(a)), sp * (W->b[i].x.y / 1.2));
        W->kp_scale = 0.03;
        for (int s = 0; s < 75; s++) world_step(W);
        W->kp_scale = 1; W->t = 0;
        for (int i = 1; i < NB; i++) W->target[i] = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q);   /* hold where she lies */
    } else { for (int s = 0; s < 10; s++) world_step(W); W->t = 0; }
    if (E->cfg.task == TASK_GETUP) stand_pose(E->ref); else crawl_pose(E->ref, &rq);
}

BHD void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed) {
    E->cfg = *cfg; E->rng = seed * 0x9E3779B97F4A7C15ULL + 1; E->step = 0; E->done = 0; E->t0 = 0;
    world_init(&E->W, 45, 20, 1.0 / CTRL_HZ);
    if (cfg->arm_tone > 0) for (int i = L_UARM; i <= R_FARM; i++) E->W.tone[i] = cfg->arm_tone;
    for (int k = 0; k < ACT_DIM; k++) E->prev_act[k] = E->filt[k] = 0;
    E->head_v = v3(0, 0, 0); E->style_has_prev = 0;
    E->next_push = 0.5 + env_rand(&E->rng) * 2 * cfg->push_every;
    if (cfg->task == TASK_IMITATE) {
        const Clip *c = cfg->clip; const double T = (c->frames - 1) / c->fps;
        E->t0 = cfg->fixed_start ? 0 : env_rand(&E->rng) * T;
        E->clip_shift = v3(0, 0, 0);
        Quat rel[NB], rel2[NB], wq[NB]; Vec root, root2; const double dt = 1.0 / CTRL_HZ;
        ref_at(E, E->t0, rel, wq, &root); ref_at(E, E->t0 + dt, rel2, wq, &root2);
        E->clip_shift = v3(-root.x, 0, -root.z);   /* the reference starts above the origin */
        root = vadd(root, E->clip_shift); root2 = vadd(root2, E->clip_shift);
        /* body velocities by finite difference of the reference pose */
        World W2; world_init(&W2, 45, 20, dt);
        world_pose(&W2, root2, rel2[0], rel2, v3(0, 0, 0));
        world_pose(&E->W, root, rel[0], rel, v3(0, 0, 0));
        for (int i = 0; i < NB; i++) {
            Body *b = E->W.b + i; b->v = vscale(vsub(W2.b[i].x, b->x), 1 / dt);
            Quat dq = qmul(W2.b[i].q, qconj(b->q)); b->w = vscale(qlog(dq), 1 / dt);
        }
        /* keep the feet out of the ground */
        double lo = 1e9; for (int i = 0; i < NB; i++) { double y = world_lowest(&E->W, i); if (y < lo) lo = y; }
        if (lo < 0) for (int i = 0; i < NB; i++) E->W.b[i].x.y -= lo;
        for (int i = 0; i < NB; i++) E->W.b[i].xprev = E->W.b[i].x;
        ref_at(E, E->t0, E->ref, E->ref_world, &E->ref_root);
        for (int i = 0; i < NB; i++) E->W.target[i] = E->ref[i];
        return;
    }
    if (cfg->task == TASK_CRAWL || cfg->task == TASK_GETUP) { floor_reset(E); return; }
    if (cfg->task == TASK_GOTO) { goto_reset(E); return; }
    stand_pose(E->ref);
    Quat rel[NB];
    for (int i = 0; i < NB; i++) {
        /* drawn z, y, x: the order gcc evaluated the three calls in when they were arguments of v3 (an order C leaves
         * open, and nvcc picks differently), so the CPU episodes stay as they were and the GPU build matches them */
        const double nz = env_randn(&E->rng), ny = env_randn(&E->rng), nx = env_randn(&E->rng); Vec n = v3(nx, ny, nz);
        if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
        rel[i] = i ? qmul(E->ref[i], qexp(vscale(n, cfg->init_noise))) : QID;
        E->W.target[i] = E->ref[i];
    }
    world_pose(&E->W, BODY_DEFS[PELVIS].center, rel[0], rel, v3(0, 0, 0));
    double lo = 1e9; for (int i = 0; i < NB; i++) { double y = world_lowest(&E->W, i); if (y < lo) lo = y; }
    for (int i = 0; i < NB; i++) { E->W.b[i].x.y -= lo; E->W.b[i].xprev = E->W.b[i].x; }
}

/* heading frame: rotation removing the pelvis yaw */
static BHD Quat heading_inv(const World *W, int floor) {
    Vec f = qrot(W->b[PELVIS].q, v3(0, 0, 1));
    if (floor) f = vadd(f, qrot(W->b[PELVIS].q, v3(0, 1, 0)));   /* face down, forward is vertical: up points ahead */
    double yaw = b_atan2(f.x, f.z);
    return qexp(v3(0, -yaw, 0));
}
static BHD int floor_task(const Env *E) { return E->cfg.task == TASK_CRAWL || E->cfg.task == TASK_GETUP; }
/* Elbows and knees bend one way, from straight to fully bent: a hinge target outside that range is held at its
 * edge (the floor skills, whose wide joint ranges could otherwise drive them backwards). Flexion is -y for the
 * left elbow, +y for the right, +x for the knees; the targets are pure rotations about the hinge axis. */
static BHD Quat hinge_limit(int i, Quat q) {
    const int h = BODY_DEFS[i].hinge; const double c = h == 1 ? q.x : h == 2 ? q.y : q.z, ang = 2 * b_atan2(c, q.w);
    const double flex = i == L_FARM ? -ang : ang, mx = i == L_FARM || i == R_FARM ? 2.5 : 2.6;
    const double f = flex < 0 ? 0 : flex > mx ? mx : flex;
    if (f == flex) return q;
    const double a = i == L_FARM ? -f : f; return qexp(vscale(hinge_axis(h), a));
}

static BHD double b_sqrt_h(double x, double z) { return sqrt(x * x + z * z); }
BHD void env_obs(const Env *E, float *o) {
    const World *W = &E->W; Quat hi = heading_inv(W, floor_task(E)); Vec root = W->b[PELVIS].x; int k = 0;
    for (int i = 0; i < NB; i++) {
        const Body *b = W->b + i; Vec p = qrot(hi, vsub(b->x, root));
        if (i == 0) p.y = root.y - fmin(world_lowest(W, L_FOOT), world_lowest(W, R_FOOT));   /* the pelvis: its height above her lowest foot corner */
        Quat q = qmul(hi, b->q); Vec ux = qrot(q, v3(1, 0, 0)), uy = qrot(q, v3(0, 1, 0));
        Vec v = qrot(hi, b->v), w = qrot(hi, b->w);
        o[k++] = p.x; o[k++] = p.y; o[k++] = p.z;
        o[k++] = ux.x; o[k++] = ux.y; o[k++] = ux.z; o[k++] = uy.x; o[k++] = uy.y; o[k++] = uy.z;
        o[k++] = v.x; o[k++] = v.y; o[k++] = v.z; o[k++] = w.x * .2; o[k++] = w.y * .2; o[k++] = w.z * .2;
    }
    const double wt = W->total_mass * 9.81;   /* foot contact from touch: more than 2% of her weight on the foot */
    o[k++] = W->touch[L_FOOT] > 0.02 * wt; o[k++] = W->touch[R_FOOT] > 0.02 * wt;
    Vec cv, c = world_com(W, &cv); Vec d = qrot(hi, vsub(c, root));
    o[k++] = d.x; o[k++] = d.z;
    double ph = 0;
    if (E->cfg.task == TASK_IMITATE || (E->cfg.task == TASK_GOTO && E->cfg.clip)) { const Clip *cl = E->cfg.clip; ph = 2 * M_PI * env_phase(E) / ((cl->frames - 1) / cl->fps); }
    o[k++] = b_sin(ph); o[k++] = b_cos(ph);
    if (E->cfg.task == TASK_GOTO) {   /* the goal from her pelvis, heading frame; far goals at 4 m in their direction */
        const Vec g = qrot(hi, vsub(E->goal, root)); const double r = b_sqrt_h(g.x, g.z), s = r > 4 ? 4 / r : 1;
        o[k++] = g.x * s; o[k++] = g.y; o[k++] = g.z * s; o[k++] = fmin(r, 4);
    } else for (int j = 0; j < OBS_GOAL; j++) o[k++] = 0;
    for (int i = 0; i < NB; i++) o[k++] = fmin(3, W->touch[i] / wt);   /* touch */
}

/* Golden thread (0..1): head lifted at least lift_ref above the pelvis, head upright, shoulders level,
 * little head spin and acceleration. */
static BHD double golden_thread(Env *E, double lift_ref) {
    const World *W = &E->W; const Body *hb = W->b + HEAD;
    double lift = (hb->x.y - W->b[PELVIS].x.y) - lift_ref, sag = lift < 0 ? lift : 0;
    Vec hu = qrot(hb->q, v3(0, 1, 0)), sx = qrot(W->b[CHEST].q, v3(1, 0, 0));
    Vec acc = vscale(vsub(hb->v, E->head_v), CTRL_HZ); E->head_v = hb->v;
    return 0.3 * b_exp(-200 * sag * sag) + 0.25 * b_exp(-10 * (1 - hu.y)) + 0.2 * b_exp(-30 * sx.y * sx.y)
         + 0.15 * b_exp(-0.3 * vdot(hb->w, hb->w)) + 0.1 * b_exp(-0.01 * vdot(acc, acc));
}
/* Muscle effort (rad^2 summed over joints): how far each joint target sits from where the joint is. */
static BHD double effort(const World *W) {
    double e2 = 0;
    for (int i = 1; i < NB; i++) { Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(W->target[i], qconj(rel))); e2 += vdot(e, e); }
    return e2;
}
/* ---------- motion style prior ---------- */
typedef struct { const Clip *c; float *f; int n; } StyleClip;
#ifdef __CUDACC__
__device__ StyleClip style_clips[16]; __device__ int style_n;   /* filled by tssenv_cuda.cu */
#else
static StyleClip style_clips[16]; static int style_n;
#endif
static BHD void pose_features(const Quat *wq, float *f) {   /* joint rotations (child in parent), gravity in the pelvis frame */
    int k = 0;
    for (int i = 1; i < NB; i++) { Quat r = qmul(qconj(wq[BODY_DEFS[i].parent]), wq[i]); f[k++] = r.w; f[k++] = r.x; f[k++] = r.y; f[k++] = r.z; }
    Vec gl = qrot(qconj(wq[PELVIS]), v3(0, -1, 0)); f[k++] = gl.x; f[k++] = gl.y; f[k++] = gl.z;
}
#ifndef __CUDACC__
void env_style_prepare(const Clip *c) {
    for (int i = 0; i < style_n; i++) if (style_clips[i].c == c) return;
    StyleClip *s = style_clips + style_n++; s->c = c; s->n = c->frames; s->f = malloc(sizeof(float) * STYLE_FEAT * c->frames);
    for (int t = 0; t < c->frames; t++) pose_features(c->q + (size_t)t * NB, s->f + (size_t)t * STYLE_FEAT);
}
#endif
static BHD double feat_dist(const float *a, const float *b) {   /* about the summed squared joint angles, plus the tilt */
    double d = 0;
    for (int j = 0; j < NB - 1; j++) { const float *p = a + 4 * j, *q = b + 4 * j; const double c = p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]; d += 4 * (1 - c * c); }
    for (int k = 4 * (NB - 1); k < STYLE_FEAT; k++) d += (a[k] - b[k]) * (a[k] - b[k]);
    return d;
}
/* 0..1: how closely her last two poses match two consecutive frames of a style clip */
static BHD double style_reward(Env *E) {
    if (!E->cfg.nstyle) return 0;
    Quat wq[NB]; for (int i = 0; i < NB; i++) wq[i] = E->W.b[i].q;
    float cur[STYLE_FEAT]; pose_features(wq, cur);
    double best = 1e9;
    for (int s = 0; s < E->cfg.nstyle; s++) for (int i = 0; i < style_n; i++) if (style_clips[i].c == E->cfg.style[s]) {
        const StyleClip *sc = style_clips + i;
        for (int t = 1; t < sc->n; t++) {
            double d = feat_dist(cur, sc->f + (size_t)t * STYLE_FEAT); if (d >= best) continue;
            if (E->style_has_prev) d += feat_dist(E->style_prev, sc->f + (size_t)(t - 1) * STYLE_FEAT);
            if (d < best) best = d;
        }
    }
    if (!E->style_has_prev) best *= 2;
    memcpy(E->style_prev, cur, sizeof cur); E->style_has_prev = 1;
    return b_exp(-best / NJ);
}
static BHD double pose_err(const World *W, const Quat *ref) {
    double e2 = 0; for (int i = 1; i < NB; i++) { Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(rel, qconj(ref[i]))); e2 += vdot(e, e); }
    return e2;
}
/* Height of the lowest corner at one end of a limb box: the far end (1: the hand end of a forearm, the knee end of
 * a shin) or the near end (0: the elbow end of a forearm, the ankle end of a shin). */
static BHD double end_height(const World *W, int i, int far) {
    const Vec h = BODY_DEFS[i].half; const int arm = i == L_FARM || i == R_FARM;
    const double sgn = arm ? (i == L_FARM ? 1 : -1) * (far ? 1 : -1) : (far ? 1 : -1);   /* forearms run along x, shins along y (knee up) */
    double lo = 1e9;
    for (int a = -1; a <= 1; a += 2) for (int b = -1; b <= 1; b += 2) {
        const Vec r = arm ? v3(sgn * h.x, a * h.y, b * h.z) : v3(a * h.x, sgn * h.y, b * h.z);
        const double y = body_point(W->b + i, r).y; if (y < lo) lo = y;
    }
    return lo;
}
static BHD double floor_reward(Env *E, double act2) {
    World *W = &E->W; const double calm = 0.08 * b_exp(-20 * effort(W) / NJ) + 0.07 * b_exp(-2 * act2);
    if (E->step >= E->cfg.steps) E->done = 1;
    Vec cv; world_com(W, &cv);
    if (E->cfg.task == TASK_CRAWL) {
        /* on palms and knees, the torso carried: hand ends of the forearms down, knee
         * ends of the shins down; pelvis, spine, chest, head, thighs, upper arms and elbows off the floor. The
         * pelvis, chest or head touching ends the episode. */
        const World *Wc = W;
        if (world_lowest(Wc, PELVIS) < 0.02 || world_lowest(Wc, CHEST) < 0.02 || world_lowest(Wc, HEAD) < 0.02) { E->done = 1; return 0; }
        Vec f = vadd(qrot(W->b[PELVIS].q, v3(0, 0, 1)), qrot(W->b[PELVIS].q, v3(0, 1, 0))); f.y = 0; f = vscale(f, 1 / (vlen(f) + 1e-9));
        const double v = vdot(cv, f), side = cv.x * f.z - cv.z * f.x, py = W->b[PELVIS].x.y;
        const Vec front = qrot(W->b[CHEST].q, v3(0, 0, 1));
        if (front.y > 0.3) { E->done = 1; return 0; }   /* rolled over */
        const double palmL = end_height(Wc, L_FARM, 1), palmR = end_height(Wc, R_FARM, 1), kneeL = end_height(Wc, L_SHIN, 1), kneeR = end_height(Wc, R_SHIN, 1);
        const double contact = 0.25 * (b_exp(-palmL / 0.03) + b_exp(-palmR / 0.03) + b_exp(-kneeL / 0.03) + b_exp(-kneeR / 0.03));
        /* the forearm's palm side (-y of the box) toward her feet: hands flat with the fingers forward, as a wrist
         * bent back would place them (the hand is part of the forearm box; the viewer draws the wrist) */
        const Vec back = vscale(f, -1);
        const double palms = 0.5 * (fmax(0, vdot(qrot(W->b[L_FARM].q, v3(0, -1, 0)), back)) + fmax(0, vdot(qrot(W->b[R_FARM].q, v3(0, -1, 0)), back)));
        int clear = 1; const int up_parts[] = {PELVIS, SPINE, CHEST, HEAD, L_UARM, R_UARM};   /* (the thighs' knee ends rest on the floor too) */
        for (int j = 0; j < 6; j++) if (world_lowest(Wc, up_parts[j]) < 0.06) clear = 0;
        if (end_height(Wc, L_FARM, 0) < 0.06 || end_height(Wc, R_FARM, 0) < 0.06) clear = 0;   /* elbows up */
        const double yaw_rate = W->b[PELVIS].w.y;
        const double task = 0.3 * b_exp(-b_sq((v - 0.4) / 0.25)) + 0.2 * contact + 0.1 * palms + 0.1 * clear + 0.1 * b_exp(-b_sq((py - 0.45) / 0.12))
             + 0.05 * fmax(0, -front.y) + 0.05 * b_exp(-2 * side * side - 0.5 * yaw_rate * yaw_rate) + calm;
        return E->cfg.nstyle ? 0.65 * task + 0.35 * style_reward(E) : task;
    }
    /* rising pays only with weight on the feet (a leap or a kick reaches the heights too, and must not pay),
     * up to standing height and no further; standing pays for being still, vertically too */
    const int planted = world_lowest(W, L_FOOT) < 0.03 || world_lowest(W, R_FOOT) < 0.03 || world_lowest(W, L_SHIN) < 0.03 || world_lowest(W, R_SHIN) < 0.03
                      || world_lowest(W, L_FARM) < 0.03 || world_lowest(W, R_FARM) < 0.03;
    const int both_feet = world_lowest(W, L_FOOT) < 0.03 && world_lowest(W, R_FOOT) < 0.03;
    const double ph = W->b[PELVIS].x.y, hh = W->b[HEAD].x.y;
    const double hd = planted ? fmin(1, fmax(0, hh / 1.42)) * (hh > 1.5 ? b_exp(-40 * (hh - 1.5) * (hh - 1.5)) : 1) : 0;
    const double pv = planted ? fmin(1, fmax(0, ph / 0.92)) * (ph > 0.95 ? b_exp(-40 * (ph - 0.95) * (ph - 0.95)) : 1) : 0;
    const double up = qrot(W->b[CHEST].q, v3(0, 1, 0)).y;
    const double stand = both_feet && hh > 1.3 && ph > 0.85 ? b_exp(-2 * pose_err(W, E->ref) / NJ) * b_exp(-4 * vdot(cv, cv)) : 0;
    const double task = 0.35 * hd + 0.2 * pv + 0.15 * fmax(up, 0) + 0.15 * stand + calm;
    return E->cfg.nstyle ? 0.7 * task + 0.3 * style_reward(E) : task;
}
static BHD int fallen(const World *W) {   /* heights from the surface under her (y = 0 on open floor) */
    const Vec p = W->b[PELVIS].x; const double sp = support_y(p.x, p.z, p.y);
    if (p.y - sp < 0.55 || W->b[HEAD].x.y - sp < 1.0) return 1;
    for (int i = 0; i < NB; i++) if (i != L_FOOT && i != R_FOOT && i != L_SHIN && i != R_SHIN) {
        const Vec c = W->b[i].x; if (world_lowest(W, i) - support_y(c.x, c.z, c.y) < 0.05) return 1; }
    return 0;
}

/* ---------- TASK_GOTO (physics_gym) ---------- */
static BHD double arena(void) { return room_size() ? 5.4 : 3.0; }   /* half size of the area starts and goals are drawn from */
/* room to stand at (x, z): open floor there and around, nothing from the floor up to head height */
static BHD int free_spot(double x, double z) {
    for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) if (column_hit(x + 0.45 * a, z + 0.45 * b, 0.005, 1.9)) return 0;
    return 1;
}
/* a new goal 1 to 6 m away: on open floor, or (30%) on the top of a box that is flat enough and at most 0.8 m up */
static BHD void goto_goal(Env *E) {
    const double R = arena(); const Vec p = E->W.b[PELVIS].x;
    for (int tries = 0; tries < 64; tries++) {
        Vec g;
        if (room_size() && env_rand(&E->rng) < 0.3) {
            const RoomBox *o = room_at((int)(env_rand(&E->rng) * room_size()) % room_size());
            const double u = 2 * env_rand(&E->rng) - 1; const double w = 2 * env_rand(&E->rng) - 1;
            if (qrot(o->q, v3(0, 1, 0)).y < 0.9) continue;
            const Vec top = vadd(o->c, qrot(o->q, v3(0.7 * u * o->h.x, o->h.y, 0.7 * w * o->h.z)));
            const double y = support_y(top.x, top.z, top.y + 0.01);
            if (y < 0.05 || y > 0.8 || column_hit(top.x, top.z, y + 0.01, y + 1.9)) continue;
            g = v3(top.x, y, top.z);
        } else {
            const double x = (2 * env_rand(&E->rng) - 1) * R; const double z = (2 * env_rand(&E->rng) - 1) * R;
            if (column_hit(x, z, 0.005, 1.9)) continue;
            g = v3(x, 0, z);
        }
        const double dx = g.x - p.x, dz = g.z - p.z, d2 = dx * dx + dz * dz;
        if (d2 < 1.0 || d2 > 36.0) continue;
        E->goal = g; return;
    }
    E->goal = v3(p.x, 0, p.z + 1.5);
}
BHD void env_set_goal(Env *E, Vec goal) { E->goal = goal; }
/* standing on a free spot of floor, any heading, the stand pose with a little noise */
static BHD void goto_reset(Env *E) {
    const double R = arena(); double x = 0, z = 0;
    for (int tries = 0; tries < 64; tries++) {
        const double a = (2 * env_rand(&E->rng) - 1) * R; const double b = (2 * env_rand(&E->rng) - 1) * R;
        if (free_spot(a, b)) { x = a; z = b; break; }
    }
    const double yaw = env_rand(&E->rng) * 2 * M_PI;
    stand_pose(E->ref); Quat rel[NB];
    for (int i = 0; i < NB; i++) {
        const double nz = env_randn(&E->rng), ny = env_randn(&E->rng), nx = env_randn(&E->rng); Vec n = v3(nx, ny, nz);
        if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
        rel[i] = i ? qmul(E->ref[i], qexp(vscale(n, E->cfg.init_noise))) : QID;
        E->W.target[i] = E->ref[i];
    }
    world_pose(&E->W, v3(x, BODY_DEFS[PELVIS].center.y, z), qexp(v3(0, yaw, 0)), rel, v3(0, 0, 0));
    double lo = 1e9; for (int i = 0; i < NB; i++) { const double y = world_lowest(&E->W, i); if (y < lo) lo = y; }
    for (int i = 0; i < NB; i++) { E->W.b[i].x.y -= lo; E->W.b[i].xprev = E->W.b[i].x; }
    E->goals = 0; goto_goal(E);
}
/* toward the goal near walking speed (still once there), facing it, chest up, calm; a bonus and a new goal on arrival */
static BHD double goto_reward(Env *E, double act2) {
    World *W = &E->W; const Body *pb = W->b + PELVIS;
    if (E->step >= E->cfg.steps) E->done = 1;
    if (fallen(W)) { E->done = 1; return 0; }
    const double sp = support_y(pb->x.x, pb->x.z, pb->x.y);
    Vec cv; world_com(W, &cv); const Vec vh = v3(cv.x, 0, cv.z);
    Vec to = vsub(E->goal, pb->x); to.y = 0; const double d = vlen(to); const Vec dir = d > 1e-9 ? vscale(to, 1 / d) : v3(0, 0, 0);
    const double move = d > 0.4 ? b_exp(-b_sq(vdot(vh, dir) - 0.9) / 0.18) : b_exp(-2 * vdot(vh, vh));
    Vec f = qrot(pb->q, v3(0, 0, 1)); f.y = 0; const double fl = vlen(f);
    const double face = d > 0.4 && fl > 1e-9 ? fmax(0, vdot(f, dir) / fl) : 1;
    const double up = fmax(0, qrot(W->b[CHEST].q, v3(0, 1, 0)).y);
    const double task = 0.45 * move + 0.15 * face + 0.15 * up + 0.1 * b_exp(-2 * act2) + 0.15 * b_exp(-20 * effort(W) / NJ);
    double r = E->cfg.nstyle ? 0.7 * task + 0.3 * style_reward(E) : task;
    if (d < 0.3 && fabs(sp - E->goal.y) < 0.2) { r += 2; E->goals++; goto_goal(E); }
    return r;
}

/* env_step in two halves around the physics, so a batch can step its worlds together (env_step_many) */
static BHD double env_begin(Env *E, const float *a) {
    World *W = &E->W; int k = 0; double act2 = 0; const float *f = E->filt;
    if (E->cfg.task == TASK_BALANCE && E->cfg.sway > 0) sway_pose(E->ref, W->t, E->cfg.sway);
    for (int j = 0; j < ACT_DIM; j++) E->filt[j] += (float)ACT_FILTER * (a[j] - E->filt[j]);
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Vec r;
        if (d->hinge) { r = vscale(hinge_axis(d->hinge), f[k++]); }
        else { r = v3(f[k], f[k + 1], f[k + 2]); k += 3; }
        W->target[i] = qmul(E->ref[i], qexp(vscale(r, ACT_SCALES[i] * (E->cfg.act_mul > 0 ? E->cfg.act_mul : 1))));
        if (d->hinge && floor_task(E)) W->target[i] = hinge_limit(i, W->target[i]);
    }
    for (int j = 0; j < ACT_DIM; j++) { double dd = a[j] - E->prev_act[j]; act2 += dd * dd; E->prev_act[j] = a[j]; }
    if (E->cfg.push_max > 0 && W->t >= E->next_push) {
        double ang = env_rand(&E->rng) * 2 * M_PI, f = E->cfg.push_max * (0.3 + 0.7 * env_rand(&E->rng));
        W->push_body = env_rand(&E->rng) < .5 ? CHEST : PELVIS; W->push_force = v3(f * b_cos(ang), 0, f * b_sin(ang)); W->push_left = 0.2;
        E->next_push = W->t + E->cfg.push_every * (0.5 + env_rand(&E->rng));
    }
    if (E->cfg.task == TASK_IMITATE && E->cfg.assist > 0) {   /* training wheels toward the reference root */
        Quat rr[NB], rw[NB]; Vec r2;
        ref_at(E, env_phase(E) + 1.0 / CTRL_HZ, rr, rw, &r2);
        W->assist = E->cfg.assist; W->assist_x = r2; W->assist_q = rw[PELVIS];   /* where the reference is at the end of this step */
        W->assist_v = vscale(vsub(r2, E->ref_root), CTRL_HZ);
    } else if ((E->cfg.task == TASK_GETUP || E->cfg.task == TASK_GOTO) && E->cfg.assist > 0) {   /* a hand up: toward standing height above the surface under her, upright */
        Vec px = W->b[PELVIS].x, f = qrot(W->b[PELVIS].q, v3(0, 0, 1)); f = vadd(f, qrot(W->b[PELVIS].q, v3(0, 1, 0)));
        W->assist = E->cfg.assist; W->assist_x = v3(px.x, support_y(px.x, px.z, px.y) + BODY_DEFS[PELVIS].center.y, px.z); W->assist_v = v3(0, 0, 0);
        W->assist_q = qexp(v3(0, b_atan2(f.x, f.z), 0));
    } else W->assist = 0;
    return act2;
}
static BHD double env_end(Env *E, double act2) {
    World *W = &E->W;
    E->step++;
    if (floor_task(E)) return floor_reward(E, act2);
    if (E->cfg.task == TASK_GOTO) return goto_reward(E, act2);
    int fell = fallen(W);
    if (E->cfg.task == TASK_IMITATE) {
        ref_at(E, env_phase(E), E->ref, E->ref_world, &E->ref_root);
        Vec er = vsub(W->b[PELVIS].x, E->ref_root);
        if (vlen(er) > 0.6) fell = 1;   /* drifted off the reference: end the episode */
        if (fell || E->step >= E->cfg.steps) E->done = 1;
        if (fell) return 0;
        /* pose: body world rotations; velocity: body angular velocity against the reference's */
        double pose = 0, vel = 0, ee = 0;
        Quat rw2[NB], rr2[NB]; Vec root2; ref_at(E, env_phase(E) + 1.0 / CTRL_HZ, rr2, rw2, &root2);
        for (int i = 1; i < NB; i++) {
            Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(rel, qconj(E->ref[i]))); pose += vdot(e, e);
            Vec wr = vscale(qlog(qmul(rw2[i], qconj(E->ref_world[i]))), CTRL_HZ); Vec dw = vsub(W->b[i].w, wr); vel += vdot(dw, dw);
        }
        { Vec e = qlog(qmul(W->b[PELVIS].q, qconj(E->ref_world[PELVIS]))); pose += vdot(e, e); }
        /* hands and feet relative to the pelvis, against forward kinematics of the reference rotations */
        Vec rp = W->b[PELVIS].x; Quat rq = W->b[PELVIS].q; Vec fx[NB]; Quat fq[NB];
        fx[0] = v3(0, 0, 0); fq[0] = QID;
        for (int b = 1; b < NB; b++) {
            const BodyDef *d = BODY_DEFS + b, *pd = BODY_DEFS + d->parent;
            Vec an = vadd(fx[d->parent], qrot(fq[d->parent], vsub(d->anchor, pd->center)));
            fq[b] = qmul(fq[d->parent], E->ref[b]); fx[b] = vadd(an, qrot(fq[b], vsub(d->center, d->anchor)));
        }
        const int EE[4] = {L_FARM, R_FARM, L_FOOT, R_FOOT};
        for (int j = 0; j < 4; j++) { Vec d = vsub(qrot(qconj(rq), vsub(W->b[EE[j]].x, rp)), fx[EE[j]]); ee += vdot(d, d); }
        Vec ce = vsub(W->b[PELVIS].x, E->ref_root);
        double track = 0.5 * b_exp(-2 * pose / NJ) + 0.1 * b_exp(-0.1 * vel / NJ) + 0.2 * b_exp(-10 * ee) + 0.2 * b_exp(-10 * vdot(ce, ce));
        return 0.75 * track + 0.25 * golden_thread(E, qrot(E->ref_world[PELVIS], fx[HEAD]).y);
    }
    if (fell || E->step >= E->cfg.steps) E->done = 1;
    if (fell) return 0;
    /* balance: upper body near the stand pose, torso upright, head over the feet, coming to rest, a calm trunk,
     * smooth commands, the golden thread, and low muscle effort (relaxed, not braced); the legs are free */
    double pose = 0;
    for (int i = SPINE; i <= R_FARM; i++) { Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(rel, qconj(E->ref[i]))); pose += vdot(e, e); }
    Vec feet = vscale(vadd(W->b[L_FOOT].x, W->b[R_FOOT].x), .5), hd = vsub(W->b[HEAD].x, feet); double lean = hd.x * hd.x + hd.z * hd.z;
    Vec up = qrot(W->b[CHEST].q, v3(0, 1, 0)), cv; world_com(W, &cv); double sp = cv.x * cv.x + cv.z * cv.z;
    double spin = vdot(W->b[PELVIS].w, W->b[PELVIS].w) + vdot(W->b[CHEST].w, W->b[CHEST].w);
    const double stand_lift = BODY_DEFS[HEAD].center.y - BODY_DEFS[PELVIS].center.y;
    return 1 + 0.3 * b_exp(-2 * pose / 7) + 0.3 * b_exp(-20 * lean) + 0.2 * fmax(up.y, 0) + 0.2 * b_exp(-4 * sp) + 0.2 * b_exp(-0.5 * spin)
             + 0.2 * b_exp(-2 * act2) + 0.4 * golden_thread(E, stand_lift) + 0.3 * b_exp(-20 * effort(W) / NJ);
}

BHD double env_step(Env *E, const float *a) { const double act2 = env_begin(E, a); world_step(&E->W); return env_end(E, act2); }
#ifndef __CUDACC__
void env_step_many(Env *const *E, const float *const *a, double *reward, int n) {
    double act2[8]; World *W[8];
    for (int s = 0; s < n; s += 8) {
        const int m = n - s < 8 ? n - s : 8;
        for (int j = 0; j < m; j++) { act2[j] = env_begin(E[s + j], a[s + j]); W[j] = &E[s + j]->W; }
        world_step_x8(W, m);
        for (int j = 0; j < m; j++) reward[s + j] = env_end(E[s + j], act2[j]);
    }
}
#endif
