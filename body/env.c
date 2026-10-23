#include "env.h"
#include <stdio.h>

double env_rand(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (*s >> 11) * (1.0 / 9007199254740992.0); }
static double env_randn(uint64_t *s) { double a = env_rand(s) + 1e-12, b = env_rand(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

/* action range per joint (rad per unit action), indexed by the child body */
const double ACT_SCALES[NB] = {0, 0.5, 0.5, 0.4, 0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 0.6, 1.0, 1.2, 0.6};

void stand_pose(Quat *rel) {   /* the shader page's training pose: arms relaxed 68 degrees down from the T-pose */
    for (int i = 0; i < NB; i++) rel[i] = QID;
    const double a = 68 * M_PI / 180;
    rel[L_UARM] = qexp(v3(0, 0, -a)); rel[R_UARM] = qexp(v3(0, 0, a));
}

static Vec hinge_axis(int h) { return h == 1 ? v3(1, 0, 0) : h == 2 ? v3(0, 1, 0) : v3(0, 0, 1); }

double env_phase(const Env *E) { return E->cfg.task == TASK_IMITATE ? E->t0 + E->W.t : 0; }

/* Reference at clip time t: joint rotations, body world rotations and root, the root carried across loops. */
static void ref_at(const Env *E, double t, Quat *rel, Quat *world, Vec *root) {
    const Clip *c = E->cfg.clip; const double T = (c->frames - 1) / c->fps;
    mref_sample(c, t, root, world, rel);
    double cycles = floor(t / T); Vec loop = vsub(c->root[c->frames - 1], c->root[0]); loop.y = 0;
    *root = vadd(vadd(*root, vscale(loop, cycles)), E->clip_shift);
}

void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed) {
    E->cfg = *cfg; E->rng = seed * 0x9E3779B97F4A7C15ULL + 1; E->step = 0; E->done = 0; E->t0 = 0;
    world_init(&E->W, 45, 20, 1.0 / CTRL_HZ);
    for (int k = 0; k < ACT_DIM; k++) E->prev_act[k] = 0;
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
    stand_pose(E->ref);
    Quat rel[NB];
    for (int i = 0; i < NB; i++) {
        Vec n = v3(env_randn(&E->rng), env_randn(&E->rng), env_randn(&E->rng));
        if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
        rel[i] = i ? qmul(E->ref[i], qexp(vscale(n, cfg->init_noise))) : QID;
        E->W.target[i] = E->ref[i];
    }
    world_pose(&E->W, BODY_DEFS[PELVIS].center, rel[0], rel, v3(0, 0, 0));
    double lo = 1e9; for (int i = 0; i < NB; i++) { double y = world_lowest(&E->W, i); if (y < lo) lo = y; }
    for (int i = 0; i < NB; i++) { E->W.b[i].x.y -= lo; E->W.b[i].xprev = E->W.b[i].x; }
}

/* heading frame: rotation removing the pelvis yaw */
static Quat heading_inv(const World *W) {
    Vec f = qrot(W->b[PELVIS].q, v3(0, 0, 1)); double yaw = atan2(f.x, f.z);
    return qexp(v3(0, -yaw, 0));
}

void env_obs(const Env *E, float *o) {
    const World *W = &E->W; Quat hi = heading_inv(W); Vec root = W->b[PELVIS].x; int k = 0;
    for (int i = 0; i < NB; i++) {
        const Body *b = W->b + i; Vec p = qrot(hi, vsub(b->x, root));
        if (i == 0) p.y = root.y;   /* the pelvis reports its height instead of a zero offset */
        Quat q = qmul(hi, b->q); Vec ux = qrot(q, v3(1, 0, 0)), uy = qrot(q, v3(0, 1, 0));
        Vec v = qrot(hi, b->v), w = qrot(hi, b->w);
        o[k++] = p.x; o[k++] = p.y; o[k++] = p.z;
        o[k++] = ux.x; o[k++] = ux.y; o[k++] = ux.z; o[k++] = uy.x; o[k++] = uy.y; o[k++] = uy.z;
        o[k++] = v.x; o[k++] = v.y; o[k++] = v.z; o[k++] = w.x * .2; o[k++] = w.y * .2; o[k++] = w.z * .2;
    }
    o[k++] = world_lowest(W, L_FOOT) < 0.01; o[k++] = world_lowest(W, R_FOOT) < 0.01;
    Vec cv, c = world_com(W, &cv); Vec d = qrot(hi, vsub(c, root));
    o[k++] = d.x; o[k++] = d.z;
    double ph = 0;
    if (E->cfg.task == TASK_IMITATE) { const Clip *cl = E->cfg.clip; ph = 2 * M_PI * env_phase(E) / ((cl->frames - 1) / cl->fps); }
    o[k++] = sin(ph); o[k++] = cos(ph);
}

static int fallen(const World *W) {
    if (W->b[PELVIS].x.y < 0.55 || W->b[HEAD].x.y < 1.0) return 1;
    for (int i = 0; i < NB; i++) if (i != L_FOOT && i != R_FOOT && i != L_SHIN && i != R_SHIN && world_lowest(W, i) < 0.05) return 1;
    return 0;
}

double env_step(Env *E, const float *a) {
    World *W = &E->W; int k = 0; double act2 = 0;
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Vec r;
        if (d->hinge) { r = vscale(hinge_axis(d->hinge), a[k++]); }
        else { r = v3(a[k], a[k + 1], a[k + 2]); k += 3; }
        W->target[i] = qmul(E->ref[i], qexp(vscale(r, ACT_SCALES[i])));
    }
    for (int j = 0; j < ACT_DIM; j++) { double dd = a[j] - E->prev_act[j]; act2 += dd * dd; E->prev_act[j] = a[j]; }
    if (E->cfg.push_max > 0 && W->t >= E->next_push) {
        double ang = env_rand(&E->rng) * 2 * M_PI, f = E->cfg.push_max * (0.3 + 0.7 * env_rand(&E->rng));
        W->push_body = env_rand(&E->rng) < .5 ? CHEST : PELVIS; W->push_force = v3(f * cos(ang), 0, f * sin(ang)); W->push_left = 0.2;
        E->next_push = W->t + E->cfg.push_every * (0.5 + env_rand(&E->rng));
    }
    world_step(W);
    E->step++;
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
        return 0.5 * exp(-2 * pose / NJ) + 0.1 * exp(-0.1 * vel / NJ) + 0.2 * exp(-10 * ee) + 0.2 * exp(-10 * vdot(ce, ce));
    }
    if (fell || E->step >= E->cfg.steps) E->done = 1;
    if (fell) return 0;
    /* balance: upper body near the stand pose, torso upright, head over the feet, coming to rest; legs free */
    double pose = 0;
    for (int i = SPINE; i <= R_FARM; i++) { Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(rel, qconj(E->ref[i]))); pose += vdot(e, e); }
    Vec feet = vscale(vadd(W->b[L_FOOT].x, W->b[R_FOOT].x), .5), hd = vsub(W->b[HEAD].x, feet); double lean = hd.x * hd.x + hd.z * hd.z;
    Vec up = qrot(W->b[CHEST].q, v3(0, 1, 0)), cv; world_com(W, &cv); double sp = cv.x * cv.x + cv.z * cv.z;
    return 1 + 0.4 * exp(-2 * pose / 7) + 0.4 * exp(-20 * lean) + 0.3 * fmax(up.y, 0) + 0.3 * exp(-4 * sp) + 0.2 * exp(-0.5 * act2);
}
