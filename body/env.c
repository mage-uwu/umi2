#include "env.h"
#include <stdio.h>

double env_rand(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (*s >> 11) * (1.0 / 9007199254740992.0); }
static double env_randn(uint64_t *s) { double a = env_rand(s) + 1e-12, b = env_rand(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

void stand_pose(Quat *rel) {   /* the shader page's training pose: arms relaxed 68 degrees down from the T-pose */
    for (int i = 0; i < NB; i++) rel[i] = QID;
    const double a = 68 * M_PI / 180;
    rel[L_UARM] = qexp(v3(0, 0, -a)); rel[R_UARM] = qexp(v3(0, 0, a));
}

static Vec hinge_axis(int h) { return h == 1 ? v3(1, 0, 0) : h == 2 ? v3(0, 1, 0) : v3(0, 0, 1); }

void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed) {
    E->cfg = *cfg; E->rng = seed * 0x9E3779B97F4A7C15ULL + 1; E->step = 0; E->done = 0;
    world_init(&E->W, 45, 20, 1.0 / CTRL_HZ);
    stand_pose(E->ref);
    Quat rel[NB];
    for (int i = 0; i < NB; i++) {
        Vec n = v3(env_randn(&E->rng), env_randn(&E->rng), env_randn(&E->rng));
        if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
        rel[i] = i ? qmul(E->ref[i], qexp(vscale(n, cfg->init_noise))) : QID;
        E->W.target[i] = E->ref[i];
    }
    world_pose(&E->W, BODY_DEFS[PELVIS].center, rel[0], rel, v3(0, 0, 0));
    /* drop onto the ground: lift so the lowest corner touches y = 0 */
    double lo = 1e9; for (int i = 0; i < NB; i++) { double y = world_lowest(&E->W, i); if (y < lo) lo = y; }
    for (int i = 0; i < NB; i++) { E->W.b[i].x.y -= lo; E->W.b[i].xprev = E->W.b[i].x; }
    E->next_push = 0.5 + env_rand(&E->rng) * 2 * cfg->push_every;
    for (int k = 0; k < ACT_DIM; k++) E->prev_act[k] = 0;
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
}

double env_step(Env *E, const float *a) {
    World *W = &E->W; int k = 0; double act2 = 0;
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Vec r;
        if (d->hinge) { r = vscale(hinge_axis(d->hinge), a[k++]); }
        else { r = v3(a[k], a[k + 1], a[k + 2]); k += 3; }
        W->target[i] = qmul(E->ref[i], qexp(vscale(r, ACT_SCALE)));
    }
    for (int j = 0; j < ACT_DIM; j++) { double dd = a[j] - E->prev_act[j]; act2 += dd * dd; E->prev_act[j] = a[j]; }
    /* pushes */
    if (E->cfg.push_max > 0 && W->t >= E->next_push) {
        double ang = env_rand(&E->rng) * 2 * M_PI, f = E->cfg.push_max * (0.3 + 0.7 * env_rand(&E->rng));
        W->push_body = env_rand(&E->rng) < .5 ? CHEST : PELVIS; W->push_force = v3(f * cos(ang), 0, f * sin(ang)); W->push_left = 0.2;
        E->next_push = W->t + E->cfg.push_every * (0.5 + env_rand(&E->rng));
    }
    world_step(W);
    E->step++;
    /* fall: anything but the feet and shins near the ground, or the pelvis low */
    int fell = W->b[PELVIS].x.y < 0.55 || W->b[HEAD].x.y < 1.0;
    for (int i = 0; i < NB && !fell; i++) if (i != L_FOOT && i != R_FOOT && i != L_SHIN && i != R_SHIN && world_lowest(W, i) < 0.05) fell = 1;
    if (fell || E->step >= E->cfg.steps) E->done = 1;
    if (fell) return 0;
    /* reward: alive, close to the reference pose, head over the feet, smooth actions */
    double pose = 0;
    for (int i = 1; i < NB; i++) { Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(rel, qconj(E->ref[i]))); pose += vdot(e, e); }
    Vec feet = vscale(vadd(W->b[L_FOOT].x, W->b[R_FOOT].x), .5), hd = vsub(W->b[HEAD].x, feet); double lean = hd.x * hd.x + hd.z * hd.z;
    Vec up = qrot(W->b[PELVIS].q, v3(0, 1, 0));
    /* every term is bounded and the total stays positive, so surviving always pays */
    double r = 1 + 0.5 * exp(-2 * pose / NJ) + 0.5 * exp(-20 * lean) + 0.3 * fmax(up.y, 0) + 0.2 * exp(-0.5 * act2);
    return r;
}
