/* body.c: XPBD simulation of the 14-body character (see body.h).
 *
 * One substep of length h (Mueller et al. 2020, "Detailed Rigid Body Simulation with XPBD"):
 *   integrate: v += h g (+ push), x += h v; w += h I^-1 (-w x Iw), q += h/2 [w,0] q
 *   solve: joint attachments (rigid), joint drives (compliance 1/kp toward the target), ground contacts (rigid)
 *   velocities from the position change; contacts: no bounce, Coulomb friction on the contact-point velocity;
 *   joint damping applied implicitly on the relative angular velocity
 */
#include "body.h"
#include <stdio.h>

#define G_ACC 9.81
/* Rest T-pose, from the VRM humanoid bones (model faces +Z, left = +X). Boxes enclose each segment. */
const BodyDef BODY_DEFS[NB] = {
    /* name      bone             parent  anchor                       centre                      half extents            mass   kp    kd  hinge */
    {"pelvis", "hips", -1, {0, 0.928, 0.004}, {0, 0.915, 0.005}, {0.12, 0.07, 0.09}, 0.120, 0, 0, 0},
    {"spine", "spine", PELVIS, {0, 0.980, 0.016}, {0, 1.036, 0.017}, {0.11, 0.055, 0.08}, 0.140, 300, 30, 0},
    {"chest", "chest", SPINE, {0, 1.093, 0.019}, {0, 1.205, 0.000}, {0.13, 0.11, 0.09}, 0.170, 300, 30, 0},
    {"head", "head", CHEST, {0, 1.332, -0.034}, {0, 1.450, -0.010}, {0.08, 0.115, 0.09}, 0.070, 60, 6, 0},
    {"l_uarm", "leftUpperArm", CHEST, {0.105, 1.293, -0.025}, {0.215, 1.293, -0.025}, {0.11, 0.035, 0.035}, 0.026, 120, 12, 0},
    {"l_farm", "leftLowerArm", L_UARM, {0.324, 1.293, -0.025}, {0.472, 1.293, -0.025}, {0.148, 0.03, 0.03}, 0.022, 60, 6, 2},
    {"r_uarm", "rightUpperArm", CHEST, {-0.105, 1.293, -0.025}, {-0.215, 1.293, -0.025}, {0.11, 0.035, 0.035}, 0.026, 120, 12, 0},
    {"r_farm", "rightLowerArm", R_UARM, {-0.324, 1.293, -0.025}, {-0.472, 1.293, -0.025}, {0.148, 0.03, 0.03}, 0.022, 60, 6, 2},
    {"l_thigh", "leftUpperLeg", PELVIS, {0.076, 0.888, 0.000}, {0.076, 0.709, -0.004}, {0.06, 0.18, 0.06}, 0.146, 400, 40, 0},
    {"l_shin", "leftLowerLeg", L_THIGH, {0.076, 0.529, -0.008}, {0.076, 0.319, -0.020}, {0.045, 0.21, 0.045}, 0.048, 300, 30, 1},
    {"l_foot", "leftFoot", L_SHIN, {0.076, 0.109, -0.033}, {0.076, 0.035, 0.035}, {0.04, 0.035, 0.105}, 0.013, 150, 15, 0},
    {"r_thigh", "rightUpperLeg", PELVIS, {-0.076, 0.888, 0.000}, {-0.076, 0.709, -0.004}, {0.06, 0.18, 0.06}, 0.146, 400, 40, 0},
    {"r_shin", "rightLowerLeg", R_THIGH, {-0.076, 0.529, -0.008}, {-0.076, 0.319, -0.020}, {0.045, 0.21, 0.045}, 0.048, 300, 30, 1},
    {"r_foot", "rightFoot", R_SHIN, {-0.076, 0.109, -0.033}, {-0.076, 0.035, 0.035}, {0.04, 0.035, 0.105}, 0.013, 150, 15, 0},
};

void world_init(World *W, double total_mass, int substeps, double dt) {
    memset(W, 0, sizeof(*W));
    W->total_mass = total_mass; W->substeps = substeps; W->h = dt / substeps; W->mu = 0.9; W->push_body = -1; W->kp_scale = W->kd_scale = 1; W->iters = 4;
    double fsum = 0; for (int i = 0; i < NB; i++) fsum += BODY_DEFS[i].mass_frac;
    for (int i = 0; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *b = W->b + i;
        const double m = total_mass * d->mass_frac / fsum, X = 2 * d->half.x, Y = 2 * d->half.y, Z = 2 * d->half.z;
        b->inv_mass = 1 / m;
        b->inv_inertia = v3(12 / (m * (Y * Y + Z * Z)), 12 / (m * (X * X + Z * Z)), 12 / (m * (X * X + Y * Y)));
        W->target[i] = QID;
    }
}

void world_pose(World *W, Vec root_x, Quat root_q, const Quat *rel, Vec root_v) {
    for (int i = 0; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *b = W->b + i;
        if (d->parent < 0) { b->q = root_q; b->x = root_x; }
        else {
            const Body *p = W->b + d->parent; const BodyDef *pd = BODY_DEFS + d->parent;
            Vec a = vadd(p->x, qrot(p->q, vsub(d->anchor, pd->center)));
            b->q = qnorm(qmul(p->q, rel ? rel[i] : QID));
            b->x = vadd(a, qrot(b->q, vsub(d->center, d->anchor)));
        }
        b->v = root_v; b->w = v3(0, 0, 0); b->xprev = b->x; b->qprev = b->q;
    }
}

/* world-space inverse inertia applied to a vector */
static Vec inv_I(const Body *b, Vec v) {
    Vec l = qrot(qconj(b->q), v);
    l = v3(l.x * b->inv_inertia.x, l.y * b->inv_inertia.y, l.z * b->inv_inertia.z);
    return qrot(b->q, l);
}
static void rotate_by(Body *b, Vec dw) { b->q = qnorm(qmul(qexp(dw), b->q)); }   /* exact rotation by dw */
static double gen_w(const Body *b, Vec r, Vec n) { Vec rn = vcross(r, n); return b->inv_mass + vdot(rn, inv_I(b, rn)); }

/* Positional constraint: move world points p1 (on b1, offset r1 from its centre) and p2 (on b2, or a fixed
 * point when b2 is NULL) to remove violation C along n (n points from p2 to p1). Returns the multiplier. */
static double solve_pos(Body *b1, Body *b2, Vec r1, Vec r2, Vec n, double C, double alpha, double h, double cap) {
    const double w1 = gen_w(b1, r1, n), w2 = b2 ? gen_w(b2, r2, n) : 0, at = alpha / (h * h);
    if (w1 + w2 + at <= 0) return 0;
    double dl = -C / (w1 + w2 + at);
    if (cap >= 0 && fabs(dl) > cap) dl = dl < 0 ? -cap : cap;
    Vec p = vscale(n, dl);
    b1->x = vadd(b1->x, vscale(p, b1->inv_mass)); rotate_by(b1, inv_I(b1, vcross(r1, p)));
    if (b2) { b2->x = vsub(b2->x, vscale(p, b2->inv_mass)); rotate_by(b2, vscale(inv_I(b2, vcross(r2, p)), -1)); }
    return dl;
}
/* Angular constraint: remove rotation error theta (b1 relative to b2, about unit axis n). */
static void solve_ang(Body *b1, Body *b2, Vec n, double theta, double alpha, double h) {
    const double w = vdot(n, inv_I(b1, n)) + vdot(n, inv_I(b2, n)), at = alpha / (h * h);
    const double dl = -theta / (w + at);
    Vec p = vscale(n, dl);
    rotate_by(b1, inv_I(b1, p)); rotate_by(b2, vscale(inv_I(b2, p), -1));
}

static const double CORNER[NCORNER][3] = {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1}};
static Vec corner(int i, int k) { const Vec hf = BODY_DEFS[i].half; return v3(CORNER[k][0] * hf.x, CORNER[k][1] * hf.y, CORNER[k][2] * hf.z); }

/* Velocity impulse at a body point along unit n changing the point's velocity there by dv. */
static void vel_impulse(Body *b, Vec r, Vec n, double dv) {
    const double w = gen_w(b, r, n); if (w <= 0) return;
    Vec J = vscale(n, dv / w);
    b->v = vadd(b->v, vscale(J, b->inv_mass)); b->w = vadd(b->w, inv_I(b, vcross(r, J)));
}
void (*body_debug)(World *, int) = 0;
static void substep(World *W) {
    const double h = W->h;
    int cb[NB * NCORNER], ck[NB * NCORNER], nc = 0; double cl[NB * NCORNER];
    for (int i = 0; i < NB; i++) {
        Body *b = W->b + i;
        b->v.y -= h * G_ACC;
        if (i == W->push_body && W->push_left > 0) b->v = vadd(b->v, vscale(W->push_force, h * b->inv_mass));
        if (i == PELVIS) { b->v = vadd(b->v, vscale(W->assist_f, h * b->inv_mass)); b->w = vadd(b->w, vscale(inv_I(b, W->assist_t), h)); }
        b->xprev = b->x; b->x = vadd(b->x, vscale(b->v, h));
        /* gyroscopic term, body frame */
        Vec wl = qrot(qconj(b->q), b->w), Il = v3(wl.x / b->inv_inertia.x, wl.y / b->inv_inertia.y, wl.z / b->inv_inertia.z);
        Vec tq = vcross(wl, Il); tq = v3(-tq.x * b->inv_inertia.x, -tq.y * b->inv_inertia.y, -tq.z * b->inv_inertia.z);
        b->w = vadd(b->w, vscale(qrot(b->q, tq), h));
        b->qprev = b->q; rotate_by(b, vscale(b->w, h));
    }
    if (W->push_left > 0) W->push_left -= h;
    if (body_debug) body_debug(W, 0);
    /* joints: attachment, then hinge alignment, then drive (iterated) */
    for (int it = 0; it < W->iters; it++)
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *c = W->b + i, *p = W->b + d->parent; const BodyDef *pd = BODY_DEFS + d->parent;
        Vec rc = qrot(c->q, vsub(d->anchor, d->center)), rp = qrot(p->q, vsub(d->anchor, pd->center));
        Vec D = vsub(vadd(c->x, rc), vadd(p->x, rp)); double C = vlen(D);
        if (C > 1e-12) solve_pos(c, p, rc, rp, vscale(D, 1 / C), C, 0, h, -1);
        /* drive: target world rotation of the child is q_parent * target; error rotation current * target^-1 */
        Quat tw = qmul(p->q, W->target[i]); Vec e = qlog(qmul(c->q, qconj(tw))); double th = vlen(e);
        if (d->hinge) {   /* hinge: the off-axis components are held rigidly, the axis one by the drive */
            Vec ax = d->hinge == 1 ? v3(1, 0, 0) : d->hinge == 2 ? v3(0, 1, 0) : v3(0, 0, 1);
            Vec aw = qrot(p->q, ax); Vec off = vsub(e, vscale(aw, vdot(e, aw))); double to = vlen(off);
            if (to > 1e-12) solve_ang(c, p, vscale(off, 1 / to), to, 0, h);
            tw = qmul(p->q, W->target[i]); e = qlog(qmul(c->q, qconj(tw))); double ta = vdot(e, aw);
            if (fabs(ta) > 1e-12) solve_ang(c, p, ta > 0 ? aw : vscale(aw, -1), fabs(ta), 1 / (d->kp * W->kp_scale), h);
        } else if (th > 1e-12) solve_ang(c, p, vscale(e, 1 / th), th, 1 / (d->kp * W->kp_scale), h);
    }
    if (body_debug) body_debug(W, 1);
    /* ground contacts (y = 0) at box corners; friction acts in the velocity pass */
    for (int i = 0; i < NB; i++) for (int k = 0; k < NCORNER; k++) {
        Body *b = W->b + i; Vec r = qrot(b->q, corner(i, k)), pw = vadd(b->x, r);
        if (pw.y >= 0) continue;
        double ln = solve_pos(b, NULL, r, v3(0, 0, 0), v3(0, -1, 0), -pw.y, 0, h, -1);
        cb[nc] = i; ck[nc] = k; cl[nc++] = ln;
    }
    /* velocities */
    for (int i = 0; i < NB; i++) {
        Body *b = W->b + i;
        b->v = vscale(vsub(b->x, b->xprev), 1 / h);
        Quat dq = qmul(b->q, qconj(b->qprev)); Vec w = vscale(v3(dq.x, dq.y, dq.z), 2 / h);
        b->w = dq.w >= 0 ? w : vscale(w, -1);
    }
    /* contact velocities: no bounce, dynamic friction up to mu * normal impulse */
    for (int c = 0; c < nc; c++) {
        Body *b = W->b + cb[c]; Vec r = qrot(b->q, corner(cb[c], ck[c]));
        Vec vp = vadd(b->v, vcross(b->w, r));
        if (vp.y < 0) vel_impulse(b, r, v3(0, 1, 0), -vp.y);
        vp = vadd(b->v, vcross(b->w, r)); Vec vt = v3(vp.x, 0, vp.z); double s = vlen(vt);
        if (s > 1e-9) { double dv = fmin(s, W->mu * fabs(cl[c]) / h); vel_impulse(b, r, vscale(vt, 1 / s), -dv); }
    }
    /* joint damping, implicit: remove a fraction min(1, h kd w_eff) of the relative angular velocity */
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *c = W->b + i, *p = W->b + d->parent;
        Vec dw = vsub(c->w, p->w); double m = vlen(dw); if (m < 1e-12) continue;
        Vec n = vscale(dw, 1 / m); double weff = vdot(n, inv_I(c, n)) + vdot(n, inv_I(p, n));
        double f = h * d->kd * W->kd_scale * weff; if (f > 1) f = 1;
        Vec imp = vscale(n, -f * m / weff);   /* angular impulse on the child; the parent gets the opposite */
        c->w = vadd(c->w, inv_I(c, imp)); p->w = vsub(p->w, inv_I(p, imp));
    }
}

void world_step(World *W) { for (int s = 0; s < W->substeps; s++) substep(W); W->t += W->h * W->substeps; }

Vec world_com(const World *W, Vec *vel) {
    Vec c = v3(0, 0, 0), v = v3(0, 0, 0); double m = 0;
    for (int i = 0; i < NB; i++) { double mi = 1 / W->b[i].inv_mass; c = vadd(c, vscale(W->b[i].x, mi)); v = vadd(v, vscale(W->b[i].v, mi)); m += mi; }
    if (vel) *vel = vscale(v, 1 / m);
    return vscale(c, 1 / m);
}
double world_lowest(const World *W, int i) {
    double lo = 1e9; for (int k = 0; k < NCORNER; k++) { double y = body_point(W->b + i, corner(i, k)).y; if (y < lo) lo = y; } return lo;
}
