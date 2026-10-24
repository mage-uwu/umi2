/* body.c: XPBD simulation of the 14-body character (see body.h).
 *
 * One substep of length h (Mueller et al. 2020, "Detailed Rigid Body Simulation with XPBD"):
 *   integrate: v += h g (+ push), x += h v; w += h I^-1 (-w x Iw), q += h/2 [w,0] q
 *   solve: joint attachments (rigid), joint drives (compliance 1/kp toward the target), ground contacts (rigid),
 *          room box contacts (physics_gym: rigid, or soft for foam)
 *   velocities from the position change; contacts: no bounce, Coulomb friction on the contact-point velocity;
 *   joint damping applied implicitly on the relative angular velocity
 */
#include "body.h"
#include "joint_limits.h"
#include "muscle.h"         /* MUSCLE_MAX, MUSCLE_SLEW: strength and slew of each joint's muscles (tools/muscle_envelope.py) */   /* JLIM_LO, JLIM_HI: joint ranges from the motion capture (tools/joint_limits.py) */
#include <stdio.h>

#define G_ACC 9.81
#define ASSIST_K 2700.0    /* N/m at full assist (45 kg x 60 s^-2) */
#define ASSIST_KR 2000.0   /* N m / rad at full assist */
/* Rest T-pose, from the VRM humanoid bones (model faces +Z, left = +X). Boxes enclose each segment.
 * Joint gains follow the DeepMimic humanoid (Peng et al. 2018, also about 45 kg); kd = 0.1 kp. */
BCONST BodyDef BODY_DEFS[NB] = {
    /* name      bone             parent  anchor                       centre                      half extents            mass   kp    kd  hinge */
    {"pelvis", "hips", -1, {0, 0.928, 0.004}, {0, 0.915, 0.005}, {0.12, 0.07, 0.09}, 0.120, 0, 0, 0},
    {"spine", "spine", PELVIS, {0, 0.980, 0.016}, {0, 1.036, 0.017}, {0.11, 0.055, 0.08}, 0.140, 1000, 100, 0},
    {"chest", "chest", SPINE, {0, 1.093, 0.019}, {0, 1.205, 0.000}, {0.13, 0.11, 0.09}, 0.170, 1000, 100, 0},
    {"head", "head", CHEST, {0, 1.332, -0.034}, {0, 1.450, -0.010}, {0.08, 0.115, 0.09}, 0.070, 100, 10, 0},
    {"l_uarm", "leftUpperArm", CHEST, {0.105, 1.293, -0.025}, {0.215, 1.293, -0.025}, {0.11, 0.035, 0.035}, 0.026, 400, 40, 0},
    {"l_farm", "leftLowerArm", L_UARM, {0.324, 1.293, -0.025}, {0.472, 1.293, -0.025}, {0.148, 0.03, 0.03}, 0.022, 300, 30, 2},
    {"r_uarm", "rightUpperArm", CHEST, {-0.105, 1.293, -0.025}, {-0.215, 1.293, -0.025}, {0.11, 0.035, 0.035}, 0.026, 400, 40, 0},
    {"r_farm", "rightLowerArm", R_UARM, {-0.324, 1.293, -0.025}, {-0.472, 1.293, -0.025}, {0.148, 0.03, 0.03}, 0.022, 300, 30, 2},
    {"l_thigh", "leftUpperLeg", PELVIS, {0.076, 0.888, 0.000}, {0.076, 0.709, -0.004}, {0.06, 0.18, 0.06}, 0.146, 500, 50, 0},
    {"l_shin", "leftLowerLeg", L_THIGH, {0.076, 0.529, -0.008}, {0.076, 0.319, -0.020}, {0.045, 0.21, 0.045}, 0.048, 500, 50, 1},
    {"l_foot", "leftFoot", L_SHIN, {0.076, 0.109, -0.033}, {0.076, 0.035, 0.035}, {0.04, 0.035, 0.105}, 0.013, 400, 40, 0},
    {"r_thigh", "rightUpperLeg", PELVIS, {-0.076, 0.888, 0.000}, {-0.076, 0.709, -0.004}, {0.06, 0.18, 0.06}, 0.146, 500, 50, 0},
    {"r_shin", "rightLowerLeg", R_THIGH, {-0.076, 0.529, -0.008}, {-0.076, 0.319, -0.020}, {0.045, 0.21, 0.045}, 0.048, 500, 50, 1},
    {"r_foot", "rightFoot", R_SHIN, {-0.076, 0.109, -0.033}, {-0.076, 0.035, 0.035}, {0.04, 0.035, 0.105}, 0.013, 400, 40, 0},
};

BHD void world_init(World *W, double total_mass, int substeps, double dt) {
    memset(W, 0, sizeof(*W));
    W->total_mass = total_mass; W->substeps = substeps; W->h = dt / substeps; W->mu = 0.9; W->push_body = -1; W->kp_scale = W->kd_scale = 1; W->iters = 4; W->limit_k = 5000; W->strength = 1; W->grip_max = GRIP_MAX; W->grip_box[0] = W->grip_box[1] = -1; for (int i = 0; i < NB; i++) W->tone[i] = 1;
    double fsum = 0; for (int i = 0; i < NB; i++) fsum += BODY_DEFS[i].mass_frac;
    for (int i = 0; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *b = W->b + i;
        const double m = total_mass * d->mass_frac / fsum, X = 2 * d->half.x, Y = 2 * d->half.y, Z = 2 * d->half.z;
        b->inv_mass = 1 / m;
        b->inv_inertia = v3(12 / (m * (Y * Y + Z * Z)), 12 / (m * (X * X + Z * Z)), 12 / (m * (X * X + Y * Y)));
        W->target[i] = QID;
    }
}

BHD void world_pose(World *W, Vec root_x, Quat root_q, const Quat *rel, Vec root_v) {
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
static BHD Vec inv_I(const Body *b, Vec v) {
    Vec l = qrot(qconj(b->q), v);
    l = v3(l.x * b->inv_inertia.x, l.y * b->inv_inertia.y, l.z * b->inv_inertia.z);
    return qrot(b->q, l);
}
/* Exact rotation by dw. The rotations here are tiny (a substep's worth), where sin and cos are replaced by
 * their series, accurate to double precision below 0.1 rad; larger ones take the library functions. */
static BHD void rotate_by(Body *b, Vec dw) {
    const double a2 = vdot(dw, dw); Quat e;
    if (a2 < 1e-2) {
        const double s = 0.5 + a2 * (-1.0 / 48 + a2 * (1.0 / 3840 + a2 * (-1.0 / 645120 + a2 * (1.0 / 185794560))));
        e.w = 1 + a2 * (-1.0 / 8 + a2 * (1.0 / 384 + a2 * (-1.0 / 46080 + a2 * (1.0 / 10321920))));
        e.x = dw.x * s; e.y = dw.y * s; e.z = dw.z * s;
    } else e = qexp(dw);
    const Quat q = qmul(e, b->q); const double r = 1 / sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    b->q.w = q.w * r; b->q.x = q.x * r; b->q.y = q.y * r; b->q.z = q.z * r;
}
/* Rotation vector of q (shortest), with atan2 replaced by its series for the small angles of most joint
 * errors (accurate to double precision for s < 0.1 w). */
static BHD Vec qlog_fast(Quat q) {
    if (q.w < 0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
    const double s2 = q.x * q.x + q.y * q.y + q.z * q.z, s = sqrt(s2);
    if (s < 1e-12) return v3(2 * q.x, 2 * q.y, 2 * q.z);
    double f;
    if (s < 0.1 * q.w) {   /* 2 atan(t) / s with t = s / w: 2/w (1 - t^2/3 + t^4/5 - ...) */
        const double t2 = s2 / (q.w * q.w);
        f = 2 / q.w * (1 + t2 * (-1.0 / 3 + t2 * (1.0 / 5 + t2 * (-1.0 / 7 + t2 * (1.0 / 9 + t2 * (-1.0 / 11 + t2 * (1.0 / 13 + t2 * (-1.0 / 15 + t2 * (1.0 / 17)))))))));
    } else f = 2 * b_atan2(s, q.w) / s;
    return vscale(v3(q.x, q.y, q.z), f);
}
/* Generalised inverse mass of a body point along n, and the world rotation per unit impulse (I^-1 (r x n)),
 * which the solvers reuse to apply the correction. */
static BHD double gen_w_k(const Body *b, Vec r, Vec n, Vec *k) { Vec rn = vcross(r, n); *k = inv_I(b, rn); return b->inv_mass + vdot(rn, *k); }

/* Positional constraint: move world points p1 (on b1, offset r1 from its centre) and p2 (on b2, or a fixed
 * point when b2 is NULL) to remove violation C along n (n points from p2 to p1). Returns the multiplier. */
static BHD double solve_pos(Body *b1, Body *b2, Vec r1, Vec r2, Vec n, double C, double alpha, double h, double cap) {
    Vec k1, k2 = {0, 0, 0}; const double w1 = gen_w_k(b1, r1, n, &k1), w2 = b2 ? gen_w_k(b2, r2, n, &k2) : 0, at = alpha / (h * h);
    if (w1 + w2 + at <= 0) return 0;
    double dl = -C / (w1 + w2 + at);
    if (cap >= 0 && fabs(dl) > cap) dl = dl < 0 ? -cap : cap;
    Vec p = vscale(n, dl);   /* I^-1 (r x p) = dl I^-1 (r x n) */
    b1->x = vadd(b1->x, vscale(p, b1->inv_mass)); rotate_by(b1, vscale(k1, dl));
    if (b2) { b2->x = vsub(b2->x, vscale(p, b2->inv_mass)); rotate_by(b2, vscale(k2, -dl)); }
    return dl;
}
/* Angular constraint: remove rotation error theta (b1 relative to b2, about unit axis n); with cap >= 0 the
 * correction is at most cap (an angular impulse times h). Returns the multiplier. */
static BHD double solve_ang(Body *b1, Body *b2, Vec n, double theta, double alpha, double h, double cap) {
    const Vec k1 = inv_I(b1, n), k2 = inv_I(b2, n);
    const double w = vdot(n, k1) + vdot(n, k2), at = alpha / (h * h);
    double dl = -theta / (w + at);
    if (cap >= 0 && fabs(dl) > cap) dl = dl < 0 ? -cap : cap;
    rotate_by(b1, vscale(k1, dl)); rotate_by(b2, vscale(k2, -dl));
    return dl;
}

static BCONST double CORNER[NCORNER][3] = {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1}};
/* the legs as capsules (self-collision between the two legs): pairs, axis, half length of the core segment, radius */
#define LEG_PAIRS 5
static BCONST int LEG_PAIR[LEG_PAIRS][2] = {{L_THIGH, R_THIGH}, {L_SHIN, R_SHIN}, {L_FOOT, R_FOOT}, {L_SHIN, R_FOOT}, {L_FOOT, R_SHIN}};
static BHD int leg_is_foot(int i) { return i == L_FOOT || i == R_FOOT; }
static BHD Vec leg_axis(int i) { return leg_is_foot(i) ? v3(0, 0, 1) : v3(0, 1, 0); }
static BHD double leg_rad(int i) { const Vec h = BODY_DEFS[i].half; return 0.9 * fmin(h.x, leg_is_foot(i) ? h.y : h.z); }
static BHD double leg_half(int i) { const Vec h = BODY_DEFS[i].half; return fmax(0, (leg_is_foot(i) ? h.z : h.y) - leg_rad(i)); }
/* Static friction at the position level (after Mueller et al. 2020, "Detailed Rigid Body Simulation with Extended
 * Position Based Dynamics"), once per body and surface after its corners' normal corrections: the slide along the
 * surface during this substep (from the body's pose at its start, at the touching corners' mean point) is undone when
 * it is at most mu times how deep the body had gone in (the deepest corner): Coulomb's cone in displacements, so a
 * block on a slope holds while tan(slope) <= mu. A bigger slide is left to the kinetic friction of the velocity pass.
 * The slide is undone by moving the body (no turn): turning it about one corner moved its other corners, and
 * correcting corner by corner in a fixed order walked the feet along flat ground. Ramps felt like ice twice: with no
 * static friction (a resting contact crept downhill) and with it judged corner by corner against each corner's share
 * of the load measured as a correction impulse (a light foot's corrections go mostly into turning it, so a loaded
 * foot read as barely pressed and slid). */
static BHD void static_friction(Body *b, Vec rl, Vec n, double depth, double mu) {
    const Vec p = vadd(b->x, qrot(b->q, rl)), p0 = vadd(b->xprev, qrot(b->qprev, rl));
    Vec dp = vsub(p, p0); dp = vsub(dp, vscale(n, vdot(dp, n))); const double lt = vlen(dp);
    if (lt < 1e-12 || lt > mu * depth) return;
    b->x = vsub(b->x, dp);
}
static BHD Vec corner(int i, int k) { const Vec hf = BODY_DEFS[i].half; return v3(CORNER[k][0] * hf.x, CORNER[k][1] * hf.y, CORNER[k][2] * hf.z); }

/* Velocity impulse at a body point along unit n changing the point's velocity there by dv. */
static BHD void vel_impulse(Body *b, Vec r, Vec n, double dv) {
    Vec k; const double w = gen_w_k(b, r, n, &k); if (w <= 0) return;
    const double j = dv / w;
    b->v = vadd(b->v, vscale(n, j * b->inv_mass)); b->w = vadd(b->w, vscale(k, j));
}
/* the room bank: ROOM_BANK rooms of up to ROOM_MAX boxes; each world collides with its own (World.room) */
#ifdef __CUDACC__
__device__ RoomBox room_box[ROOM_BANK][ROOM_MAX]; __device__ int room_n[ROOM_BANK], room_nb;   /* set by tssenv_kernels.cu */
#else
static RoomBox room_box[ROOM_BANK][ROOM_MAX]; static int room_n[ROOM_BANK], room_nb;
int room_set_bank(int r, const double *v, int n) {
    if (r < 0 || r >= ROOM_BANK || n < 0 || n > ROOM_MAX) return -1;
    for (int j = 0; j < n; j++, v += ROOM_REC) {
        RoomBox *o = room_box[r] + j; o->c = v3(v[0], v[1], v[2]); o->h = v3(v[3], v[4], v[5]);
        o->q.w = v[6]; o->q.x = v[7]; o->q.y = v[8]; o->q.z = v[9]; o->mu = v[10]; o->soft = v[11]; o->rad = vlen(o->h); o->grip = (int)v[12];
    }
    room_n[r] = n; if (r >= room_nb) room_nb = r + 1;
    for (int pass = 0; pass < 3; pass++) for (int c = 0; c < NAV_N * NAV_N; c++) nav_pass(r, pass, c);
    return n;
}
int room_set(const double *v, int n) { room_nb = 0; const int k = room_set_bank(0, v, n); if (!n) room_nb = 0; return k; }   /* one room for every world */
int room_count(void) { return room_n[0]; }
int room_get(double *v) {   /* room 0 back as room_set took it */
    for (int j = 0; j < room_n[0]; j++, v += ROOM_REC) { const RoomBox *o = room_box[0] + j; const double t[ROOM_REC] = {o->c.x, o->c.y, o->c.z, o->h.x, o->h.y, o->h.z, o->q.w, o->q.x, o->q.y, o->q.z, o->mu, o->soft, (double)o->grip}; memcpy(v, t, sizeof t); }
    return room_n[0];
}
#endif
BHD int room_banks(void) { return room_nb; }
BHD int room_size(int r) { return room_n[r]; }
BHD const RoomBox *room_at(int r, int j) { return room_box[r] + j; }
/* grip: the point of box o a hand at p would hold, and its distance (1e30 if o has none): a handhold anywhere on it; a
 * ledge on the four edges of its top face, if that face looks up */
static BHD double grip_point(const RoomBox *o, Vec p, Vec *at) {
    const Vec l = qrot(qconj(o->q), vsub(p, o->c)); Vec c;
    if (o->grip == GRIP_HOLD) c = v3(fmax(-o->h.x, fmin(o->h.x, l.x)), fmax(-o->h.y, fmin(o->h.y, l.y)), fmax(-o->h.z, fmin(o->h.z, l.z)));
    else if (o->grip == GRIP_LEDGE && qrot(o->q, v3(0, 1, 0)).y > 0.9) {
        const double ux = fmax(-o->h.x, fmin(o->h.x, l.x)), uz = fmax(-o->h.z, fmin(o->h.z, l.z));
        const Vec e[4] = {v3(ux, o->h.y, o->h.z), v3(ux, o->h.y, -o->h.z), v3(o->h.x, o->h.y, uz), v3(-o->h.x, o->h.y, uz)};
        double best = 1e30; c = e[0];
        for (int k = 0; k < 4; k++) { const Vec d = vsub(l, e[k]); const double d2 = vdot(d, d); if (d2 < best) { best = d2; c = e[k]; } }
    } else return 1e30;
    *at = vadd(o->c, qrot(o->q, c)); return vlen(vsub(p, *at));
}
BHD int grip_find(int r, Vec p, double reach, Vec *at) {
    int best = -1; double bd = reach;
    for (int j = 0; j < room_n[r]; j++) {
        const RoomBox *o = room_box[r] + j; if (!o->grip) continue;
        const Vec dc = vsub(p, o->c); if (vdot(dc, dc) > (o->rad + reach) * (o->rad + reach)) continue;
        Vec a; const double d = grip_point(o, p, &a); if (d <= bd) { bd = d; best = j; *at = a; }
    }
    return best;
}
static BHD Vec hand_local(int hand) { const int i = hand ? R_FARM : L_FARM; return v3(hand ? -BODY_DEFS[i].half.x : BODY_DEFS[i].half.x, 0, 0); }
BHD Vec hand_point(const World *W, int hand) { const Body *b = W->b + (hand ? R_FARM : L_FARM); return vadd(b->x, qrot(b->q, hand_local(hand))); }
/* the heights where the vertical line through (x, z) is inside box o: [*t0, *t1]; 0 if it misses */
static BHD int column_span(const RoomBox *o, double x, double z, double *t0, double *t1) {
    const double dx = x - o->c.x, dz = z - o->c.z; if (dx * dx + dz * dz > o->rad * o->rad) return 0;
    const Quat qi = qconj(o->q); const Vec a = qrot(qi, v3(dx, -o->c.y, dz)), b = qrot(qi, v3(0, 1, 0));
    const double A[3] = {a.x, a.y, a.z}, Bv[3] = {b.x, b.y, b.z}, H[3] = {o->h.x, o->h.y, o->h.z};
    double lo = -1e30, hi = 1e30;
    for (int k = 0; k < 3; k++) {
        if (fabs(Bv[k]) < 1e-12) { if (fabs(A[k]) > H[k]) return 0; continue; }
        double ta = (-H[k] - A[k]) / Bv[k], tb = (H[k] - A[k]) / Bv[k]; if (ta > tb) { const double t = ta; ta = tb; tb = t; }
        if (ta > lo) lo = ta;
        if (tb < hi) hi = tb;
    }
    if (lo > hi) return 0;
    *t0 = lo; *t1 = hi; return 1;
}
BHD double support_y(int r, double x, double z, double ytop) {
    double best = 0, t0, t1;
    for (int j = 0; j < room_n[r]; j++) if (column_span(room_box[r] + j, x, z, &t0, &t1) && t1 <= ytop + 1e-9 && t1 > best) best = t1;
    return best;
}
/* the distance along unit direction d from o to the first box of room r, at most maxd (0 from inside a box): the
 * slab test in each box's own frame */
BHD double ray_dist(int r, Vec o, Vec d, double maxd) {
    double best = maxd;
    for (int j = 0; j < room_n[r]; j++) {
        const RoomBox *b = room_box[r] + j; const Vec oc = vsub(o, b->c);
        const double t_c = vdot(oc, d), dist2 = vdot(oc, oc) - t_c * t_c;   /* broad phase: the ray's line against the bounding sphere */
        if (dist2 > b->rad * b->rad || (t_c > 0 && vdot(oc, oc) > b->rad * b->rad)) continue;
        const Quat qi = qconj(b->q); const Vec lo = qrot(qi, oc), ld = qrot(qi, d);
        const double O[3] = {lo.x, lo.y, lo.z}, D[3] = {ld.x, ld.y, ld.z}, H[3] = {b->h.x, b->h.y, b->h.z};
        double t0 = 0, t1 = best; int hit = 1;
        for (int k = 0; k < 3 && hit; k++) {
            if (fabs(D[k]) < 1e-12) { if (fabs(O[k]) > H[k]) hit = 0; continue; }
            double ta = (-H[k] - O[k]) / D[k], tb = (H[k] - O[k]) / D[k]; if (ta > tb) { const double t = ta; ta = tb; tb = t; }
            if (ta > t0) t0 = ta;
            if (tb < t1) t1 = tb;
            if (t0 > t1) hit = 0;
        }
        if (hit && t0 < best) best = t0;
    }
    return best;
}
BHD int column_hit(int r, double x, double z, double y0, double y1) {
    double t0, t1;
    for (int j = 0; j < room_n[r]; j++) if (column_span(room_box[r] + j, x, z, &t0, &t1) && t1 > y0 && t0 < y1) return 1;
    return 0;
}
#ifdef __CUDACC__
__device__ float room_nav[ROOM_BANK][NAV_N * NAV_N];
#else
static float room_nav[ROOM_BANK][NAV_N * NAV_N];
#endif
BHD float nav_h(int r, int c) { return room_nav[r][c]; }
static BHD float nav_orig(float v) { return v < 0 ? -v - 1 : v; }   /* pass 1 marks a cell to block as -h - 1 */
BHD void nav_pass(int r, int pass, int c) {
    float *g = room_nav[r]; const int ix = c % NAV_N, iz = c / NAV_N;
    if (pass == 0) {   /* the surface height, or blocked */
        const double x = -NAV_HALF + (ix + 0.5) * NAV_CELL, z = -NAV_HALF + (iz + 0.5) * NAV_CELL, h = support_y(r, x, z, 2.0);
        g[c] = column_hit(r, x, z, h + 0.05, h + 1.6) ? NAV_BLOCK : (float)h;
    } else if (pass == 1) {   /* next to a blocked cell or a rise of more than NAV_STEP: marked */
        if (g[c] >= NAV_BLOCK) return;
        const float h = nav_orig(g[c]); int near = 0;
        for (int dz = -1; dz <= 1 && !near; dz++) for (int dx = -1; dx <= 1; dx++) {
            const int nx = ix + dx, nz = iz + dz; if ((!dx && !dz) || nx < 0 || nz < 0 || nx >= NAV_N || nz >= NAV_N) continue;
            const float v = nav_orig(g[nz * NAV_N + nx]); if (v >= NAV_BLOCK || v > h + (float)NAV_STEP) { near = 1; break; }
        }
        if (near) g[c] = -h - 1;
    } else if (g[c] < 0) g[c] = NAV_BLOCK;   /* pass 2: the marks become blocks */
}
#ifdef __CUDACC__
#define BODY_DEBUG(W, k)
#else
void (*body_debug)(World *, int) = 0;
#define BODY_DEBUG(W, k) if (body_debug) body_debug(W, k)
#endif
static BHD void substep(World *W) {
    const double h = W->h;
    int cb[NB * NCORNER], ck[NB * NCORNER], nc = 0; double cl[NB * NCORNER], cv0[NB * NCORNER];
    for (int i = 0; i < NB; i++) {
        Body *b = W->b + i;
        b->v.y -= h * G_ACC;
        if (i == W->push_body && W->push_left > 0) b->v = vadd(b->v, vscale(W->push_force, h * b->inv_mass));
        b->xprev = b->x; b->x = vadd(b->x, vscale(b->v, h));
        /* gyroscopic term, body frame */
        Vec wl = qrot(qconj(b->q), b->w), Il = v3(wl.x / b->inv_inertia.x, wl.y / b->inv_inertia.y, wl.z / b->inv_inertia.z);
        Vec tq = vcross(wl, Il); tq = v3(-tq.x * b->inv_inertia.x, -tq.y * b->inv_inertia.y, -tq.z * b->inv_inertia.z);
        b->w = vadd(b->w, vscale(qrot(b->q, tq), h));
        b->qprev = b->q; rotate_by(b, vscale(b->w, h));
    }
    if (W->push_left > 0) W->push_left -= h;
    BODY_DEBUG(W, 0);
    /* muscles: each joint's drive may add up to budget (torque times h^2) this substep, its strength or its last torque
     * plus what its slew allows, whichever is less (all solver iterations together); budget < 0: unlimited */
    double budget[NB], used[NB];
    for (int i = 0; i < NB; i++) { used[i] = 0; budget[i] = W->strength > 0 ? fmin(MUSCLE_MAX[i] * W->strength, W->mtau[i] + MUSCLE_SLEW[i] * h) * h * h : -1; }
    /* joints: attachment, then hinge alignment, then drive (iterated) */
    for (int it = 0; it < W->iters; it++)
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *c = W->b + i, *p = W->b + d->parent; const BodyDef *pd = BODY_DEFS + d->parent;
        Vec rc = qrot(c->q, vsub(d->anchor, d->center)), rp = qrot(p->q, vsub(d->anchor, pd->center));
        Vec D = vsub(vadd(c->x, rc), vadd(p->x, rp)); double C = vlen(D);
        if (C > 1e-12) solve_pos(c, p, rc, rp, vscale(D, 1 / C), C, 0, h, -1);
        /* drive: target world rotation of the child is q_parent * target; error rotation current * target^-1 */
        Quat tw = qmul(p->q, W->target[i]); Vec e = qlog_fast(qmul(c->q, qconj(tw))); double th = vlen(e);
        if (d->hinge) {   /* hinge: the off-axis components are held rigidly, the axis one by the drive */
            Vec ax = d->hinge == 1 ? v3(1, 0, 0) : d->hinge == 2 ? v3(0, 1, 0) : v3(0, 0, 1);
            Vec aw = qrot(p->q, ax); Vec off = vsub(e, vscale(aw, vdot(e, aw))); double to = vlen(off);
            if (to > 1e-12) solve_ang(c, p, vscale(off, 1 / to), to, 0, h, -1);
            tw = qmul(p->q, W->target[i]); e = qlog_fast(qmul(c->q, qconj(tw))); double ta = vdot(e, aw);
            if (fabs(ta) > 1e-12) used[i] += fabs(solve_ang(c, p, ta > 0 ? aw : vscale(aw, -1), fabs(ta), 1 / (d->kp * W->kp_scale * W->tone[i]), h, budget[i] < 0 ? -1 : fmax(0, budget[i] - used[i])));
        } else if (th > 1e-12) used[i] += fabs(solve_ang(c, p, vscale(e, 1 / th), th, 1 / (d->kp * W->kp_scale * W->tone[i]), h, budget[i] < 0 ? -1 : fmax(0, budget[i] - used[i])));
        if (W->limit_k > 0) {   /* ligament: back inside the joint's range from the capture, if it left it */
            const Vec r = qlog_fast(qmul(qconj(p->q), c->q)), lo = v3(JLIM_LO[i][0], JLIM_LO[i][1], JLIM_LO[i][2]), hi = v3(JLIM_HI[i][0], JLIM_HI[i][1], JLIM_HI[i][2]);
            const Vec rc = v3(r.x < lo.x ? lo.x : r.x > hi.x ? hi.x : r.x, r.y < lo.y ? lo.y : r.y > hi.y ? hi.y : r.y, r.z < lo.z ? lo.z : r.z > hi.z ? hi.z : r.z);
            if (rc.x != r.x || rc.y != r.y || rc.z != r.z) {
                const Vec el = qlog_fast(qmul(c->q, qconj(qmul(p->q, qexp(rc))))); const double tl = vlen(el);
                if (tl > 1e-12) solve_ang(c, p, vscale(el, 1 / tl), tl, 1 / W->limit_k, h, -1);
            }
        }
    }
    if (W->strength > 0) for (int i = 1; i < NB; i++) {   /* this substep's torques, for the next one's slew; activation and its cost */
        W->mtau[i] = used[i] / (h * h); const double a = W->mtau[i] / (MUSCLE_MAX[i] * W->strength); W->act[i] += a; W->metab += a * a;
    }
    if (W->assist > 0) {   /* pelvis support: compliant position and rotation constraints (stable at any stiffness) */
        Body *p = W->b + PELVIS; Vec D = vsub(p->x, W->assist_x); double C = vlen(D);
        if (C > 1e-12) solve_pos(p, NULL, v3(0, 0, 0), v3(0, 0, 0), vscale(D, 1 / C), C, 1 / (W->assist * ASSIST_K), h, -1);
        Vec e = qlog_fast(qmul(p->q, qconj(W->assist_q))); double th = vlen(e);
        if (th > 1e-12) { Vec n = vscale(e, 1 / th); double w = vdot(n, inv_I(p, n)), at = 1 / (W->assist * ASSIST_KR * h * h);
            rotate_by(p, inv_I(p, vscale(n, -th / (w + at)))); }
    }
    for (int k = 0; k < 2; k++) if (W->grip_box[k] >= 0) {   /* grip: the hand held to its point, at most grip_max; lost past 2 GRIP_REACH */
        Body *b = W->b + (k ? R_FARM : L_FARM); const Vec r = qrot(b->q, hand_local(k)), D = vsub(vadd(b->x, r), W->grip_at[k]); const double C = vlen(D);
        if (C > 2 * GRIP_REACH) { W->grip_box[k] = -1; continue; }
        if (C > 1e-12) {
            const double f = fabs(solve_pos(b, NULL, r, v3(0, 0, 0), vscale(D, 1 / C), C, 1e-6, h, W->grip_max * h * h)) / (h * h);
            W->grip_force[k] += f; W->touch[k ? R_FARM : L_FARM] += f;
            if (W->strength > 0) { const double a = f / W->grip_max; W->metab += a * a; }   /* holding on costs like a muscle */
        }
    }
    BODY_DEBUG(W, 1);
    /* ground contacts (y = 0) at box corners; dynamic friction acts in the velocity pass, static friction once per body
     * after its corners (static_friction) */
    for (int i = 0; i < NB; i++) {
        const Vec hf = BODY_DEFS[i].half; if (W->b[i].x.y * W->b[i].x.y > vdot(hf, hf) && W->b[i].x.y > 0) continue;   /* above its bounding sphere: no corner can touch */
        Body *b = W->b + i; double sl = 0; int n0 = 0; Vec rs = v3(0, 0, 0);
        for (int k = 0; k < NCORNER; k++) {
        Vec r = qrot(b->q, corner(i, k)), pw = vadd(b->x, r);
        if (pw.y >= 0) continue;
        const double v0 = vadd(b->v, vcross(b->w, r)).y;   /* its approach speed before the solve (b->v is still the predicted velocity) */
        double ln = solve_pos(b, NULL, r, v3(0, 0, 0), v3(0, -1, 0), -pw.y, 0, h, -1);
        W->touch[i] += fabs(ln) / (h * h);   /* the contact force this substep (N) */
        sl = fmax(sl, -pw.y); rs = vadd(rs, corner(i, k)); n0++;
        cb[nc] = i; ck[nc] = k; cv0[nc] = v0; cl[nc++] = ln;
        }
        if (n0) static_friction(b, vscale(rs, 1.0 / n0), v3(0, 1, 0), sl, W->mu);
    }
    /* room boxes: a corner inside one is pushed out through the face it is nearest, at the box's softness (XPBD
     * compliance; foam gives a little); friction along that face in the velocity pass */
    int rb[ROOM_CONTACTS], rk[ROOM_CONTACTS], nr = 0; double rl[ROOM_CONTACTS], rmu[ROOM_CONTACTS], rv0[ROOM_CONTACTS]; Vec rn[ROOM_CONTACTS];
    const int R = W->room, RN = room_n[R];
    for (int i = 0; i < NB && RN; i++) {
        Body *b = W->b + i; const double br = vlen(BODY_DEFS[i].half);
        for (int j = 0; j < RN; j++) {
            const RoomBox *o = room_box[R] + j; const Vec dc = vsub(b->x, o->c);
            if (vdot(dc, dc) > (br + o->rad) * (br + o->rad)) continue;   /* bounding spheres apart */
            double sl = 0; int n0 = 0; Vec rs = v3(0, 0, 0), ns = v3(0, 0, 0);
            for (int k = 0; k < NCORNER; k++) {
                const Vec r = qrot(b->q, corner(i, k)), l = qrot(qconj(o->q), vsub(vadd(b->x, r), o->c));
                const double dx = o->h.x - fabs(l.x), dy = o->h.y - fabs(l.y), dz = o->h.z - fabs(l.z);
                if (dx <= 0 || dy <= 0 || dz <= 0) continue;
                Vec nl; double d;
                if (dx <= dy && dx <= dz) { d = dx; nl = v3(l.x < 0 ? -1 : 1, 0, 0); }
                else if (dy <= dz) { d = dy; nl = v3(0, l.y < 0 ? -1 : 1, 0); }
                else { d = dz; nl = v3(0, 0, l.z < 0 ? -1 : 1); }
                const Vec n = qrot(o->q, nl);   /* outward */
                const double v0 = vdot(vadd(b->v, vcross(b->w, r)), n);
                const double ln = solve_pos(b, NULL, r, v3(0, 0, 0), vscale(n, -1), d, o->soft, h, -1);
                W->touch[i] += fabs(ln) / (h * h);
                sl = fmax(sl, d); rs = vadd(rs, corner(i, k)); ns = vadd(ns, n); n0++;
                if (nr < ROOM_CONTACTS) { rb[nr] = i; rk[nr] = k; rl[nr] = ln; rn[nr] = n; rmu[nr] = o->mu; rv0[nr] = v0; nr++; }
            }
            if (n0) { const double nl = vlen(ns); if (nl > 1e-9) static_friction(b, vscale(rs, 1.0 / n0), vscale(ns, 1 / nl), sl, o->mu); }
        }
    }
    /* the legs against each other: each leg body is a capsule along its long axis (thighs and shins: y, feet: z), a
     * little thinner than its box; where two from different legs overlap, both are pushed apart (no friction) */
    for (int c = 0; c < LEG_PAIRS; c++) {
        const int ia = LEG_PAIR[c][0], ib = LEG_PAIR[c][1]; Body *A = W->b + ia, *Bb = W->b + ib;
        const Vec ua = qrot(A->q, leg_axis(ia)), ub = qrot(Bb->q, leg_axis(ib));
        const double la = leg_half(ia), lb = leg_half(ib), R = leg_rad(ia) + leg_rad(ib);
        const Vec d0 = vsub(A->x, Bb->x); if (vdot(d0, d0) > (la + lb + R) * (la + lb + R)) continue;
        /* closest points of the segments A->x + s ua (|s| <= la) and Bb->x + t ub (|t| <= lb) */
        const double b_ = vdot(ua, ub), dd = vdot(ua, d0), e = vdot(ub, d0), den = 1 - b_ * b_;
        double sa = den > 1e-9 ? (b_ * e - dd) / den : 0; sa = fmax(-la, fmin(la, sa));
        double tb = e + b_ * sa; tb = fmax(-lb, fmin(lb, tb));
        sa = fmax(-la, fmin(la, b_ * tb - dd));
        const Vec pa = vadd(A->x, vscale(ua, sa)), pb = vadd(Bb->x, vscale(ub, tb)), D = vsub(pb, pa); const double dist = vlen(D);
        if (dist >= R || dist < 1e-9) continue;
        const double ln = solve_pos(A, Bb, vsub(pa, A->x), vsub(pb, Bb->x), vscale(D, 1 / dist), R - dist, 0, h, -1);
        W->touch[ia] += fabs(ln) / (h * h); W->touch[ib] += fabs(ln) / (h * h);
    }
    /* velocities */
    for (int i = 0; i < NB; i++) {
        Body *b = W->b + i;
        b->v = vscale(vsub(b->x, b->xprev), 1 / h);
        Quat dq = qmul(b->q, qconj(b->qprev)); Vec w = vscale(v3(dq.x, dq.y, dq.z), 2 / h);
        b->w = dq.w >= 0 ? w : vscale(w, -1);
    }
    /* contact velocities: no bounce (a corner that came in leaves with no normal speed: the push out of the surface
     * would otherwise throw it back up, feet hopping on slopes), dynamic friction up to mu * normal impulse */
    for (int c = 0; c < nc; c++) {
        Body *b = W->b + cb[c]; Vec r = qrot(b->q, corner(cb[c], ck[c]));
        Vec vp = vadd(b->v, vcross(b->w, r));
        if (vp.y < 0 || cv0[c] < 0) vel_impulse(b, r, v3(0, 1, 0), -vp.y);
        vp = vadd(b->v, vcross(b->w, r)); Vec vt = v3(vp.x, 0, vp.z); double s = vlen(vt);
        if (s > 1e-9) { double dv = fmin(s, W->mu * fabs(cl[c]) / h); vel_impulse(b, r, vscale(vt, 1 / s), -dv); }
    }
    for (int c = 0; c < nr; c++) {   /* the room's contacts, the same along each contact's own normal */
        Body *b = W->b + rb[c]; const Vec r = qrot(b->q, corner(rb[c], rk[c])), n = rn[c];
        Vec vp = vadd(b->v, vcross(b->w, r)); double vn = vdot(vp, n);
        if (vn < 0 || rv0[c] < 0) vel_impulse(b, r, n, -vn);
        vp = vadd(b->v, vcross(b->w, r)); vn = vdot(vp, n); const Vec vt = vsub(vp, vscale(n, vn)); const double s = vlen(vt);
        if (s > 1e-9) { const double dv = fmin(s, rmu[c] * fabs(rl[c]) / h); vel_impulse(b, r, vscale(vt, 1 / s), -dv); }
    }
    if (W->assist > 0) {   /* pelvis support damping, implicit */
        Body *p = W->b + PELVIS; double f = fmin(1, h * W->assist * 20);
        p->v = vadd(p->v, vscale(vsub(W->assist_v, p->v), f)); p->w = vscale(p->w, 1 - f);
    }
    /* joint damping, implicit: remove a fraction min(1, h kd w_eff) of the relative angular velocity */
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Body *c = W->b + i, *p = W->b + d->parent;
        Vec dw = vsub(c->w, p->w); double m = vlen(dw); if (m < 1e-12) continue;
        Vec n = vscale(dw, 1 / m), kc = inv_I(c, n), kp = inv_I(p, n); double weff = vdot(n, kc) + vdot(n, kp);
        double f = h * d->kd * W->kd_scale * W->tone[i] * weff; if (f > 1) f = 1;
        const double j = -f * m / weff;   /* angular impulse on the child along n; the parent gets the opposite */
        c->w = vadd(c->w, vscale(kc, j)); p->w = vsub(p->w, vscale(kp, j));
    }
}

BHD void world_step(World *W) {
    for (int i = 0; i < NB; i++) W->touch[i] = W->act[i] = 0;
    W->metab = 0;
    for (int k = 0; k < 2; k++) {   /* grip: close on what is within reach, or let go */
        W->grip_force[k] = 0;
        if (W->grip_cmd[k] <= 0 || W->grip_max <= 0) W->grip_box[k] = -1;
        else if (W->grip_box[k] < 0 && W->grip_cmd[k] > GRIP_ON) W->grip_box[k] = grip_find(W->room, hand_point(W, k), GRIP_REACH, &W->grip_at[k]);
    }
    for (int s = 0; s < W->substeps; s++) substep(W);
    for (int k = 0; k < 2; k++) W->grip_force[k] /= W->substeps;
    for (int i = 0; i < NB; i++) { W->touch[i] /= W->substeps; W->act[i] /= W->substeps; }   /* means over the step */
    W->metab /= W->substeps;
    W->t += W->h * W->substeps;
}

BHD Vec world_com(const World *W, Vec *vel) {
    Vec c = v3(0, 0, 0), v = v3(0, 0, 0); double m = 0;
    for (int i = 0; i < NB; i++) { double mi = 1 / W->b[i].inv_mass; c = vadd(c, vscale(W->b[i].x, mi)); v = vadd(v, vscale(W->b[i].v, mi)); m += mi; }
    if (vel) *vel = vscale(v, 1 / m);
    return vscale(c, 1 / m);
}
BHD double world_lowest(const World *W, int i) {
    double lo = 1e9; for (int k = 0; k < NCORNER; k++) { double y = body_point(W->b + i, corner(i, k)).y; if (y < lo) lo = y; } return lo;
}
