/* body.h: a 14-body physical character (the ANIME NEURAL SHADER model's skeleton) simulated with XPBD.
 *
 * Every body is a box whose frame is world-aligned in the model's rest T-pose, so a body's world rotation
 * is exactly the bone's world rotation of the VRM / motion clips (relative to rest). Joints are ball
 * joints (point attachment) with a PD "muscle" per joint: an XPBD compliant orientation constraint toward
 * a target rotation (stiffness kp) plus implicit damping kd. Gravity, ground contact with Coulomb
 * friction, no self-collision. Units: metres, kilograms, seconds. The model faces +Z, its left is +X.
 *
 * The same equations are mirrored in body.js for the browser, so a policy trained here runs there.
 */
#ifndef BODY_H
#define BODY_H
#include <math.h>
#include <string.h>

#define NB 14          /* bodies */
#define NJ 13          /* joints: body b>0 hangs from its parent */
#define NCORNER 8

typedef struct { double w, x, y, z; } Quat;
typedef struct { double x, y, z; } Vec;

enum { PELVIS, SPINE, CHEST, HEAD, L_UARM, L_FARM, R_UARM, R_FARM, L_THIGH, L_SHIN, L_FOOT, R_THIGH, R_SHIN, R_FOOT };

typedef struct {
    const char *name, *bone;   /* bone: the VRM humanoid bone this body drives */
    int parent;
    Vec anchor;                /* joint position with the parent, rest pose, world */
    Vec center, half;          /* box centre and half extents, rest pose, world */
    double mass_frac, kp, kd;  /* share of total mass; joint stiffness (N m / rad) and damping (N m s / rad) */
    int hinge;                 /* 0 ball; else 1+axis (1=x, 2=y, 3=z) of a one-dof joint (knee, elbow) */
} BodyDef;

typedef struct {
    Vec x, v, w, xprev;        /* centre of mass position, velocity, angular velocity (world) */
    Quat q, qprev;
    double inv_mass; Vec inv_inertia;   /* body-frame diagonal */
} Body;

typedef struct {
    Body b[NB];
    Quat target[NB];           /* joint target: child rotation relative to parent (b>0) */
    double total_mass, h, mu;  /* substep, friction coefficient */
    int substeps, iters;   /* joint solver iterations per substep */
    double t, kp_scale, kd_scale;   /* muscle strength multipliers (1 = BODY_DEFS) */
    Vec push_force; int push_body; double push_left;   /* an external push (N) on one body for push_left s */
    /* training aid: a compliant constraint holding the pelvis to (assist_x, assist_q) with strength assist in
     * 0..1 (stiffness assist*ASSIST_K, assist*ASSIST_KR), damped toward the velocity assist_v */
    double assist; Vec assist_x, assist_v; Quat assist_q;
} World;

extern const BodyDef BODY_DEFS[NB];

/* quaternion / vector helpers */
static inline Vec v3(double x, double y, double z) { Vec r = {x, y, z}; return r; }
static inline Vec vadd(Vec a, Vec b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec vsub(Vec a, Vec b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec vscale(Vec a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline double vdot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec vcross(Vec a, Vec b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline double vlen(Vec a) { return sqrt(vdot(a, a)); }
static inline Quat qmul(Quat a, Quat b) {
    Quat r = {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
    return r;
}
static inline Quat qconj(Quat a) { Quat r = {a.w, -a.x, -a.y, -a.z}; return r; }
static inline Quat qnorm(Quat a) { double n = sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z); Quat r = {a.w / n, a.x / n, a.y / n, a.z / n}; return r; }
static inline Vec qrot(Quat q, Vec v) {   /* q v q* */
    Vec u = v3(q.x, q.y, q.z), t = vscale(vcross(u, v), 2);
    return vadd(vadd(v, vscale(t, q.w)), vcross(u, t));
}
static inline Quat qexp(Vec r) {          /* rotation vector -> quaternion */
    double a = vlen(r); if (a < 1e-12) { Quat i = {1, r.x * .5, r.y * .5, r.z * .5}; return qnorm(i); }
    double s = sin(a * .5) / a; Quat q = {cos(a * .5), r.x * s, r.y * s, r.z * s}; return q;
}
static inline Vec qlog(Quat q) {          /* quaternion -> rotation vector (shortest) */
    if (q.w < 0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
    double s = sqrt(q.x * q.x + q.y * q.y + q.z * q.z); if (s < 1e-12) return v3(2 * q.x, 2 * q.y, 2 * q.z);
    double a = 2 * atan2(s, q.w); return vscale(v3(q.x, q.y, q.z), a / s);
}
static const Quat QID = {1, 0, 0, 0};

void world_init(World *W, double total_mass, int substeps, double dt);
/* Place the bodies by forward kinematics from a root position/rotation and per-joint rotations (child
 * relative to parent), with the given root linear velocity; angular velocities zero. */
void world_pose(World *W, Vec root_x, Quat root_q, const Quat *rel, Vec root_v);
/* Advance dt = substeps * h, holding the current joint targets. */
void world_step(World *W);
/* World position of a body's local point (relative to its centre, rest frame). */
static inline Vec body_point(const Body *b, Vec r) { return vadd(b->x, qrot(b->q, r)); }
Vec world_com(const World *W, Vec *vel);
double world_lowest(const World *W, int body);   /* lowest corner height of a body */

#endif
