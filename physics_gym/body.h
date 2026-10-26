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

/* The same source also builds for the GPU (gpu/tssenv_cuda.cu, one thread per world): there BHD marks device
 * functions and BCONST device constants. For the CPU both are empty. */
#ifdef __CUDACC__
#define BHD __device__
#define BCONST __device__ const
#else
#define BHD
#define BCONST const
#endif
#include "bmath.h"   /* sin, cos, exp, log, atan2 with the same bits on every build (and in body.js) */

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
    double tone[NB];                /* per-joint muscle tone (multiplies kp and kd of the joint above each body) */
    Vec push_force; int push_body; double push_left;   /* an external push (N) on one body for push_left s */
    /* training aid: a compliant constraint holding the pelvis to (assist_x, assist_q) with strength assist in
     * 0..1 (stiffness assist*ASSIST_K, assist*ASSIST_KR), damped toward the velocity assist_v */
    double assist; Vec assist_x, assist_v; Quat assist_q;
    double touch[NB];   /* contact force on each body, mean over the last world_step (N): floor and room contacts */
    double limit_k;     /* joint ligaments (physics_gym): stiffness (N m / rad) holding each joint inside its range from
                         * the motion capture (joint_limits.h); 0 = no limits, as in body/ */
    double strength;    /* muscles (physics_gym): 1 = each joint's drive torque held within its strength and allowed to rise
                         * only at its slew rate (muscle.h, from the capture and published maximum torques); 0 = unlimited,
                         * as in body/ */
    double mtau[NB];    /* each joint's drive torque in the last substep (N m): where its slew limit starts from */
    double act[NB];     /* each joint's activation, mean over the last world_step: drive torque / strength (0..1) */
    double metab;       /* muscle cost of the last world_step: the sum over joints of activation squared, mean over substeps */
    /* grip (physics_gym): each hand (the far end of each forearm: 0 left, 1 right) closes on a handhold or a ledge's top edge
     * within GRIP_REACH while its command is on (> GRIP_ON; it lets go at <= 0), and is then held to that point by a constraint
     * of at most grip_max (N): past that the hand slides, and beyond 2 GRIP_REACH the grip is lost. 0 = no grip. */
    double grip_max, grip_cmd[2];
    int grip_box[2];    /* room box held, -1 none */
    Vec grip_at[2];     /* the point held (world; room boxes do not move) */
    double grip_force[2];   /* the grip's pull, mean over the last world_step (N) */
    int room;           /* which room of the bank this world is in (0 with a single room) */
} World;

#ifndef __CUDACC__   /* the GPU build defines it ahead of every use, in one unit */
extern const BodyDef BODY_DEFS[NB];
#endif

/* quaternion / vector helpers */
static inline BHD Vec v3(double x, double y, double z) { Vec r = {x, y, z}; return r; }
static inline BHD Vec vadd(Vec a, Vec b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline BHD Vec vsub(Vec a, Vec b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline BHD Vec vscale(Vec a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline BHD double vdot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline BHD Vec vcross(Vec a, Vec b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline BHD double vlen(Vec a) { return sqrt(vdot(a, a)); }
static inline BHD Quat qmul(Quat a, Quat b) {
    Quat r = {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
    return r;
}
static inline BHD Quat qconj(Quat a) { Quat r = {a.w, -a.x, -a.y, -a.z}; return r; }
static inline BHD Quat qnorm(Quat a) { double n = sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z); Quat r = {a.w / n, a.x / n, a.y / n, a.z / n}; return r; }
static inline BHD Vec qrot(Quat q, Vec v) {   /* q v q* */
    Vec u = v3(q.x, q.y, q.z), t = vscale(vcross(u, v), 2);
    return vadd(vadd(v, vscale(t, q.w)), vcross(u, t));
}
static inline BHD Quat qexp(Vec r) {          /* rotation vector -> quaternion */
    double a = vlen(r); if (a < 1e-12) { Quat i = {1, r.x * .5, r.y * .5, r.z * .5}; return qnorm(i); }
    double s = b_sin(a * .5) / a; Quat q = {b_cos(a * .5), r.x * s, r.y * s, r.z * s}; return q;
}
static inline BHD Vec qlog(Quat q) {          /* quaternion -> rotation vector (shortest) */
    if (q.w < 0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
    double s = sqrt(q.x * q.x + q.y * q.y + q.z * q.z); if (s < 1e-12) return v3(2 * q.x, 2 * q.y, 2 * q.z);
    double a = 2 * b_atan2(s, q.w); return vscale(v3(q.x, q.y, q.z), a / s);
}
static BCONST Quat QID = {1, 0, 0, 0};

BHD void world_init(World *W, double total_mass, int substeps, double dt);
/* Place the bodies by forward kinematics from a root position/rotation and per-joint rotations (child
 * relative to parent), with the given root linear velocity; angular velocities zero. */
BHD void world_pose(World *W, Vec root_x, Quat root_q, const Quat *rel, Vec root_v);
/* Advance dt = substeps * h, holding the current joint targets. */
BHD void world_step(World *W);
/* world_step for n <= 8 worlds at once, one per vector lane (body_x8.c); they must share substeps, iters, h */
void world_step_x8(World **W, int n);
/* World position of a body's local point (relative to its centre, rest frame). */
static inline BHD Vec body_point(const Body *b, Vec r) { return vadd(b->x, qrot(b->q, r)); }
BHD Vec world_com(const World *W, Vec *vel);
BHD double world_lowest(const World *W, int body);   /* lowest corner height of a body */

/* The room (physics_gym, room.json): static oriented boxes the bodies' corners collide with, besides the floor.
 * c centre, h half extents, q rotation, mu friction, soft contact compliance (m/N, 0 rigid), rad = |h| (broad
 * phase). One room for all worlds; empty by default, which leaves the solver exactly as body/ had it. */
typedef struct { Vec c, h; Quat q; double mu, soft, rad; int grip; } RoomBox;
#define ROOM_MAX 256
#define ROOM_BANK 64        /* rooms held at once (procedural variants): each world collides with its own, World.room */
#define ROOM_REC 13         /* doubles per box as room_set takes them: c (3), h (3), q (w x y z), mu, soft, grip */
enum { GRIP_NONE, GRIP_LEDGE, GRIP_HOLD };   /* a box's grip label: nothing; its top edges (if its top faces up); all of it */
#define GRIP_REACH 0.06     /* m: how close a hand must be to close on a hold */
#define GRIP_ON 0.2         /* grip command that closes a hand */
#define GRIP_MAX 450.0      /* N per hand: about her weight (45 kg); one hand just holds her hanging */
#define ROOM_CONTACTS (NB * NCORNER * 2)
/* Terrain queries (every build): the vertical line through (x, z) against the room.
 * support_y: height of the highest surface under the line at or below ytop (the floor, y = 0, if none): what she
 *   stands on there; boxes reaching above ytop there (a tunnel roof overhead) are not support.
 * column_hit: whether any box occupies the line between heights y0 and y1. */
BHD double support_y(int room, double x, double z, double ytop);
BHD int column_hit(int room, double x, double z, double y0, double y1);
/* the nav grid, for env.c's path helper: NAV_N x NAV_N cells of NAV_CELL m over x, z in +-NAV_HALF, each the height of
 * the surface she would stand on there (support_y below 2 m), or NAV_BLOCK where something stands in the way above it
 * (to 1.6 m) or the cell is next to one of those or to a rise of more than NAV_STEP (paths keep a cell off walls). Built
 * once per room after its boxes are set (room_set_bank on the CPU, tssenv_kernels.cu on the device: nav_pass 0, 1, 2
 * over every cell, each pass reading only what the one before wrote, so cells can go in any order or all at once). */
#define NAV_N 64
#define NAV_CELL 0.25
#define NAV_HALF 8.0
#define NAV_STEP 0.3
#define NAV_BLOCK 1000.0f
BHD float nav_h(int room, int cell);
BHD void nav_pass(int room, int pass, int cell);
BHD double ray_dist(int room, Vec o, Vec d, double maxd);   /* distance along unit d from o to the first box (at most maxd) */
BHD int room_banks(void);                 /* rooms in the bank (0: none, the open floor) */
BHD int room_size(int room);
BHD int grip_find(int room, Vec p, double reach, Vec *at);   /* the nearest holdable point within reach of p: its box, or -1 */
BHD Vec hand_point(const World *W, int hand);       /* the hand (0 left, 1 right): the far end of the forearm, world */
BHD const RoomBox *room_at(int room, int j);
#ifndef __CUDACC__
int room_set(const double *boxes, int n);    /* n boxes of ROOM_REC doubles, as the only room; returns n or -1 */
int room_set_bank(int room, const double *boxes, int n);   /* room `room` of the bank (rooms 0..room-1 set first) */
int room_count(void);
int room_load(const char *path);             /* room.json (room.c); returns the box count or -1 */
int room_load_bank(const char *path);        /* a room bank, or one room.json, as rooms 0, 1, ... (room.c); returns the room count or -1 */
#endif

#endif
