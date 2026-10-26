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
static BHD int nav_on(const Env *E);
static BHD void matched_start(Env *E);
static BHD Vec nav_waypoint(const Env *E, double *len);
static BHD Vec nav_ahead(const Env *E, int steps, double *len);
static BHD Vec hinge_axis(int h) { return h == 1 ? v3(1, 0, 0) : h == 2 ? v3(0, 1, 0) : v3(0, 0, 1); }

BHD double env_phase(const Env *E) { return E->cfg.task == TASK_IMITATE || (E->cfg.task == TASK_GOTO && E->cfg.clip) ? E->t0 + E->W.t : 0; }

/* Reference at clip time t: joint rotations, body world rotations and root, the root carried across loops. */
/* TASK_IMITATE over a clip set (cfg.style): see env.h */
static BHD int clip_set(const Env *E) { return E->cfg.task == TASK_IMITATE && E->cfg.nstyle > 0; }
/* a clip set played in a room is a course (tools/jump_course.mjs): room r is built round clip r % nstyle in that clip's
 * own frame, platforms where its feet come down and a gap where they are off the ground; dropping below the platform
 * tops ends the episode (she starts over) */
static BHD int course(const Env *E) { return clip_set(E) && room_size(E->W.room) > 0; }
/* over a box's top (horizontally inside it; `back` m short of its near end counts too): how far onto it from that
 * near end along its length (m, negative short of it), or 0 when not over it */
static BHD double over_box(const RoomBox *o, Vec p, double back) {
    const Vec l = qrot(qconj(o->q), vsub(p, o->c));
    return fabs(l.x) < o->h.x && l.z > -o->h.z - back && l.z < o->h.z ? l.z + o->h.z + 1e-9 : 0;
}
static BHD void ref_at(const Env *E, double t, Quat *rel, Quat *world, Vec *root) {
    const Clip *c = E->rc; const double T = (c->frames - 1) / c->fps;
    if (clip_set(E)) {   /* once through, then its last frame held (standing, at the end of a get-up) */
        mref_sample(c, fmin(fmax(t, 0), T * (1 - 1e-9)), root, world, rel); *root = vadd(*root, E->clip_shift); return;
    }
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

/* TASK_BALANCE with goal_tops: standing on the top of a room box (a ramp, a landing, a platform, a stone: tilted at most
 * about 25 degrees, at most 1 m up, clear to head height), any heading, each body lifted so the lowest point of all
 * sits on the surface under it. The pose stays as drawn when no such spot turns up. */
static BHD void top_start(Env *E) {
    const int rm = E->W.room;
    for (int tries = 0; tries < 32; tries++) {
        const RoomBox *o = room_at(rm, (int)(env_rand(&E->rng) * room_size(rm)) % room_size(rm));
        const double u = 2 * env_rand(&E->rng) - 1, w = 2 * env_rand(&E->rng) - 1, yaw = env_rand(&E->rng) * 2 * M_PI;
        if (qrot(o->q, v3(0, 1, 0)).y < 0.9) continue;
        const Vec top = vadd(o->c, qrot(o->q, v3(0.6 * u * o->h.x, o->h.y, 0.6 * w * o->h.z)));
        const double y = support_y(rm, top.x, top.z, top.y + 0.01);
        if (y < 0.05 || y > 1.0 || column_hit(rm, top.x, top.z, y + 0.3, y + 1.9)) continue;
        const Quat qy = qexp(v3(0, yaw, 0)); const Vec p0 = E->W.b[PELVIS].x;
        for (int i = 0; i < NB; i++) { Body *b = E->W.b + i; const Vec d = qrot(qy, vsub(b->x, p0)); b->x = v3(top.x + d.x, p0.y + d.y, top.z + d.z); b->q = qmul(qy, b->q); }
        double lift = -1e9;
        for (int i = 0; i < NB; i++) { const Vec c = E->W.b[i].x; lift = fmax(lift, support_y(rm, c.x, c.z, y + 1.9) - world_lowest(&E->W, i)); }
        for (int i = 0; i < NB; i++) { E->W.b[i].x.y += lift; E->W.b[i].xprev = E->W.b[i].x; E->W.b[i].qprev = E->W.b[i].q; }
        return;
    }
}
BHD void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed) {
    E->cfg = *cfg; E->rng = seed * 0x9E3779B97F4A7C15ULL + 1; E->step = 0; E->done = 0; E->t0 = 0;
    world_init(&E->W, 45, 20, 1.0 / CTRL_HZ);
    if (room_banks() > 1) E->W.room = (int)(((seed * 0xD1B54A32D192ED03ULL) >> 33) % (uint64_t)room_banks());   /* her room of the bank, from the seed (no draw from rng: one room plays as before) */
    if (cfg->arm_tone > 0) for (int i = L_UARM; i <= R_FARM; i++) E->W.tone[i] = cfg->arm_tone;
    for (int k = 0; k < ACT_DIM; k++) E->prev_act[k] = E->filt[k] = 0;
    E->head_v = v3(0, 0, 0); E->style_has_prev = 0; E->path_len = 0; E->jumps = 0;
    for (int k = 0; k < 2; k++) { E->foot_on[k] = 1; E->foot_vy[k] = 0; }
    E->next_push = 0.5 + env_rand(&E->rng) * 2 * cfg->push_every;
    E->rc = cfg->clip;
    if (cfg->task == TASK_IMITATE && cfg->nstyle > 0 && cfg->style_start > 0 && env_rand(&E->rng) < cfg->style_start) { matched_start(E); return; }
    if (cfg->task == TASK_IMITATE) {
        if (cfg->nstyle > 0) E->rc = cfg->style[(int)(env_rand(&E->rng) * cfg->nstyle) % cfg->nstyle];   /* a clip of the set */
        if (course(E)) E->rc = cfg->style[E->W.room % cfg->nstyle];   /* the clip its room was built round */
        const Clip *c = E->rc; const double T = (c->frames - 1) / c->fps;
        E->t0 = cfg->fixed_start ? 0 : env_rand(&E->rng) * T * (cfg->nstyle > 0 ? 0.85 : 1);
        if (course(E) && !cfg->fixed_start) E->t0 *= 0.7 / 0.85;   /* a course: mostly from the run-up, so the takeoff gets practised (not only the landing) */
        E->clip_shift = v3(0, 0, 0);
        Quat rel[NB], rel2[NB], wq[NB]; Vec root, root2; const double dt = 1.0 / CTRL_HZ;
        ref_at(E, E->t0, rel, wq, &root); ref_at(E, E->t0 + dt, rel2, wq, &root2);
        E->clip_shift = v3(-root.x, 0, -root.z);   /* the reference starts above the origin */
        if (course(E)) E->clip_shift = v3(-c->root[0].x, support_y(E->W.room, 0, 0, 5), -c->root[0].z);   /* the clip's frame 0 over the origin, on the platform */
        root = vadd(root, E->clip_shift); root2 = vadd(root2, E->clip_shift);
        /* body velocities by finite difference of the reference pose */
        World W2; world_init(&W2, 45, 20, dt);
        world_pose(&W2, root2, rel2[0], rel2, v3(0, 0, 0));
        world_pose(&E->W, root, rel[0], rel, v3(0, 0, 0));
        for (int i = 0; i < NB; i++) {
            Body *b = E->W.b + i; b->v = vscale(vsub(W2.b[i].x, b->x), 1 / dt);
            Quat dq = qmul(W2.b[i].q, qconj(b->q)); b->w = vscale(qlog(dq), 1 / dt);
        }
        /* keep the feet out of the ground (in a course, out of whatever is under each body) */
        double lo = 1e9; for (int i = 0; i < NB; i++) { const Body *b = E->W.b + i; double y = world_lowest(&E->W, i) - (course(E) ? support_y(E->W.room, b->x.x, b->x.z, b->x.y + 0.5) : 0); if (y < lo) lo = y; }
        if (lo < 0) for (int i = 0; i < NB; i++) E->W.b[i].x.y -= lo;
        for (int i = 0; i < NB; i++) E->W.b[i].xprev = E->W.b[i].x;
        ref_at(E, E->t0, E->ref, E->ref_world, &E->ref_root);
        for (int i = 0; i < NB; i++) E->W.target[i] = E->ref[i];
        if (course(E)) E->landed = !over_box(room_at(E->W.room, 0), E->W.b[PELVIS].x, 0);   /* a landing to come only when she starts on the takeoff platform */
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
    if (cfg->task == TASK_BALANCE && cfg->goal_tops > 0 && room_size(E->W.room) && env_rand(&E->rng) < cfg->goal_tops) top_start(E);
}

/* heading frame: rotation removing the pelvis yaw */
static BHD Quat heading_q(Quat q) { const Vec f = qrot(q, v3(0, 0, 1)); return qexp(v3(0, b_atan2(f.x, f.z), 0)); }   /* the yaw of q alone */
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
static BCONST double PROBE_DIST[6] = {0.3, 0.6, 0.9, 1.2, 0.6, 0.6}, PROBE_ANG[6] = {0, 0, 0, 0, 0.5, -0.5};   /* the ground probes (m, rad) */
BHD void env_body_feats(const World *W, int floor, float *o) {
    const Quat hi = heading_inv(W, floor); const Vec root = W->b[PELVIS].x; int k = 0;
    for (int i = 0; i < NB; i++) {
        const Body *b = W->b + i; Vec p = qrot(hi, vsub(b->x, root));
        if (i == 0) p.y = root.y - fmin(world_lowest(W, L_FOOT), world_lowest(W, R_FOOT));   /* the pelvis: its height above her lowest foot corner */
        Quat q = qmul(hi, b->q); Vec ux = qrot(q, v3(1, 0, 0)), uy = qrot(q, v3(0, 1, 0));
        Vec v = qrot(hi, b->v), w = qrot(hi, b->w);
        o[k++] = p.x; o[k++] = p.y; o[k++] = p.z;
        o[k++] = ux.x; o[k++] = ux.y; o[k++] = ux.z; o[k++] = uy.x; o[k++] = uy.y; o[k++] = uy.z;
        o[k++] = v.x; o[k++] = v.y; o[k++] = v.z; o[k++] = w.x * .2; o[k++] = w.y * .2; o[k++] = w.z * .2;
    }
}
BHD void env_obs(const Env *E, float *o) {
    const World *W = &E->W; Quat hi = heading_inv(W, floor_task(E) || clip_set(E)); Vec root = W->b[PELVIS].x; int k = AMP_FEAT;
    env_body_feats(W, floor_task(E) || clip_set(E), o);
    const double wt = W->total_mass * 9.81;   /* foot contact from touch: more than 2% of her weight on the foot */
    o[k++] = W->touch[L_FOOT] > 0.02 * wt; o[k++] = W->touch[R_FOOT] > 0.02 * wt;
    Vec cv, c = world_com(W, &cv); Vec d = qrot(hi, vsub(c, root));
    o[k++] = d.x; o[k++] = d.z;
    double ph = 0;
    if ((E->cfg.task == TASK_IMITATE && !clip_set(E)) || (E->cfg.task == TASK_GOTO && E->cfg.clip)) { const Clip *cl = E->cfg.clip; ph = 2 * M_PI * env_phase(E) / ((cl->frames - 1) / cl->fps); }
    o[k++] = b_sin(ph); o[k++] = b_cos(ph);
    if (E->cfg.task == TASK_GOTO) {   /* the goal from her pelvis, heading frame; far goals at 4 m in their direction */
        const Vec g = qrot(hi, vsub(E->goal, root)); const double r = b_sqrt_h(g.x, g.z), s = r > 4 ? 4 / r : 1;
        o[k++] = g.x * s; o[k++] = g.y; o[k++] = g.z * s; o[k++] = fmin(r, 4);
    } else for (int j = 0; j < OBS_GOAL; j++) o[k++] = 0;
    for (int i = 0; i < NB; i++) o[k++] = fmin(3, W->touch[i] / wt);   /* touch */
    for (int i = 1; i < NB; i++) o[k++] = W->act[i];   /* muscle sense */
    for (int h = 0; h < 2; h++) {   /* grip: what each hand holds, and the nearest thing it could */
        const Vec p = hand_point(W, h); Vec at; const int j = grip_find(W->room, p, 1.0, &at);
        o[k++] = W->grip_box[h] >= 0;
        if (j >= 0) { const Vec d = qrot(hi, vsub(at, p)); o[k++] = d.x; o[k++] = d.y; o[k++] = d.z; o[k++] = 1; }
        else { o[k++] = 0; o[k++] = 0; o[k++] = 0; o[k++] = 0; }
    }
    {   /* probes: the ground ahead and what stands in the way (env.h OBS_PROBE) */
        const Quat h = qconj(hi); const double foot = fmin(world_lowest(W, L_FOOT), world_lowest(W, R_FOOT));
        for (int i = 0; i < 6; i++) {
            const Vec d = qrot(h, v3(b_sin(PROBE_ANG[i]), 0, b_cos(PROBE_ANG[i])));
            const double x = root.x + PROBE_DIST[i] * d.x, z = root.z + PROBE_DIST[i] * d.z;
            o[k++] = fmax(-1, fmin(1, support_y(W->room, x, z, foot + 0.8) - foot));
        }
        for (int lv = 0; lv < 2; lv++) for (int i = 0; i < 5; i++) {
            const double a = -1.0 + 0.5 * i; const Vec d = qrot(h, v3(b_sin(a), 0, b_cos(a)));
            o[k++] = ray_dist(W->room, v3(root.x, foot + (lv ? 0.9 : 0.35), root.z), d, 1.5) / 1.5;
        }
    }
    if (E->cfg.task == TASK_GOTO) {   /* the path helper: the waypoint and the path to go (env.h OBS_PATH) */
        double len; const Vec g = qrot(hi, vsub(nav_waypoint(E, &len), root)); const double r = b_sqrt_h(g.x, g.z), s = r > 2 ? 2 / r : 1;
        o[k++] = g.x * s; o[k++] = g.y; o[k++] = g.z * s; o[k++] = fmin(len, 10);
    } else for (int j = 0; j < OBS_PATH; j++) o[k++] = 0;
    if (E->cfg.task == TASK_GOTO || course(E)) {   /* where to step (env.h OBS_LOOK): the path nearer and further on, the ground all round */
        if (E->cfg.task != TASK_GOTO) for (int j = 0; j < 6; j++) o[k++] = 0;   /* (a course has no path: its ground only) */
        else for (int a = 0; a < 2; a++) {
            double len; const Vec g = qrot(hi, vsub(nav_ahead(E, a ? 8 : 2, &len), root)); const double r = b_sqrt_h(g.x, g.z), s = r > 3 ? 3 / r : 1;
            o[k++] = g.x * s; o[k++] = g.y; o[k++] = g.z * s;
        }
        const Quat h = qconj(hi); const double foot = fmin(world_lowest(W, L_FOOT), world_lowest(W, R_FOOT));
        for (int ring = 0; ring < 2; ring++) for (int i = 0; i < 8; i++) {
            const double a = i * (M_PI / 4), rr = ring ? 0.8 : 0.4; const Vec d = qrot(h, v3(b_sin(a), 0, b_cos(a)));
            o[k++] = fmax(-1, fmin(1, support_y(W->room, root.x + rr * d.x, root.z + rr * d.z, foot + 0.8) - foot));
        }
    } else for (int j = 0; j < OBS_LOOK; j++) o[k++] = 0;
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
__device__ StyleClip style_clips[MAX_STYLE]; __device__ int style_n;   /* filled by tssenv_cuda.cu */
#else
static StyleClip style_clips[MAX_STYLE]; static int style_n;
#endif
static BHD void pose_features(const Quat *wq, float *f) {   /* joint rotations (child in parent), gravity in the pelvis frame */
    int k = 0;
    for (int i = 1; i < NB; i++) { Quat r = qmul(qconj(wq[BODY_DEFS[i].parent]), wq[i]); f[k++] = r.w; f[k++] = r.x; f[k++] = r.y; f[k++] = r.z; }
    Vec gl = qrot(qconj(wq[PELVIS]), v3(0, -1, 0)); f[k++] = gl.x; f[k++] = gl.y; f[k++] = gl.z;
}
#ifndef __CUDACC__
void env_style_prepare(const Clip *c) {
    for (int i = 0; i < style_n; i++) if (style_clips[i].c == c) return;
    if (style_n >= MAX_STYLE) return;
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
/* TASK_IMITATE over a clip set, a share (style_start) of its episodes: a real fall first (limp, shoved, settling where
 * she lands; or on all fours), then the clip frame nearest her pose (pose_features, all frames of the set) is where the
 * reference picks up. She is turned about the vertical to the frame's heading (the floor does not mind) and the clip
 * is shifted under her, so the tracking reward and the drift limit apply as from a clip's own frame; she learns to
 * join a get-up from wherever a fall left her (the viewer does the same matching when she falls). */
static BHD double match_dist(const float *a, const float *b, double ha, double hb) {
    double d = feat_dist(a, b);
    for (int k = 4 * (NB - 1); k < STYLE_FEAT; k++) d += 3 * (a[k] - b[k]) * (a[k] - b[k]);   /* the tilt 4x in all */
    return d + 20 * (ha - hb) * (ha - hb);   /* 0.5 m of height: about as far as 1.2 rad over the joints */
}
static BHD double floor_yaw(Quat q) { const Vec f = vadd(qrot(q, v3(0, 0, 1)), qrot(q, v3(0, 1, 0))); return b_atan2(f.x, f.z); }
/* Where a fall leaves her: on all fours with noise and settled, or limp from standing, shoved, and settled where she lands. */
static BHD void limp_fall(World *W, uint64_t *rng, int steps) {
    Quat rel[NB], rq; const int fall = env_rand(rng) < 0.6;
    crawl_pose(rel, &rq); if (fall) stand_pose(rel), rq = QID;
    for (int i = 1; i < NB; i++) {
        const double nz = env_randn(rng), ny = env_randn(rng), nx = env_randn(rng); Vec n = v3(nx, ny, nz);
        if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
        rel[i] = qmul(rel[i], qexp(vscale(n, 0.15)));
    }
    rq = qmul(qexp(v3(0, env_rand(rng) * 2 * M_PI, 0)), rq);
    world_pose(W, v3(0, 1, 0), rq, rel, v3(0, 0, 0)); for (int i = 0; i < NB; i++) W->target[i] = rel[i];
    settle_to_ground(W);
    if (fall) {
        const double a = env_rand(rng) * 2 * M_PI, sp = 1 + 2 * env_rand(rng);
        for (int i = 0; i < NB; i++) W->b[i].v = vscale(v3(b_cos(a), 0, b_sin(a)), sp * (W->b[i].x.y / 1.2));
        W->kp_scale = 0.03; for (int s = 0; s < steps; s++) world_step(W); W->kp_scale = 1;
    } else for (int s = 0; s < 10; s++) world_step(W);
}
/* The falls for the matched starts, simulated once (env_fall_bank, tss_fall_bank) rather than in each reset: a reset runs
 * inside the batch's step, and on the GPU one environment's fall held up the whole step. Pelvis over the origin. */
#ifdef __CUDACC__
__device__ FallState fall_bank[FALL_N]; __device__ int fall_n;   /* filled by tssenv_kernels.cu */
#else
static FallState fall_bank[FALL_N]; static int fall_n;
#endif
BHD void fall_make(int i) {
    World W; world_init(&W, 45, 20, 1.0 / CTRL_HZ); uint64_t rng = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ULL + 1;
    limp_fall(&W, &rng, 75);
    FallState *f = fall_bank + i; const Vec p0 = W.b[PELVIS].x;
    for (int k = 0; k < NB; k++) { f->x[k] = v3(W.b[k].x.x - p0.x, W.b[k].x.y, W.b[k].x.z - p0.z); f->q[k] = W.b[k].q; f->v[k] = W.b[k].v; f->w[k] = W.b[k].w; }
}
#ifndef __CUDACC__
void env_fall_bank(void) {
    if (fall_n) return;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < FALL_N; i++) fall_make(i);
    fall_n = FALL_N;
}
#endif
#ifdef __CUDACC__
__device__ FallState start_bank[START_N]; __device__ int start_n;   /* filled by tssenv_kernels.cu tss_set_starts */
#else
static FallState start_bank[START_N]; static int start_n;
void env_set_starts(const FallState *s, int n) { if (n > START_N) n = START_N; memcpy(start_bank, s, sizeof(FallState) * n); start_n = n; }
#endif
static BHD void matched_start(Env *E) {
    World *W = &E->W;
    if (fall_n) {   /* a fall of the bank (her heading is set below, from the frame matched) */
        const FallState *f = fall_bank + (int)(env_rand(&E->rng) * fall_n) % fall_n;
        for (int i = 0; i < NB; i++) { Body *b = W->b + i; b->x = b->xprev = f->x[i]; b->q = b->qprev = f->q[i]; b->v = f->v[i]; b->w = f->w[i]; }
    } else limp_fall(W, &E->rng, 40);
    /* the nearest frame of the set: joint angles, and (weighted up) the pelvis's tilt and height, since a limp body's
     * straight legs and loose arms are also a standing pose's */
    Quat wq[NB]; for (int i = 0; i < NB; i++) wq[i] = W->b[i].q;
    float cur[STYLE_FEAT]; pose_features(wq, cur);
    const double hp = W->b[PELVIS].x.y;
    const Clip *bc = E->cfg.style[0]; int bt = 0; double best = 1e30;
    for (int k = 0; k < E->cfg.nstyle; k++) for (int i = 0; i < style_n; i++) if (style_clips[i].c == E->cfg.style[k]) {
        const StyleClip *sc = style_clips + i;
        for (int t = 0; t < sc->n - 1; t++) { const double d = match_dist(cur, sc->f + (size_t)t * STYLE_FEAT, hp, sc->c->root[t].y); if (d < best) { best = d; bc = sc->c; bt = t; } }
    }
    E->rc = bc; E->t0 = bt / bc->fps;
    /* her turned to the frame's heading, about her pelvis; the clip shifted under her */
    Vec root; Quat rw[NB]; mref_sample(bc, E->t0, &root, rw, NULL);
    const Quat turn = qexp(v3(0, floor_yaw(rw[PELVIS]) - floor_yaw(W->b[PELVIS].q), 0)); const Vec p0 = W->b[PELVIS].x;
    for (int i = 0; i < NB; i++) {
        Body *b = W->b + i; b->x = vadd(p0, qrot(turn, vsub(b->x, p0))); b->q = qnorm(qmul(turn, b->q)); b->v = qrot(turn, b->v); b->w = qrot(turn, b->w);
        b->xprev = b->x; b->qprev = b->q;
    }
    E->clip_shift = v3(p0.x - root.x, 0, p0.z - root.z); W->t = 0;
    ref_at(E, E->t0, E->ref, E->ref_world, &E->ref_root);
    for (int i = 0; i < NB; i++) W->target[i] = E->ref[i];
}
/* where her feet point (a landing can leave the pelvis twisted): the viewer aims the next jump by it */
static BHD double feet_yaw(const Quat *q) { const Vec f = vadd(qrot(q[L_FOOT], v3(0, 0, 1)), qrot(q[R_FOOT], v3(0, 0, 1))); return b_atan2(f.x, f.z); }
/* a jump course, chained (TaskCfg.chain): her carried back to the clip's start on the takeoff platform, turned so her feet point
 * along the clip's start give or take chain_yaw, the clip from its start */
static BHD void chain_next(Env *E) {
    World *W = &E->W; Quat wq[NB], hq[NB]; Vec root; mref_sample(E->rc, 0, &root, wq, NULL);
    for (int i = 0; i < NB; i++) hq[i] = W->b[i].q;
    const double err = (2 * env_rand(&E->rng) - 1) * E->cfg.chain_yaw;
    const Quat turn = qexp(v3(0, feet_yaw(wq) - feet_yaw(hq) + err, 0));
    const Vec p0 = W->b[PELVIS].x, to = vadd(root, E->clip_shift);
    for (int i = 0; i < NB; i++) {
        Body *b = W->b + i; const Vec d = qrot(turn, vsub(b->x, p0));
        b->x = b->xprev = v3(to.x + d.x, b->x.y, to.z + d.z); b->q = b->qprev = qnorm(qmul(turn, b->q)); b->v = qrot(turn, b->v); b->w = qrot(turn, b->w);
    }
    W->grip_box[0] = W->grip_box[1] = -1; E->head_v = W->b[HEAD].v;
    E->t0 = -W->t; E->landed = !over_box(room_at(W->room, 0), W->b[PELVIS].x, 0); E->jumps++;
    ref_at(E, 0, E->ref, E->ref_world, &E->ref_root);
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
    const Vec p = W->b[PELVIS].x; const double sp = support_y(W->room, p.x, p.z, p.y);
    if (p.y - sp < 0.55 || W->b[HEAD].x.y - sp < 1.0) return 1;
    for (int i = 0; i < NB; i++) if (i != L_FOOT && i != R_FOOT && i != L_SHIN && i != R_SHIN) {
        const Vec c = W->b[i].x; if (world_lowest(W, i) - support_y(W->room, c.x, c.z, c.y) < 0.05) return 1; }
    return 0;
}

/* ---------- TASK_GOTO (physics_gym) ---------- */
/* half size of the area starts and goals are drawn from: the room's outer walls (its farthest box centres) less 0.6 m */
static BHD double arena(const Env *E) {
    const int r = E->W.room; if (!room_size(r)) return 3.0;
    double e = 0;
    for (int j = 0; j < room_size(r); j++) { const RoomBox *o = room_at(r, j); e = fmax(e, fmax(fabs(o->c.x), fabs(o->c.z))); }
    return e - 0.6;
}
/* room to stand at (x, z): open floor there and around, nothing from the floor up to head height */
static BHD int free_spot(int r, double x, double z) {
    for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) if (column_hit(r, x + 0.45 * a, z + 0.45 * b, 0.005, 1.9)) return 0;
    return 1;
}
/* ---------- the path helper: a breadth-first search over the room's nav grid (body.h) from the goal's cell ---------- */
static BHD int nav_on(const Env *E) { return room_size(E->W.room) > 0; }
static BHD int nav_cell(double x, double z) {
    int ix = (int)floor((x + NAV_HALF) / NAV_CELL), iz = (int)floor((z + NAV_HALF) / NAV_CELL);
    ix = ix < 0 ? 0 : ix >= NAV_N ? NAV_N - 1 : ix; iz = iz < 0 ? 0 : iz >= NAV_N ? NAV_N - 1 : iz;
    return iz * NAV_N + ix;
}
/* a step from cell a to its neighbour b: b clear, the height change at most NAV_STEP */
static BHD int nav_link(int r, int a, int b) { const float hb = nav_h(r, b); return hb < NAV_BLOCK && fabsf(hb - nav_h(r, a)) <= (float)NAV_STEP; }
static BCONST int NAV_DX[8] = {1, -1, 0, 0, 1, 1, -1, -1}, NAV_DZ[8] = {0, 0, 1, -1, 1, -1, 1, -1};
static BHD void nav_field(Env *E) {
    const int r = E->W.room; unsigned char *d = E->nav_d; unsigned short *q = E->nav_q;
    for (int c = 0; c < NAV_N * NAV_N; c++) d[c] = 255;
    if (!nav_on(E)) return;
    const int g = nav_cell(E->goal.x, E->goal.z); int head = 0, tail = 0; d[g] = 0; q[tail++] = (unsigned short)g;
    while (head < tail) {
        const int c = q[head++], cx = c % NAV_N, cz = c / NAV_N; if (d[c] >= 254) continue;
        for (int k = 0; k < 8; k++) {
            const int nx = cx + NAV_DX[k], nz = cz + NAV_DZ[k]; if (nx < 0 || nz < 0 || nx >= NAV_N || nz >= NAV_N) continue;
            const int n = nz * NAV_N + nx; if (d[n] != 255 || !nav_link(r, c, n)) continue;
            if (k >= 4 && (!nav_link(r, c, cz * NAV_N + nx) || !nav_link(r, c, nz * NAV_N + cx))) continue;   /* no cutting corners */
            d[n] = (unsigned char)(d[c] + 1); q[tail++] = (unsigned short)n;
        }
    }
}
/* the cell she is in, or (off the grid's clear cells: against a wall, on a box's edge) the nearest reached one within
 * 3 cells; -1 if none */
static BHD int nav_from(const Env *E, Vec p) {
    const int c = nav_cell(p.x, p.z); if (E->nav_d[c] != 255) return c;
    const int cx = c % NAV_N, cz = c / NAV_N; int best = -1, bd = 1 << 30;
    for (int dz = -3; dz <= 3; dz++) for (int dx = -3; dx <= 3; dx++) {
        const int nx = cx + dx, nz = cz + dz; if (nx < 0 || nz < 0 || nx >= NAV_N || nz >= NAV_N) continue;
        const int n = nz * NAV_N + nx; if (E->nav_d[n] == 255) continue;
        const int score = 16 * (dx * dx + dz * dz) + E->nav_d[n]; if (score < bd) { bd = score; best = n; }
    }
    return best;
}
/* about 1 m along the path: four steps down the search's distances from her cell (the goal itself when that close or
 * out of the grid's reach, and the straight line where there is no room); *len: the path still to go (m) */
static BHD Vec nav_ahead(const Env *E, int steps, double *len);
static BHD Vec nav_waypoint(const Env *E, double *len) { return nav_ahead(E, 4, len); }
static BHD Vec nav_ahead(const Env *E, int steps, double *len) {   /* (steps cells down the path: 4 is nav_waypoint) */
    const Vec p = E->W.b[PELVIS].x; Vec to = vsub(E->goal, p); to.y = 0; const double straight = vlen(to);
    *len = straight; if (!nav_on(E)) return E->goal;
    int w = nav_from(E, p); if (w < 0) return E->goal;
    const int r = E->W.room;
    for (int s = 0; s < steps && E->nav_d[w] > 0; s++) {
        const int cx = w % NAV_N, cz = w / NAV_N; int best = w;
        for (int k = 0; k < 8; k++) {
            const int nx = cx + NAV_DX[k], nz = cz + NAV_DZ[k]; if (nx < 0 || nz < 0 || nx >= NAV_N || nz >= NAV_N) continue;
            const int n = nz * NAV_N + nx; if (E->nav_d[n] < E->nav_d[best] && nav_link(r, w, n)) best = n;
        }
        if (best == w) break;
        w = best;
    }
    if (E->nav_d[w] == 0) { *len = straight; return E->goal; }
    const Vec at = v3(-NAV_HALF + (w % NAV_N + 0.5) * NAV_CELL, nav_h(r, w), -NAV_HALF + (w / NAV_N + 0.5) * NAV_CELL);
    Vec dw = vsub(at, p); dw.y = 0;
    *len = E->nav_d[w] * NAV_CELL + vlen(dw);
    return at;
}
/* a new goal 1 to 6 m away: on open floor, or (30%) on the top of a box that is flat enough and at most 0.8 m up */
static BHD void goto_goal(Env *E) {
    const double R = arena(E); const Vec p = E->W.b[PELVIS].x; const int rm = E->W.room;
    for (int tries = 0; tries < 64; tries++) {
        Vec g;
        if (room_size(rm) && env_rand(&E->rng) < (E->cfg.goal_tops > 0 ? E->cfg.goal_tops : 0.3)) {
            const RoomBox *o = room_at(rm, (int)(env_rand(&E->rng) * room_size(rm)) % room_size(rm));
            const double u = 2 * env_rand(&E->rng) - 1; const double w = 2 * env_rand(&E->rng) - 1;
            if (qrot(o->q, v3(0, 1, 0)).y < 0.9) continue;
            const Vec top = vadd(o->c, qrot(o->q, v3(0.7 * u * o->h.x, o->h.y, 0.7 * w * o->h.z)));
            const double y = support_y(rm, top.x, top.z, top.y + 0.01);
            if (y < 0.05 || y > 0.8 || column_hit(rm, top.x, top.z, y + 0.01, y + 1.9)) continue;
            g = v3(top.x, y, top.z);
        } else {
            const double x = (2 * env_rand(&E->rng) - 1) * R; const double z = (2 * env_rand(&E->rng) - 1) * R;
            if (column_hit(rm, x, z, 0.005, 1.9)) continue;
            g = v3(x, 0, z);
        }
        const double dx = g.x - p.x, dz = g.z - p.z, d2 = dx * dx + dz * dz;
        if (d2 < 1.0 || d2 > 36.0) continue;
        if (E->cfg.goal_cone > 0) {   /* within the cone ahead of her */
            const Vec f = qrot(E->W.b[PELVIS].q, v3(0, 0, 1)); const double fl = sqrt(f.x * f.x + f.z * f.z);
            if (fl < 1e-9 || (f.x * dx + f.z * dz) / (fl * sqrt(d2)) < b_cos(E->cfg.goal_cone)) continue;
        }
        E->goal = g; E->goal_d = sqrt(d2);
        if (nav_on(E)) {   /* a goal she can walk to (the path helper reaches her), else draw again */
            nav_field(E); if (nav_from(E, p) < 0 && tries < 60) continue;
            double len; nav_waypoint(E, &len); E->path_len = len;
        }
        return;
    }
    E->goal = v3(p.x, 0, p.z + 1.5); E->goal_d = 1.5;
    if (nav_on(E)) { nav_field(E); double len; nav_waypoint(E, &len); E->path_len = len; }
}
BHD void env_set_goal(Env *E, Vec goal) {
    const Vec p = E->W.b[PELVIS].x; E->goal = goal; E->goal_d = sqrt((goal.x - p.x) * (goal.x - p.x) + (goal.z - p.z) * (goal.z - p.z));
    if (nav_on(E)) { nav_field(E); double len; nav_waypoint(E, &len); E->path_len = len; }
}
/* standing on a free spot of floor, any heading, the stand pose with a little noise */
static BHD void goto_reset(Env *E) {
    const double R = arena(E); double x = 0, z = 0;
    for (int tries = 0; tries < 64; tries++) {
        const double a = (2 * env_rand(&E->rng) - 1) * R; const double b = (2 * env_rand(&E->rng) - 1) * R;
        if (free_spot(E->W.room, a, b)) { x = a; z = b; break; }
    }
    const double yaw = env_rand(&E->rng) * 2 * M_PI; const Quat qy = qexp(v3(0, yaw, 0));
    stand_pose(E->ref); E->t0 = 0;
    if (E->cfg.clip && env_rand(&E->rng) < 0.7) {   /* mid-stride: a random frame of the walk clip, its velocities by finite difference */
        const Clip *c = E->cfg.clip; const double dt = 1.0 / CTRL_HZ; E->t0 = env_rand(&E->rng) * (c->frames - 1) / c->fps;
        Quat r1[NB], r2[NB], w1[NB], w2[NB]; Vec root1, root2; mref_sample(c, E->t0, &root1, w1, r1); mref_sample(c, E->t0 + dt, &root2, w2, r2);
        const Vec at = v3(x, root1.y, z), at2 = vadd(at, qrot(qy, vsub(root2, root1)));
        r1[0] = qmul(qy, r1[0]); r2[0] = qmul(qy, r2[0]);
        World W2; world_init(&W2, 45, 20, dt); world_pose(&W2, at2, r2[0], r2, v3(0, 0, 0)); world_pose(&E->W, at, r1[0], r1, v3(0, 0, 0));
        for (int i = 0; i < NB; i++) {
            Body *b = E->W.b + i; b->v = vscale(vsub(W2.b[i].x, b->x), 1 / dt);
            const Quat dq = qmul(W2.b[i].q, qconj(b->q)); b->w = vscale(qlog(dq), 1 / dt);
        }
        for (int i = 0; i < NB; i++) E->W.target[i] = i ? r1[i] : QID;
    } else if (!E->cfg.clip && start_n && env_rand(&E->rng) < 0.8) {   /* no clip (env.h START_N): a moment of the motion capture */
        const FallState *s = start_bank + (int)(env_rand(&E->rng) * start_n) % start_n;
        for (int i = 0; i < NB; i++) {
            Body *b = E->W.b + i; b->x = vadd(v3(x, 0, z), qrot(qy, s->x[i])); b->q = b->qprev = qnorm(qmul(qy, s->q[i]));
            b->v = qrot(qy, s->v[i]); b->w = qrot(qy, s->w[i]);
        }
        for (int i = 1; i < NB; i++) E->W.target[i] = qmul(qconj(E->W.b[BODY_DEFS[i].parent].q), E->W.b[i].q);   /* holding the pose it is in */
    } else {   /* standing, the stand pose with a little noise */
        Quat rel[NB];
        for (int i = 0; i < NB; i++) {
            const double nz = env_randn(&E->rng), ny = env_randn(&E->rng), nx = env_randn(&E->rng); Vec n = v3(nx, ny, nz);
            if (BODY_DEFS[i].hinge) n = vscale(hinge_axis(BODY_DEFS[i].hinge), n.x);
            rel[i] = i ? qmul(E->ref[i], qexp(vscale(n, E->cfg.init_noise))) : QID;
            E->W.target[i] = E->ref[i];
        }
        world_pose(&E->W, v3(x, BODY_DEFS[PELVIS].center.y, z), qy, rel, v3(0, 0, 0));
    }
    double lo = 1e9; for (int i = 0; i < NB; i++) { const double y = world_lowest(&E->W, i); if (y < lo) lo = y; }
    for (int i = 0; i < NB; i++) { E->W.b[i].x.y -= lo; E->W.b[i].xprev = E->W.b[i].x; }
    E->goals = 0; goto_goal(E);
}
/* toward the goal near walking speed (still once there), facing it, chest up, calm; a bonus and a new goal on arrival */
static BHD double goto_reward(Env *E, double act2) {
    World *W = &E->W; const Body *pb = W->b + PELVIS;
    if (E->step >= E->cfg.steps) E->done = 1;
    if (fallen(W)) { E->done = 1; return 0; }
    const double sp = support_y(W->room, pb->x.x, pb->x.z, pb->x.y);
    Vec cv; world_com(W, &cv); const Vec vh = v3(cv.x, 0, cv.z);
    Vec to = vsub(E->goal, pb->x); to.y = 0; const double d = vlen(to);
    /* in a room, the way is the path helper's: toward its waypoint, progress along the path (round walls, up the ramp) */
    double len = d; Vec way = to; if (nav_on(E)) { way = vsub(nav_waypoint(E, &len), pb->x); way.y = 0; }
    const double wl = vlen(way); const Vec dir = wl > 1e-9 ? vscale(way, 1 / wl) : v3(0, 0, 0);
    const double move = d > 0.4 ? fmax(-1, fmin(1, vdot(vh, dir) / 0.9)) : b_exp(-2 * vdot(vh, vh));   /* toward it at up to walking speed; still once there */
    Vec f = qrot(pb->q, v3(0, 0, 1)); f.y = 0; const double fl = vlen(f);
    const double face = d > 0.4 && fl > 1e-9 ? fmax(0, vdot(f, dir) / fl) : 1;
    const double up = fmax(0, qrot(W->b[CHEST].q, v3(0, 1, 0)).y);
    /* distance gained this step (m): 0.03 at 0.9 m/s; paid up to 1.2 m/s (0.04), so there is nothing in rushing */
    const double prog = nav_on(E) ? fmax(-0.1, fmin(0.04, E->path_len - len)) : fmax(-0.1, fmin(0.04, E->goal_d - d)); E->goal_d = d; E->path_len = len;
    /* care at an edge: heading at more than 0.5 m/s for a drop of over NAV_STEP within half a metre costs the excess speed
     * squared (a person slows to look, and steps down or turns along it, rather than running off) */
    double edge = 0; const double hs = vlen(vh);
    if (hs > 0.5) {
        const double foot = fmin(world_lowest(W, L_FOOT), world_lowest(W, R_FOOT)), ux = vh.x / hs, uz = vh.z / hs;
        for (int i = 1; i <= 2; i++) if (foot - support_y(W->room, pb->x.x + 0.25 * i * ux, pb->x.z + 0.25 * i * uz, foot + 0.3) > NAV_STEP) { edge = (hs - 0.5) * (hs - 0.5); break; }
    }
    /* soft landings: a foot coming down fast (a stamp) costs its speed squared */
    double land = 0; const double wt = W->total_mass * 9.81;
    for (int k = 0; k < 2; k++) {
        const int fi = k ? R_FOOT : L_FOOT, on = W->touch[fi] > 0.02 * wt;
        if (on && !E->foot_on[k] && E->foot_vy[k] < 0) land += E->foot_vy[k] * E->foot_vy[k];
        E->foot_on[k] = on; E->foot_vy[k] = W->b[fi].v.y;
    }
    const double relax = W->strength > 0 ? b_exp(-0.5 * W->metab) : b_exp(-20 * effort(W) / NJ);   /* muscle cost: relaxed is cheap */
    const double task = 0.35 * move + 0.1 * face + 0.1 * up + 0.05 * b_exp(-2 * act2) + 0.15 * relax;
    double r = (E->cfg.nstyle ? 0.75 * task + 0.25 * style_reward(E) : task) + 6 * prog - 0.05 * land - 0.3 * edge;
    if (d < 0.3 && fabs(sp - E->goal.y) < 0.2) { r += 2; E->goals++; goto_goal(E); }
    return r;
}

/* env_step in two halves around the physics, so a batch can step its worlds together (env_step_many) */
static BHD double env_begin(Env *E, const float *a) {
    World *W = &E->W; int k = 0; double act2 = 0; const float *f = E->filt;
    if (E->cfg.task == TASK_BALANCE && E->cfg.sway > 0) sway_pose(E->ref, W->t, E->cfg.sway);
    for (int j = 0; j < ACT_DIM; j++) E->filt[j] += (float)ACT_FILTER * (a[j] - E->filt[j]);
    if (E->cfg.task == TASK_GOTO && E->cfg.clip) mref_sample(E->cfg.clip, env_phase(E), NULL, NULL, E->ref);   /* the walk cycle as the gait prior: actions act around its pose */
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; Vec r;
        if (d->hinge) { r = vscale(hinge_axis(d->hinge), f[k++]); }
        else { r = v3(f[k], f[k + 1], f[k + 2]); k += 3; }
        W->target[i] = qmul(E->ref[i], qexp(vscale(r, ACT_SCALES[i] * (E->cfg.act_mul > 0 ? E->cfg.act_mul : 1))));
        if (d->hinge && (floor_task(E) || clip_set(E))) W->target[i] = hinge_limit(i, W->target[i]);
    }
    for (int j = 0; j < ACT_DIM; j++) { double dd = a[j] - E->prev_act[j]; act2 += dd * dd; E->prev_act[j] = a[j]; }
    W->grip_cmd[0] = a[ACT_JOINT]; W->grip_cmd[1] = a[ACT_JOINT + 1];   /* grip: the raw command (opening a hand is not low-passed) */
    if (E->cfg.push_max > 0 && W->t >= E->next_push) {
        double ang = env_rand(&E->rng) * 2 * M_PI, f = E->cfg.push_max * (0.3 + 0.7 * env_rand(&E->rng));
        W->push_body = env_rand(&E->rng) < .5 ? CHEST : PELVIS; W->push_force = v3(f * b_cos(ang), 0, f * b_sin(ang)); W->push_left = 0.2;
        E->next_push = W->t + E->cfg.push_every * (0.5 + env_rand(&E->rng));
    }
    if (E->cfg.task == TASK_IMITATE && E->cfg.assist > 0 && !E->cfg.free_heading) {   /* training wheels toward the reference root */
        Quat rr[NB], rw[NB]; Vec r2;
        ref_at(E, env_phase(E) + 1.0 / CTRL_HZ, rr, rw, &r2);
        W->assist = E->cfg.assist; W->assist_x = r2; W->assist_q = rw[PELVIS];   /* where the reference is at the end of this step */
        W->assist_v = vscale(vsub(r2, E->ref_root), CTRL_HZ);
    } else if ((E->cfg.task == TASK_GETUP || E->cfg.task == TASK_GOTO || E->cfg.task == TASK_IMITATE) && E->cfg.assist > 0) {   /* a hand up: toward standing height above the surface under her, upright */
        Vec px = W->b[PELVIS].x, f = qrot(W->b[PELVIS].q, v3(0, 0, 1)); f = vadd(f, qrot(W->b[PELVIS].q, v3(0, 1, 0)));
        W->assist = E->cfg.assist; W->assist_x = v3(px.x, support_y(W->room, px.x, px.z, px.y) + BODY_DEFS[PELVIS].center.y, px.z); W->assist_v = v3(0, 0, 0);
        W->assist_q = qexp(v3(0, b_atan2(f.x, f.z), 0));
    } else W->assist = 0;
    return act2;
}
static BHD double env_end(Env *E, double act2) {
    World *W = &E->W;
    E->step++;
    if (floor_task(E)) return floor_reward(E, act2);
    if (E->cfg.task == TASK_GOTO) return goto_reward(E, act2);
    int fell = clip_set(E) ? 0 : fallen(W);   /* (a get-up's reference lies on the floor: only drifting off it ends one) */
    if (course(E)) {   /* down in the gap, or lying on a platform (a landing that crumples): start over, as the viewer does. Down in
                        * a gap: her feet under the tops by 0.25 m and her pelvis 0.3 m under the clip's (a clip that goes down
                        * itself, off a platform or a wall, may take her there); lying only when under the clip's pelvis too (a vault
                        * passes low over the box) */
        const Vec P = W->b[PELVIS].x; const double sp = support_y(W->room, P.x, P.z, P.y);
        if ((fmin(world_lowest(W, L_FOOT), world_lowest(W, R_FOOT)) < E->clip_shift.y - 0.25 && P.y < E->ref_root.y - 0.3) || (P.y - sp < 0.4 && W->b[HEAD].x.y - sp < 0.7 && P.y < E->ref_root.y - 0.25)) fell = 1;
    }
    if (E->cfg.task == TASK_IMITATE) {
        ref_at(E, env_phase(E), E->ref, E->ref_world, &E->ref_root);
        Vec er = vsub(W->b[PELVIS].x, E->ref_root);
        const int fh = E->cfg.free_heading;
        if (!fh && vlen(er) > (course(E) ? 1.0 : 0.6)) fell = 1;   /* drifted off the reference: end the episode (a course allows a short or long jump that still lands) */
        if (fell || E->step >= E->cfg.steps) E->done = 1;
        if (fell) return 0;
        /* pose: body world rotations; velocity: body angular velocity against the reference's */
        double pose = 0, vel = 0, ee = 0;
        Quat rw2[NB], rr2[NB]; Vec root2; ref_at(E, env_phase(E) + 1.0 / CTRL_HZ, rr2, rw2, &root2);
        /* free heading: her world turned onto the reference's heading (about the vertical) before comparing */
        const Quat to_ref = fh ? qmul(heading_q(E->ref_world[PELVIS]), qconj(heading_q(W->b[PELVIS].q))) : QID;
        for (int i = 1; i < NB; i++) {
            Quat rel = qmul(qconj(W->b[BODY_DEFS[i].parent].q), W->b[i].q); Vec e = qlog(qmul(rel, qconj(E->ref[i]))); pose += vdot(e, e);
            Vec wr = vscale(qlog(qmul(rw2[i], qconj(E->ref_world[i]))), CTRL_HZ); Vec dw = vsub(qrot(to_ref, W->b[i].w), wr); vel += vdot(dw, dw);
        }
        { Vec e = qlog(qmul(qmul(to_ref, W->b[PELVIS].q), qconj(E->ref_world[PELVIS]))); pose += vdot(e, e); }
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
        if (fh) {   /* free heading: her pelvis velocity against the reference's, each in its own heading frame (no path) */
            const Vec vr = qrot(qconj(heading_q(E->ref_world[PELVIS])), vscale(vsub(root2, E->ref_root), CTRL_HZ));
            const Vec vh = qrot(qconj(heading_q(W->b[PELVIS].q)), W->b[PELVIS].v);
            ce = vscale(v3(vh.x - vr.x, 0, vh.z - vr.z), 0.3);   /* 0.3 s of the velocity error, as a distance */
        }
        double track = 0.5 * b_exp(-2 * pose / NJ) + 0.1 * b_exp(-0.1 * vel / NJ) + 0.2 * b_exp(-10 * ee) + 0.2 * b_exp(-10 * vdot(ce, ce));
        double stick = 0;   /* a jump course: her first foot down on the landing platform pays for how far onto it (to 0.3 m) */
        if (course(E) && !E->landed) for (int k = 0; k < 2; k++) {
            const int fi = k ? R_FOOT : L_FOOT; const double on = over_box(room_at(W->room, 1), W->b[fi].x, 0);
            if (on > 0 && world_lowest(W, fi) < E->clip_shift.y + 0.03) { E->landed = 1; stick = 3 * fmin(on, 0.3) / 0.3; break; }
        }
        const double r = 0.75 * track + 0.25 * golden_thread(E, qrot(E->ref_world[PELVIS], fx[HEAD]).y) + stick;
        if (course(E) && E->cfg.chain && E->landed) {   /* landed and steady: the next jump */
            const double T = (E->rc->frames - 1) / E->rc->fps, t = env_phase(E); Vec cv; world_com(W, &cv);
            if (t > T + 0.3 && (vlen(cv) < 0.25 || t > T + 2.5)) chain_next(E);
        }
        return r;
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
