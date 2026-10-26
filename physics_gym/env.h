/* env.h: the control tasks on top of body.c.
 *
 * Control at 30 Hz. The action holds one value per joint degree of freedom (3 for a ball joint, 1 for a
 * hinge: 31 in all), each in [-1, 1]; a joint's target is the reference rotation followed by
 * exp(scale_j * f), where f is the action low-passed (ACT_FILTER), with a per-joint range (wide at the hips and knees so she can step). Observations are
 * in the heading frame (the pelvis yaw removed), plus the reference clip's phase as sin/cos.
 *
 *   TASK_BALANCE  hold the stand pose's upper body while being pushed; the legs are free, so stepping to
 *                 recover is rewarded like standing still.
 *   TASK_IMITATE  follow a motion clip (DeepMimic style): start at a random phase in the reference pose and
 *                 velocity, reward pose, joint velocity, hand/foot and root tracking; the root's motion
 *                 accumulates across loops of the clip. A posture term ("golden thread") also pays for a
 *                 lifted, upright, steady head and level shoulders.
 *   TASK_CRAWL    no clip: from all fours, move forward on palms and knees (a forward speed near 0.4 m/s; the
 *                 hand ends of the forearms and the knee ends of the shins on the floor, palms toward the feet;
 *                 torso, head, upper arms and elbows off it), calm and low effort. Rolling over, or the
 *                 pelvis, chest or head touching the floor, ends the episode. The gait is whatever the controller
 *                 finds. Elbows and knees only bend one way (floor skills; env.c hinge_limit).
 *   TASK_GETUP    no clip: from where a limp fall left her (or from all fours), rise to a calm stand: head and
 *                 pelvis up, chest upright, then the stand pose held still, with low effort and smooth commands.
 * With style clips (TaskCfg.style; motion capture of a person crawling or getting up), the floor tasks also pay
 * for moving as that person does: a pose-pair match against the clips, independent of timing and path.
 * The two floor tasks measure her heading from the pelvis's forward and up axes together (forward alone is
 * vertical when she lies face down), and take act_mul times the joint ranges.
 *   TASK_GOTO     (physics_gym) reach a goal point anywhere in the room, on the floor or on top of a surface, preferably
 *                 walking: pays for moving toward it near walking speed, facing it, an upright chest, low effort and
 *                 smooth commands, and (with style clips: the walk capture) for walking poses; reaching it (within
 *                 0.3 m, on its surface) pays a bonus and a new goal is drawn. Relaxation pays: the muscle cost (body.c
 *                 World.metab, activation squared summed over joints) is the effort term. The main pay is the distance gained toward
 *                 it each step. The walk clip's pose at the current phase is the reference the actions act around (a gait
 *                 prior, with no root tracking). Starts on a free spot of floor, mid-stride in a random frame of the walk clip
 *                 (70%) or standing. Without a clip (task goto_free) there is no gait prior: the actions act around the
 *                 stand pose, there is no phase input, and the gait is the controller's own; the motion capture only seeds
 *                 the starts (START_N) and, in training, judges the style (gpu/ppo_gpu.py --amp).
 *
 * Terrain is felt, not seen: the observation has no map or height scan. Each body reports its contact force over the
 * last control step (touch, from the physics' contact solve), the pelvis height is measured from her lowest foot
 * corner (kinematics, the same as the height above the floor while she stands there), and the foot flags come from
 * touch. What the room is like underfoot is left to the controller's memory. The environment itself (fall test,
 * reward, starts and goals) measures heights from the surface under her (body.c support_y), which on open floor is
 * y = 0 as before. The observation ends with the goal relative to her (zeros for other tasks), the 14 touches and
 * the 13 joints' activations (muscle sense: how hard each joint's muscles are working, as tendon organs report it).
 */
#ifndef ENV_H
#define ENV_H
#include "body.h"
#include "mref.h"
#include <stdint.h>

#define ACT_JOINT 31        /* joint targets: 3 per ball joint, 1 per hinge (body/ policies have only these) */
#define ACT_DIM (ACT_JOINT + 2)   /* then grip, left and right hand (physics_gym): close above GRIP_ON, let go at <= 0 */
#define OBS_BASE (NB * 15 + 6)   /* the bodies, foot contacts, centre of mass, clip phase (body/ policies read these) */
#define OBS_GOAL 4                /* goal x, y, z from her pelvis in the heading frame (horizontal at most 4 m), distance */
#define OBS_GRIP 10               /* per hand: holding (0/1); the nearest holdable point within 1 m, from the hand in the heading
                                   * frame (x, y, z), and whether there is one (0/1) */
#define OBS_PROBE 16              /* cheap looks ahead (heading frame): the ground under 6 points ahead, from her lowest foot (m,
                                   * within +-1); then 10 horizontal rays (knee and hip height, 5 directions): distance to the
                                   * first box over 1.5 m (1: nothing within reach) */
#define OBS_PATH 4                /* TASK_GOTO in a room: the path helper's waypoint, about 1 m along the shortest walkable path
                                   * to the goal (env.c nav_waypoint over the room's nav grid), from her pelvis in the heading frame
                                   * (x, y, z), and the path still to go (m, at most 10) */
#define OBS_LOOK 22                /* TASK_GOTO: where to step. The path a little nearer and further on (about 0.5 and 2 m along it:
                                   * x, y, z from her pelvis in the heading frame), so its turns and its rises and falls show
                                   * before she is at them; then the ground all round her: 8 directions (heading frame, every
                                   * 45 deg from straight ahead, turning left) at 0.4 and 0.8 m, from her lowest foot (m, within
                                   * +-1), so an edge, a drop or a step beside her is felt as well as one ahead */
#define OBS_DIM (OBS_BASE + OBS_GOAL + NB + NB - 1 + OBS_GRIP + OBS_PROBE + OBS_PATH + OBS_LOOK)   /* then touch: each body's mean contact force over the step / body
                                                       weight; then muscle sense: each joint's activation (drive torque / strength);
                                                       then grip */
#define CTRL_HZ 30
#define ACT_FILTER 0.35
#define STYLE_FEAT (4 * (NB - 1) + 3)   /* pose features: joint rotations, gravity in the pelvis frame */   /* each control step the joint command moves this share of the way to the new action */
enum { TASK_BALANCE, TASK_IMITATE, TASK_CRAWL, TASK_GETUP, TASK_GOTO };
#define MAX_STYLE 64   /* style clips, or the clips of a clip set (a course bank: one per room), at most */

typedef struct {
    int task;
    const Clip *clip;    /* TASK_IMITATE; TASK_GOTO: its cycle sets the phase input (the walk clip) */
    double push_max;     /* largest push force (N), applied for 0.2 s */
    double sway;         /* TASK_BALANCE: amplitude (rad) of the relaxed idle motion in the stand reference; 0 = still */
    double arm_tone;     /* muscle tone of shoulders and elbows (0 or 1 = full; below 1 the arms hang looser) */
    double push_every;   /* mean seconds between pushes */
    int steps;           /* episode length in control steps */
    double init_noise;   /* random joint rotation at reset (rad) */
    int fixed_start;     /* TASK_IMITATE: start at phase 0 instead of a random phase */
    double strength;     /* her muscles' strength and speed times this (0 = 1; World.strength): an athlete for parkour */
    double start0;       /* TASK_IMITATE course: this share of episodes from the clip's start (the rest as before) */
    double drift;        /* TASK_IMITATE: how far (m) her pelvis may drift from the reference before the episode ends
                          * (0 = 1.0 on a course, 0.6 otherwise) */
    int assist_pos_only; /* TASK_IMITATE: the hand-up along the reference path only, not holding the pelvis's rotation
                          * (a tracker trained with that leaned on it for balance: none of it without) */
    double tail;         /* TASK_IMITATE course: the episode ends this long (s) after the clip does, as a time limit (not a fall);
                          * 0 = it holds the clip's last frame to `steps` (most PARC clips end at a run: no one stops dead) */
    double assist;       /* TASK_IMITATE training aid: strength (0..1) of a spring-damper pulling the pelvis
                          * toward the reference root position and rotation; 0 for the final skill.
                          * TASK_GETUP: a spring toward standing height above where her pelvis is, upright */
    double act_mul;      /* joint ranges times this (0 = 1) */
    const Clip *style[MAX_STYLE]; int nstyle;   /* floor tasks: motion style prior (env_style_prepare each first). TASK_IMITATE: the clip set
                          * (the CMU get-ups): each episode imitates one, from a random point, once through and then holding
                          * its last frame, with no phase input and the floor tasks' heading frame and hinge limits, so the
                          * policy reads from her body where in the motion she is (and TASK_GETUP can carry on from it) */
    double style_start;  /* TASK_GETUP: share of episodes starting in a random frame of a style clip. TASK_IMITATE over a clip set:
                          * share starting from a real fall, the nearest clip frame taking over (env.c matched_start) */
    int free_heading;    /* TASK_IMITATE: imitate the clip's style, not its path: posture, limbs and speed in her own heading frame;
                          * only a fall ends the episode (not drifting off the clip's line), so she learns to recover */
    double goal_tops;    /* TASK_GOTO: share of goals put on top of a room box (a stair landing, platform, foam, stone; at most
                          * 0.8 m up), the rest on open floor; 0 = 0.3. TASK_BALANCE: share of episodes that start standing on
                          * the top of a room box (ramps too: env.c top_start); 0 = none */
    double goal_cone;    /* TASK_GOTO: goals are drawn within this angle (rad) of her heading; 0 = any direction (a curriculum:
                          * walking to a goal ahead comes before turning toward one) */
    int chain;           /* a jump course: jumps one after another, as the viewer chains them. Once she has landed on the far
                          * platform and is steady (the clip over by 0.3 s and her centre of mass under 0.25 m/s, or 2.5 s
                          * after its end), she is carried back onto the takeoff platform (the tops are level), turned so her
                          * feet point along the clip's start (the viewer aims each jump by her feet) give or take a random
                          * chain_yaw, and the clip starts again: every jump after the first starts from a real landing */
    double chain_yaw;    /* the largest aiming error of a chained jump (rad) */
} TaskCfg;

typedef struct {
    World W;
    Quat ref[NB];        /* current reference joint rotations (child relative to parent) */
    Quat ref_world[NB];  /* current reference body rotations (TASK_IMITATE) */
    Vec ref_root;        /* current reference pelvis position (TASK_IMITATE) */
    TaskCfg cfg;
    uint64_t rng;
    int step, done;
    double t0, next_push;
    Vec clip_shift;      /* world offset of the clip's root (xz), so the reference starts under her */
    float prev_act[ACT_DIM], filt[ACT_DIM];   /* last raw action; low-passed action that sets the targets */
    Vec head_v;          /* head velocity at the previous control step (for the head-steadiness reward) */
    float style_prev[STYLE_FEAT]; int style_has_prev;   /* the previous step's pose features (style prior) */
    Vec goal; int goals;   /* TASK_GOTO: the goal point (on a surface) and how many she has reached this episode */
    double goal_d;         /* TASK_GOTO: her horizontal distance to the goal at the last step */
    double path_len;       /* TASK_GOTO in a room: the path to go at the last step (m) */
    int foot_on[2]; double foot_vy[2];   /* each foot on the ground at the last step, and its vertical speed then (landings) */
    const Clip *rc;      /* TASK_IMITATE: this episode's reference clip (cfg.clip, or one of a clip set) */
    int landed;          /* a jump course: 0 until her first foot comes down on the landing platform (1 then, or from the start
                          * when she did not start on the takeoff platform) */
    int jumps;           /* a chained jump course: jumps landed this episode (each counted as the next one starts) */
    unsigned char nav_d[NAV_N * NAV_N];  /* TASK_GOTO in a room: steps from each nav cell to the goal's (255: no way there) */
    unsigned short nav_q[NAV_N * NAV_N]; /* the search's queue */
} Env;

#ifndef __CUDACC__   /* the GPU build defines it ahead of every use, in one unit */
extern const double ACT_SCALES[NB];
#endif
BHD void stand_pose(Quat *rel);
/* On hands and knees: joint rotations and the root rotation (torso level, facing the floor). */
BHD void crawl_pose(Quat *rel, Quat *root_q);
/* The stand pose with relaxed idle motion at time t, scaled by amp (rad of ankle sway): ankle sway forward/back
 * (0.23 Hz) and sideways (0.17 Hz) counterbalanced at the hips and spine, a slow weight shift between the
 * legs (0.07 Hz), breathing in the chest (0.25 Hz) and slow drifts of the head. */
BHD void sway_pose(Quat *rel, double t, double amp);
#define FALL_N 1024   /* the matched starts' falls (env.c fall_make), simulated once */
typedef struct { Vec x[NB], v[NB], w[NB]; Quat q[NB]; } FallState;
/* TASK_GOTO without a clip (the gait the controller's own, its style from a learned motion discriminator: gpu/ppo_gpu.py
 * --amp): 80% of episodes start in a moment of the motion capture, its pose and velocities (reference state
 * initialisation), turned to a random heading; the bank (pelvis over the origin) is filled from outside */
#define START_N 4096
void env_set_starts(const FallState *s, int n);   /* the CPU build's bank; the GPU's is tss_set_starts */
/* What the motion discriminator sees of a moment: the bodies' block of the observation (OBS_BASE's first NB x 15:
 * positions from the pelvis, axes, velocities, in the heading frame; the pelvis's height above her lowest foot), the
 * same numbers for her in the physics and for a frame of motion capture. */
#define AMP_FEAT (NB * 15)
BHD void env_body_feats(const World *W, int floor, float *o);
BHD void fall_make(int i);
void env_fall_bank(void);   /* the CPU build's bank; the GPU's is tss_fall_bank */
BHD void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed);
BHD void env_obs(const Env *E, float *obs);
BHD double env_step(Env *E, const float *act);   /* returns the reward; sets E->done on a fall or at the end */
/* env_step for n environments (sharing substeps and iters), their physics 8 at a time (body_x8.c) */
void env_step_many(Env *const *E, const float *const *act, double *reward, int n);
BHD double env_rand(uint64_t *s);
BHD double env_phase(const Env *E);
BHD void env_set_goal(Env *E, Vec goal);   /* TASK_GOTO: put the goal somewhere else (the viewer's goal marker) */
/* Motion style prior (floor tasks): each step her last two poses are compared with every pair of consecutive
 * frames of the style clips; the closest match pays. Call once per clip before stepping (not thread-safe). */
void env_style_prepare(const Clip *c);
#endif
