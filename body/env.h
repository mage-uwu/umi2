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
 */
#ifndef ENV_H
#define ENV_H
#include "body.h"
#include "mref.h"
#include <stdint.h>

#define ACT_DIM 31
#define OBS_DIM (NB * 15 + 6)
#define CTRL_HZ 30
#define ACT_FILTER 0.35
#define STYLE_FEAT (4 * (NB - 1) + 3)   /* pose features: joint rotations, gravity in the pelvis frame */   /* each control step the joint command moves this share of the way to the new action */
enum { TASK_BALANCE, TASK_IMITATE, TASK_CRAWL, TASK_GETUP };

typedef struct {
    int task;
    const Clip *clip;    /* TASK_IMITATE */
    double push_max;     /* largest push force (N), applied for 0.2 s */
    double sway;         /* TASK_BALANCE: amplitude (rad) of the relaxed idle motion in the stand reference; 0 = still */
    double arm_tone;     /* muscle tone of shoulders and elbows (0 or 1 = full; below 1 the arms hang looser) */
    double push_every;   /* mean seconds between pushes */
    int steps;           /* episode length in control steps */
    double init_noise;   /* random joint rotation at reset (rad) */
    int fixed_start;     /* TASK_IMITATE: start at phase 0 instead of a random phase */
    double assist;       /* TASK_IMITATE training aid: strength (0..1) of a spring-damper pulling the pelvis
                          * toward the reference root position and rotation; 0 for the final skill.
                          * TASK_GETUP: a spring toward standing height above where her pelvis is, upright */
    double act_mul;      /* joint ranges times this (0 = 1) */
    const Clip *style[8]; int nstyle;   /* floor tasks: motion style prior (env_style_prepare each first) */
    double style_start;  /* TASK_GETUP: share of episodes starting in a random frame of a style clip */
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
BHD void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed);
BHD void env_obs(const Env *E, float *obs);
BHD double env_step(Env *E, const float *act);   /* returns the reward; sets E->done on a fall or at the end */
/* env_step for n environments (sharing substeps and iters), their physics 8 at a time (body_x8.c) */
void env_step_many(Env *const *E, const float *const *act, double *reward, int n);
BHD double env_rand(uint64_t *s);
BHD double env_phase(const Env *E);              /* reference time in the clip (s), or 0 */
/* Motion style prior (floor tasks): each step her last two poses are compared with every pair of consecutive
 * frames of the style clips; the closest match pays. Call once per clip before stepping (not thread-safe). */
void env_style_prepare(const Clip *c);
#endif
