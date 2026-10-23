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
 */
#ifndef ENV_H
#define ENV_H
#include "body.h"
#include "mref.h"
#include <stdint.h>

#define ACT_DIM 31
#define OBS_DIM (NB * 15 + 6)
#define CTRL_HZ 30
#define ACT_FILTER 0.35   /* each control step the joint command moves this share of the way to the new action */
enum { TASK_BALANCE, TASK_IMITATE };

typedef struct {
    int task;
    const Clip *clip;    /* TASK_IMITATE */
    double push_max;     /* largest push force (N), applied for 0.2 s */
    double push_every;   /* mean seconds between pushes */
    int steps;           /* episode length in control steps */
    double init_noise;   /* random joint rotation at reset (rad) */
    int fixed_start;     /* TASK_IMITATE: start at phase 0 instead of a random phase */
    double assist;       /* TASK_IMITATE training aid: strength (0..1) of a spring-damper pulling the pelvis
                          * toward the reference root position and rotation; 0 for the final skill */
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
} Env;

extern const double ACT_SCALES[NB];
void stand_pose(Quat *rel);
void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed);
void env_obs(const Env *E, float *obs);
double env_step(Env *E, const float *act);   /* returns the reward; sets E->done on a fall or at the end */
double env_rand(uint64_t *s);
double env_phase(const Env *E);              /* reference time in the clip (s), or 0 */
#endif
