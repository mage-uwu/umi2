/* env.h: the control task on top of body.c.
 *
 * Control at 30 Hz. The action holds one value per joint degree of freedom (3 for a ball joint, 1 for a
 * hinge: 31 in all), each in [-1, 1]; the joint target is the reference pose's rotation followed by
 * exp(ACT_SCALE * a). Observations are expressed in the heading frame (the pelvis yaw removed) so the
 * policy does not depend on which way she faces.
 *
 * Task "balance": stand in the page's stand pose for 10 s while being pushed from random directions.
 */
#ifndef ENV_H
#define ENV_H
#include "body.h"
#include <stdint.h>

#define ACT_DIM 31
#define OBS_DIM (NB * 15 + 4)
#define ACT_SCALE 0.6
#define CTRL_HZ 30

typedef struct {
    double push_max;     /* largest push force (N), applied for PUSH_TIME s */
    double push_every;   /* mean seconds between pushes */
    int steps;           /* episode length in control steps */
    double init_noise;   /* random joint rotation at reset (rad) */
} TaskCfg;

typedef struct {
    World W;
    Quat ref[NB];        /* reference pose: joint rotations (child relative to parent) */
    TaskCfg cfg;
    uint64_t rng;
    int step, done;
    double next_push, last_reward;
    float prev_act[ACT_DIM];
} Env;

void stand_pose(Quat *rel);
void env_reset(Env *E, const TaskCfg *cfg, uint64_t seed);
void env_obs(const Env *E, float *obs);
double env_step(Env *E, const float *act);   /* returns the reward; sets E->done on a fall or at the end */
double env_rand(uint64_t *s);
#endif
