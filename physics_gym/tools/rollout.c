/* rollout.c: deterministic episode (no pushes, no initial noise) printing the pelvis and head heights per
 * control step, to compare against body.js (tools/rollout_check.mjs). usage: rollout policy.pol steps */
#include "../env.h"
#include "../policy.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); PolCfg pc; if (!f || fread(&pc, sizeof pc, 1, f) != 1) return 1;
    int P = pol_nparams(&pc); float *th = malloc(P * 4), mean[OBS_DIM], sd[OBS_DIM];
    if (fread(th, 4, P, f) != (size_t)P || fread(mean, 4, OBS_DIM, f) != OBS_DIM || fread(sd, 4, OBS_DIM, f) != OBS_DIM) return 1;
    TaskCfg tc = {.push_max = 0, .push_every = 2, .steps = atoi(argv[2]), .init_noise = 0};
    Env E; env_reset(&E, &tc, 1); PolState *ps = pol_state_new(&pc); pol_reset(&pc, ps);
    float o[OBS_DIM], a[ACT_DIM];
    while (!E.done) {
        env_obs(&E, o); for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; o[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
        pol_act(&pc, th, ps, o, a); if (getenv("DUMP") && E.step < atoi(getenv("DUMP"))) { for (int i = 0; i < OBS_DIM; i++) fprintf(stderr, "o%d %.5f\n", i, o[i]); for (int k = 0; k < ACT_DIM; k++) fprintf(stderr, "a%d %.5f\n", k, a[k]); }
        env_step(&E, a);
        printf("%d %.6f %.6f %.6f\n", E.step, E.W.b[PELVIS].x.y, E.W.b[HEAD].x.y, a[0]);
    }
    return 0;
}
