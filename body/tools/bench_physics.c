/* bench_physics.c: physics cost per control step and a checksum of the final state, so a faster engine can
 * be checked to give the same trajectory. Walk imitation at 7% support with small random actions (a
 * realistic mix of contacts, drives and support).   bench_physics [steps] [envs] */
#include "../env.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    const int steps = argc > 1 ? atoi(argv[1]) : 300, nenv = argc > 2 ? atoi(argv[2]) : 4;
    Motions M; if (mref_load("data/motions_ref.bin", &M)) return 1;
    TaskCfg tc = {.task = TASK_IMITATE, .clip = mref_find(&M, "walk"), .steps = 1 << 30, .assist = 0.07, .fixed_start = 1};
    Env *E = calloc(nenv, sizeof(Env)); for (int e = 0; e < nenv; e++) env_reset(&E[e], &tc, 1 + e);
    uint64_t s = 12345; float a[ACT_DIM]; double sum = 0, t0 = now_s(); long n = 0;
    for (int t = 0; t < steps; t++) for (int e = 0; e < nenv; e++) {
        for (int k = 0; k < ACT_DIM; k++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; a[k] = (float)(((s >> 11) * (1.0 / 9007199254740992.0) - 0.5) * 0.2); }
        env_step(&E[e], a); n++;
        if (E[e].done) env_reset(&E[e], &tc, 1000 + t * 7 + e);
    }
    const double dt = now_s() - t0;
    for (int e = 0; e < nenv; e++) for (int i = 0; i < NB; i++) sum += E[e].W.b[i].x.x * (i + 1) + E[e].W.b[i].x.y * 3.1 + E[e].W.b[i].q.z * 7.7;
    printf("%.1f us per control step   checksum %.17g\n", 1e6 * dt / n, sum);
    return 0;
}
