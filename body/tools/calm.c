/* calm.c: how relaxed a balance controller stands (no pushes, 10 s): head and pelvis spin, centre-of-mass sway,
 * muscle effort (joint target vs actual, rad^2 per joint) and command change per step.
 *   usage: calm [policy.pol]   (no policy: zero actions) */
#include "../env.h"
#include "../policy.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    PolCfg pc; float *th = NULL, mean[OBS_DIM], sd[OBS_DIM]; int use = argc > 1;
    if (use) { FILE *f = fopen(argv[1], "rb"); if (!f || fread(&pc, sizeof pc, 1, f) != 1) return 1; int P = pol_nparams(&pc); th = malloc(P * 4);
        if (fread(th, 4, P, f) != (size_t)P || fread(mean, 4, OBS_DIM, f) != OBS_DIM || fread(sd, 4, OBS_DIM, f) != OBS_DIM) return 1; fclose(f); }
    double hs = 0, ps = 0, cs = 0, ef = 0, da = 0, amp = 0; int n = 0, falls = 0;
    for (int seed = 1; seed <= 8; seed++) {
        TaskCfg tc = {.task = TASK_BALANCE, .steps = 300, .init_noise = 0.05, .push_max = getenv("PUSH") ? atof(getenv("PUSH")) : 0, .push_every = 2, .sway = getenv("SWAY") ? atof(getenv("SWAY")) : 0}; Env E; env_reset(&E, &tc, seed);
        PolState *ps_ = use ? pol_state_new(&pc) : NULL; if (use) pol_reset(&pc, ps_);
        float o[OBS_DIM], a[ACT_DIM] = {0}, pa[ACT_DIM] = {0}; Vec cmin = v3(1e9, 0, 1e9), cmax = v3(-1e9, 0, -1e9);
        while (!E.done) {
            env_obs(&E, o);
            if (use) { for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; o[i] = z > 5 ? 5 : z < -5 ? -5 : z; } pol_act(&pc, th, ps_, o, a); }
            env_step(&E, a);
            if (E.step < 30) continue;   /* skip settling */
            Vec cv, cc = world_com(&E.W, &cv);
            if (cc.x < cmin.x) cmin.x = cc.x; if (cc.x > cmax.x) cmax.x = cc.x; if (cc.z < cmin.z) cmin.z = cc.z; if (cc.z > cmax.z) cmax.z = cc.z;
            hs += vlen(E.W.b[HEAD].w); ps += vlen(E.W.b[PELVIS].w); cs += sqrt(cv.x * cv.x + cv.z * cv.z);
            for (int i = 1; i < NB; i++) { Quat rel = qmul(qconj(E.W.b[BODY_DEFS[i].parent].q), E.W.b[i].q); Vec e = qlog(qmul(E.W.target[i], qconj(rel))); ef += vdot(e, e) / NJ; }
            for (int k = 0; k < ACT_DIM; k++) { da += fabs(a[k] - pa[k]) / ACT_DIM; pa[k] = a[k]; }
            n++;
        }
        if (E.step < 300) falls++;
        amp += (cmax.x - cmin.x > cmax.z - cmin.z ? cmax.x - cmin.x : cmax.z - cmin.z) / 2;
    }
    printf("%-28s head spin %.3f rad/s  pelvis spin %.3f rad/s  COM sway %.4f m/s  effort %.5f rad^2/joint  command change %.4f  sway amplitude %.1f mm  falls %d/8\n",
           use ? argv[1] + (strrchr(argv[1], '/') ? strrchr(argv[1], '/') - argv[1] + 1 : 0) : "zero actions", hs / n, ps / n, cs / n, ef / n, da / n, amp / 8 * 1000, falls);
    return 0;
}
