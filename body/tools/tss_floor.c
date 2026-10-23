/* tss_floor.c: a nanotss floor controller (crawl or get-up) over many start states, mean actions.
 * Get-up: how often she reaches a calm stand (head > 1.3 m, pelvis > 0.8 m, slow, for 1 s: the viewer's test)
 * and how soon; crawl: distance covered and how often she rolls over.   tss_floor <prefix> crawl|getup <assist> <episodes> */
#include "../env.h"
#define main nanotss_main
#include "../../nanotss_blt.c"
#undef main
int main(int argc, char **argv) {
    char p[1024]; snprintf(p, sizeof p, "%s.tss", argv[1]); Model *m = load(p, 1, 64); if (!m) return 1;
    const int d = m->c.d; snprintf(p, sizeof p, "%s.head", argv[1]); FILE *f = fopen(p, "rb"); char mg[4]; int dims[3];
    const size_t ES = (size_t)d * OBS_DIM + d + (size_t)ACT_DIM * d + 2 * ACT_DIM; real *E = malloc(ES * sizeof(real)); float mean[OBS_DIM], sd[OBS_DIM];
    if (!f || fread(mg, 1, 4, f) != 4 || fread(dims, sizeof dims, 1, f) != 1 || fread(E, sizeof(real), ES, f) != ES || fread(mean, 4, OBS_DIM, f) != OBS_DIM || fread(sd, 4, OBS_DIM, f) != OBS_DIM) return 1;
    const real *Win = E, *bin = Win + (size_t)d * OBS_DIM, *Wo = bin + d, *bo = Wo + (size_t)ACT_DIM * d;
    const int crawl = !strcmp(argv[2], "crawl"), n = atoi(argv[4]);
    TaskCfg tc = {.task = crawl ? TASK_CRAWL : TASK_GETUP, .steps = crawl ? 300 : 360, .act_mul = 2.5, .assist = atof(argv[3])};
    int ok = 0, rolled = 0; double tsum = 0, dist = 0, ph = 0, hh = 0;
    for (int ep = 0; ep < n; ep++) {
        Env e; env_reset(&e, &tc, 5000 + ep * 104729); Gen *g = gen_new(&m->c); float o[OBS_DIM], a[ACT_DIM]; real x0[512], u[512], r;
        const Vec x0p = e.W.b[PELVIS].x; double calm = -1, stood = -1;
        while (!e.done && stood < 0) {
            env_obs(&e, o); for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; o[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
            for (int i = 0; i < d; i++) x0[i] = bin[i] + dotr(Win + (size_t)i * OBS_DIM, o, OBS_DIM);
            gen_core(m, g, x0, NULL); rmsnorm_fwd(g->x, m->P.norm_f, u, d, &r);
            for (int k = 0; k < ACT_DIM; k++) a[k] = (float)tanh(bo[k] + dotr(Wo + (size_t)k * d, u, d));
            env_step(&e, a);
            if (!crawl) { Vec cv; world_com(&e.W, &cv); const int still = e.W.b[HEAD].x.y > 1.3 && e.W.b[PELVIS].x.y > 0.8 && hypot(cv.x, cv.z) < 0.3;
                if (!still) calm = -1; else if (calm < 0) calm = e.W.t; if (calm >= 0 && e.W.t - calm > 1) stood = e.W.t; }
        }
        if (crawl) { dist += hypot(e.W.b[PELVIS].x.x - x0p.x, e.W.b[PELVIS].x.z - x0p.z); rolled += e.step < tc.steps; }
        else { if (stood > 0) { ok++; tsum += stood; } ph += e.W.b[PELVIS].x.y; hh += e.W.b[HEAD].x.y; }
        gen_free(g);
    }
    if (crawl) printf("crawl, assist %s: %.2f m per 10 s on average, rolled over in %d of %d\n", argv[3], dist / n, rolled, n);
    else printf("get up, assist %s: stood calmly in %d of %d (mean %.1f s); at the end pelvis %.2f m, head %.2f m on average\n", argv[3], ok, n, ok ? tsum / ok : 0, ph / n, hh / n);
    return 0;
}
