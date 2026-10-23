/* tss_run.c: a nanotss body policy (ppo_tss.c's .tss + .head) driven deterministically (mean actions).
 * Prints the episode return and length and a checksum of the first actions, so builds can be compared:
 * the AVX-512 build against the portable one (the path the browser's WebAssembly build takes).
 *   tss_run <prefix> <task> <assist> [steps]
 */
#include "../env.h"
#define main nanotss_main
#include "../../nanotss_blt.c"
#undef main
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: tss_run prefix task assist [steps]\n"); return 2; }
    char p[1024]; snprintf(p, sizeof p, "%s.tss", argv[1]);
    Model *m = load(p, 1, 64); if (!m) { fprintf(stderr, "cannot load %s\n", p); return 1; }
    const int d = m->c.d; snprintf(p, sizeof p, "%s.head", argv[1]); FILE *f = fopen(p, "rb"); char mg[4]; int dims[3];
    const size_t ES = (size_t)d * OBS_DIM + d + (size_t)ACT_DIM * d + 2 * ACT_DIM; real *E = malloc(ES * sizeof(real)); float mean[OBS_DIM], sd[OBS_DIM];
    if (!f || fread(mg, 1, 4, f) != 4 || fread(dims, sizeof dims, 1, f) != 1 || dims[0] != d || fread(E, sizeof(real), ES, f) != ES || fread(mean, 4, OBS_DIM, f) != OBS_DIM || fread(sd, 4, OBS_DIM, f) != OBS_DIM) { fprintf(stderr, "bad head\n"); return 1; }
    const real *Win = E, *bin = Win + (size_t)d * OBS_DIM, *Wo = bin + d, *bo = Wo + (size_t)ACT_DIM * d;
    Motions M; TaskCfg tc = {.steps = argc > 4 ? atoi(argv[4]) : 600, .assist = atof(argv[3])};
    if (strcmp(argv[2], "balance")) { if (mref_load("data/motions_ref.bin", &M) || !(tc.clip = mref_find(&M, argv[2]))) return 1; tc.task = TASK_IMITATE; tc.fixed_start = 1; }
    Env e; env_reset(&e, &tc, 1); Gen *g = gen_new(&m->c);
    double ret = 0, sum = 0; float o[OBS_DIM], a[ACT_DIM]; real x0[512], u[512], r;
    while (!e.done) {
        env_obs(&e, o); for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; o[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
        for (int i = 0; i < d; i++) { real s = bin[i]; for (int j = 0; j < OBS_DIM; j++) s += Win[(size_t)i * OBS_DIM + j] * o[j]; x0[i] = s; }
        gen_core(m, g, x0, NULL); rmsnorm_fwd(g->x, m->P.norm_f, u, d, &r);
        for (int k = 0; k < ACT_DIM; k++) { a[k] = (float)tanh(bo[k] + dotr(Wo + (size_t)k * d, u, d)); if (e.step < 10) sum += a[k] * (k + 1); if (e.step == 0 && getenv("FIRST")) printf("%.7f ", a[k]); }
        ret += env_step(&e, a);
    }
    printf("return %.2f  seconds %.2f  checksum(first 10 steps) %.9f\n", ret, e.step / (double)CTRL_HZ, sum);
    return 0;
}
