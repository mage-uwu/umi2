/* tss_why.c: why a nanotss walker's episodes end. Runs it from random start phases (as in training), with or
 * without the training's exploration noise, and reports which termination fired and her state then: drift from
 * the reference root split into her forward and sideways directions, pelvis and head height, torso tilt, the
 * gait phase, and how the drift grew over the episode.   tss_why <prefix> <assist> <episodes> <noise 0|1> */
#include "../env.h"
#define main nanotss_main
#include "../../nanotss_blt.c"
#undef main
static uint64_t rs = 99;
static double gs(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; double a = ((rs >> 11) + 1.0) / 9007199254740993.0; rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; double b = (rs >> 11) / 9007199254740992.0; return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
int main(int argc, char **argv) {
    char p[1024]; snprintf(p, sizeof p, "%s.tss", argv[1]); Model *m = load(p, 1, 64); if (!m) return 1;
    const int d = m->c.d; snprintf(p, sizeof p, "%s.head", argv[1]); FILE *f = fopen(p, "rb"); char mg[4]; int dims[3];
    const size_t ES = (size_t)d * OBS_DIM + d + (size_t)ACT_DIM * d + 2 * ACT_DIM; real *E = malloc(ES * sizeof(real)); float mean[OBS_DIM], sd[OBS_DIM];
    if (!f || fread(mg, 1, 4, f) != 4 || fread(dims, sizeof dims, 1, f) != 1 || fread(E, sizeof(real), ES, f) != ES || fread(mean, 4, OBS_DIM, f) != OBS_DIM || fread(sd, 4, OBS_DIM, f) != OBS_DIM) return 1;
    const real *Win = E, *bin = Win + (size_t)d * OBS_DIM, *Wo = bin + d, *bo = Wo + (size_t)ACT_DIM * d, *ls = bo + ACT_DIM;
    Motions M; mref_load("data/motions_ref.bin", &M);
    TaskCfg tc = {.task = TASK_IMITATE, .clip = mref_find(&M, "walk"), .steps = 600, .assist = atof(argv[2])};
    const int neps = atoi(argv[3]), noise = atoi(argv[4]);
    int why[5] = {0}; const char *names[5] = {"time limit", "pelvis < 0.55 m", "head < 1.0 m", "body part touched ground", "drifted > 0.6 m off the reference"};
    double sum_t[5] = {0}, fwd_s = 0, lat_s = 0, tilt_s = 0, ph_hist[10] = {0}; int nd = 0;
    double drift_at[6] = {0}; int drift_n[6] = {0};   /* |drift| at 1, 2, 3, 4, 5, 6 s */
    for (int ep = 0; ep < neps; ep++) {
        Env e; env_reset(&e, &tc, 1000 + ep * 7919); Gen *g = gen_new(&m->c);
        float o[OBS_DIM], a[ACT_DIM]; real x0[512], u[512], r;
        while (!e.done) {
            env_obs(&e, o); for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - mean[i]) / sd[i]; o[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
            for (int i = 0; i < d; i++) x0[i] = bin[i] + dotr(Win + (size_t)i * OBS_DIM, o, OBS_DIM);
            gen_core(m, g, x0, NULL); rmsnorm_fwd(g->x, m->P.norm_f, u, d, &r);
            for (int k = 0; k < ACT_DIM; k++) { double mu = tanh(bo[k] + dotr(Wo + (size_t)k * d, u, d)), v = mu + (noise ? exp(ls[k]) * gs() : 0); a[k] = (float)(v > 1 ? 1 : v < -1 ? -1 : v); }
            env_step(&e, a);
            const double tt = e.step / (double)CTRL_HZ; Vec dr = vsub(e.W.b[PELVIS].x, e.ref_root);
            for (int s = 1; s <= 6; s++) if (e.step == s * CTRL_HZ) { drift_at[s - 1] += hypot(dr.x, dr.z); drift_n[s - 1]++; }
            (void)tt;
        }
        const World *W = &e.W; Vec dr = vsub(W->b[PELVIS].x, e.ref_root);
        int k = 0;
        if (e.step >= tc.steps) k = 0;
        else if (W->b[PELVIS].x.y < 0.55) k = 1;
        else if (W->b[HEAD].x.y < 1.0) k = 2;
        else { int touch = 0; for (int i = 0; i < NB; i++) if (i != L_FOOT && i != R_FOOT && i != L_SHIN && i != R_SHIN && world_lowest(W, i) < 0.05) touch = 1; k = touch ? 3 : 4; }
        why[k]++; sum_t[k] += e.step / (double)CTRL_HZ;
        if (k) {
            /* drift in the reference pelvis's heading frame: forward (+ ahead of the reference) and to her left */
            Vec fw = qrot(e.ref_world[PELVIS], v3(0, 0, 1)); double hn = hypot(fw.x, fw.z); fw = v3(fw.x / hn, 0, fw.z / hn); Vec lf = v3(fw.z, 0, -fw.x);
            fwd_s += vdot(dr, fw); lat_s += fabs(vdot(dr, lf));
            Vec up = qrot(W->b[CHEST].q, v3(0, 1, 0)); tilt_s += acos(fmin(1, up.y)) * 57.2958;
            const Clip *cl = tc.clip; double T = (cl->frames - 1) / cl->fps, ph = fmod(env_phase(&e), T) / T; ph_hist[(int)(ph * 10) % 10]++;
            nd++;
            if (ep < 12) printf("ep %2d: %-34s at %.2f s  drift fwd %+.2f m side %+.2f m  pelvis %.2f m head %.2f m  chest tilt %.0f deg  phase %.2f\n",
                                ep, names[k], e.step / (double)CTRL_HZ, vdot(dr, fw), vdot(dr, lf), W->b[PELVIS].x.y, W->b[HEAD].x.y, acos(fmin(1, up.y)) * 57.2958, ph);
        }
        gen_free(g);
    }
    printf("\n%d episodes, assist %s, %s\n", neps, argv[2], noise ? "with the training's exploration noise" : "mean actions");
    for (int k = 0; k < 5; k++) if (why[k]) printf("  %-34s %3d  (mean %.2f s)\n", names[k], why[k], sum_t[k] / why[k]);
    if (nd) printf("at the end: mean forward drift %+.2f m, mean |sideways drift| %.2f m, chest tilt %.0f deg\n", fwd_s / nd, lat_s / nd, tilt_s / nd);
    printf("gait phase at the end (tenths):"); for (int i = 0; i < 10; i++) printf(" %.0f", ph_hist[i]); printf("\n");
    printf("mean |drift| at 1..6 s:"); for (int s = 0; s < 6; s++) if (drift_n[s]) printf(" %.2f", drift_at[s] / drift_n[s]); printf("\n");
    return 0;
}
