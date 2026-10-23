/* train.c: evolution strategies (OpenAI-ES: antithetic pairs, rank-shaped fitness, Adam) for the body policy.
 *
 *   ./train --policy ssm|ltc|mlp --seconds 300 --pairs 64 --threads 4 --out bal.pol
 * Each pair (theta + sigma eps, theta - sigma eps) shares its episode seed. Observation mean/std are
 * updated from the rollouts after each generation. Prints the mean policy's return and survival time
 * on fixed evaluation seeds.
 */
#include "env.h"
#include "policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double randn(uint64_t *s) { double a = env_rand(s) + 1e-12, b = env_rand(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

typedef struct { double sum[OBS_DIM], sq[OBS_DIM], n; float mean[OBS_DIM], std[OBS_DIM]; } Norm;

/* One episode; returns the return, sets *alive (seconds survived); accumulates sampled observations. */
static double rollout(const PolCfg *pc, const float *th, PolState *ps, const TaskCfg *tc, uint64_t seed, const Norm *nm, double *alive, double *osum, double *osq, double *on) {
    Env E; env_reset(&E, tc, seed); pol_reset(pc, ps);
    float o[OBS_DIM], on_[OBS_DIM], a[ACT_DIM]; double R = 0; uint64_t rs = seed ^ 0x5555;
    while (!E.done) {
        env_obs(&E, o);
        if (osum && env_rand(&rs) < 0.05) { for (int i = 0; i < OBS_DIM; i++) { osum[i] += o[i]; osq[i] += (double)o[i] * o[i]; } (*on)++; }
        for (int i = 0; i < OBS_DIM; i++) { float z = (o[i] - nm->mean[i]) / nm->std[i]; on_[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
        pol_act(pc, th, ps, on_, a);
        R += env_step(&E, a);
    }
    *alive = E.step / (double)CTRL_HZ;
    return R;
}

int main(int argc, char **argv) {
    const char *ptype = "ssm", *out = "body.pol", *in = NULL;
    double seconds = 300, sigma = 0.02, lr = 0.01, wd = 0.005;
    int pairs = 64, threads = 4, seed = 0, eval_eps = 8, hidden = 0;
    TaskCfg tc = {.push_max = 120, .push_every = 2.0, .steps = 300, .init_noise = 0.05};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
#define OPT(nm, dst, conv) if (!strcmp(a, nm) && v) { dst = conv(v); i++; continue; }
        OPT("--policy", ptype, (const char *)) OPT("--out", out, (const char *)) OPT("--load", in, (const char *))
        OPT("--seconds", seconds, atof) OPT("--sigma", sigma, atof) OPT("--lr", lr, atof) OPT("--pairs", pairs, atoi)
        OPT("--threads", threads, atoi) OPT("--seed", seed, atoi) OPT("--hidden", hidden, atoi)
        OPT("--push", tc.push_max, atof) OPT("--steps", tc.steps, atoi)
#undef OPT
        fprintf(stderr, "unknown argument %s\n", a); return 2;
    }
    PolCfg pc = {.in = OBS_DIM, .out = ACT_DIM};
    if (!strcmp(ptype, "mlp")) { pc.type = POL_MLP; pc.hidden = hidden ? hidden : 64; }
    else if (!strcmp(ptype, "ltc")) { pc.type = POL_LTC; pc.hidden = hidden ? hidden : 48; }
    else { pc.type = POL_SSM; pc.hidden = hidden ? hidden : 32; pc.layers = 2; pc.trees = 2; pc.depth = 3; pc.n = 8; pc.p = 8; }
    const int P = pol_nparams(&pc);
    float *th = malloc(P * sizeof(float)), *eps = malloc((size_t)pairs * P * sizeof(float)), *trial = malloc((size_t)threads * P * sizeof(float));
    double *m = calloc(P, sizeof(double)), *vv = calloc(P, sizeof(double)), *g = calloc(P, sizeof(double));
    pol_init(&pc, th, seed);
    Norm nm; memset(&nm, 0, sizeof(nm)); for (int i = 0; i < OBS_DIM; i++) { nm.mean[i] = 0; nm.std[i] = 1; }
    if (in) { FILE *f = fopen(in, "rb"); PolCfg lc; if (!f || fread(&lc, sizeof lc, 1, f) != 1 || lc.type != pc.type || pol_nparams(&lc) != P ||
              fread(th, sizeof(float), P, f) != (size_t)P || fread(nm.mean, sizeof(float), OBS_DIM, f) != OBS_DIM || fread(nm.std, sizeof(float), OBS_DIM, f) != OBS_DIM) { fprintf(stderr, "cannot load %s\n", in); return 1; } fclose(f); }
    PolState **ps = malloc(threads * sizeof(PolState *)); for (int t = 0; t < threads; t++) ps[t] = pol_state_new(&pc);
    double *F = malloc(2 * pairs * sizeof(double)), *alive = malloc(2 * pairs * sizeof(double));
    printf("{\"event\":\"config\",\"policy\":\"%s\",\"params\":%d,\"pairs\":%d,\"sigma\":%g,\"lr\":%g,\"push\":%g}\n", pol_name(pc.type), P, pairs, sigma, lr, tc.push_max);
    uint64_t rng = (uint64_t)seed * 7919 + 13; double t0 = now(), best = -1; int gen = 0;
    while (now() - t0 < seconds) {
        gen++;
        for (size_t i = 0; i < (size_t)pairs * P; i++) eps[i] = (float)randn(&rng);
        uint64_t gseed = rng ^ ((uint64_t)gen << 32);
        double osum[OBS_DIM] = {0}, osq[OBS_DIM] = {0}, on = 0;
#pragma omp parallel num_threads(threads)
        {
#ifdef _OPENMP
            int t = omp_get_thread_num();
#else
            int t = 0;
#endif
            double ls[OBS_DIM] = {0}, lq[OBS_DIM] = {0}, ln = 0; float *w = trial + (size_t)t * P;
#pragma omp for schedule(dynamic)
            for (int k = 0; k < 2 * pairs; k++) {
                const float sgn = k & 1 ? -1.f : 1.f, *e = eps + (size_t)(k >> 1) * P;
                for (int i = 0; i < P; i++) w[i] = th[i] + sgn * (float)sigma * e[i];
                F[k] = rollout(&pc, w, ps[t], &tc, gseed + (k >> 1), &nm, &alive[k], ls, lq, &ln);
            }
#pragma omp critical
            { for (int i = 0; i < OBS_DIM; i++) { osum[i] += ls[i]; osq[i] += lq[i]; } on += ln; }
        }
        /* centred ranks */
        int *idx = malloc(2 * pairs * sizeof(int)); double *rk = malloc(2 * pairs * sizeof(double));
        for (int k = 0; k < 2 * pairs; k++) idx[k] = k;
        for (int a = 1; a < 2 * pairs; a++) { int x = idx[a], b = a - 1; while (b >= 0 && F[idx[b]] > F[x]) { idx[b + 1] = idx[b]; b--; } idx[b + 1] = x; }
        for (int r = 0; r < 2 * pairs; r++) rk[idx[r]] = (double)r / (2 * pairs - 1) - 0.5;
        memset(g, 0, P * sizeof(double));
        for (int k = 0; k < pairs; k++) { const double wgt = rk[2 * k] - rk[2 * k + 1]; const float *e = eps + (size_t)k * P; for (int i = 0; i < P; i++) g[i] += wgt * e[i]; }
        const double b1 = .9, b2 = .999, c1 = 1 - pow(b1, gen), c2 = 1 - pow(b2, gen);
        for (int i = 0; i < P; i++) {
            const double gi = -g[i] / (2 * pairs * sigma) + wd * th[i];   /* minimise the negative fitness */
            m[i] = b1 * m[i] + (1 - b1) * gi; vv[i] = b2 * vv[i] + (1 - b2) * gi * gi;
            th[i] -= (float)(lr * (m[i] / c1) / (sqrt(vv[i] / c2) + 1e-8));
        }
        free(idx); free(rk);
        /* observation statistics */
        for (int i = 0; i < OBS_DIM; i++) { nm.sum[i] += osum[i]; nm.sq[i] += osq[i]; } nm.n += on;
        if (nm.n > 10) for (int i = 0; i < OBS_DIM; i++) { double mu = nm.sum[i] / nm.n, var = nm.sq[i] / nm.n - mu * mu; nm.mean[i] = (float)mu; nm.std[i] = (float)sqrt(fmax(var, 1e-2)); }
        double fm = 0, am = 0; for (int k = 0; k < 2 * pairs; k++) { fm += F[k]; am += alive[k]; }
        if (gen % 5 == 1 || now() - t0 >= seconds) {
            double er = 0, ea = 0;
#pragma omp parallel for num_threads(threads) reduction(+:er, ea)
            for (int k = 0; k < eval_eps; k++) {
#ifdef _OPENMP
                int t = omp_get_thread_num();
#else
                int t = 0;
#endif
                double al; er += rollout(&pc, th, ps[t], &tc, 1000003 + k, &nm, &al, NULL, NULL, NULL); ea += al;
            }
            er /= eval_eps; ea /= eval_eps;
            printf("{\"event\":\"gen\",\"gen\":%d,\"seconds\":%.0f,\"pop_return\":%.1f,\"pop_alive_s\":%.2f,\"eval_return\":%.1f,\"eval_alive_s\":%.2f}\n", gen, now() - t0, fm / (2 * pairs), am / (2 * pairs), er, ea);
            fflush(stdout);
            if (er > best) {
                best = er; FILE *f = fopen(out, "wb");
                if (f) { fwrite(&pc, sizeof pc, 1, f); fwrite(th, sizeof(float), P, f); fwrite(nm.mean, sizeof(float), OBS_DIM, f); fwrite(nm.std, sizeof(float), OBS_DIM, f); fclose(f); }
            }
        }
    }
    printf("{\"event\":\"done\",\"generations\":%d,\"best_eval_return\":%.1f}\n", gen, best);
    return 0;
}
