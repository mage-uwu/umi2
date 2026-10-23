#include "mref.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef __CUDACC__
int mref_load(const char *path, Motions *M) {
    FILE *f = fopen(path, "rb"); char mg[4]; int n;
    if (!f || fread(mg, 1, 4, f) != 4 || memcmp(mg, "MREF", 4) || fread(&n, 4, 1, f) != 1) { if (f) fclose(f); return 1; }
    M->n = n; M->c = calloc(n, sizeof(Clip));
    for (int k = 0; k < n; k++) {
        Clip *c = M->c + k;
        if (fread(c->name, 1, 16, f) != 16 || fread(&c->frames, 4, 1, f) != 1 || fread(&c->fps, 4, 1, f) != 1) { fclose(f); return 1; }
        c->root = malloc(sizeof(Vec) * c->frames); c->q = malloc(sizeof(Quat) * c->frames * NB);
        for (int fr = 0; fr < c->frames; fr++) {
            float v[3 + 4 * NB]; if (fread(v, sizeof v, 1, f) != 1) { fclose(f); return 1; }
            c->root[fr] = v3(v[0], v[1], v[2]);
            for (int i = 0; i < NB; i++) { Quat q = {v[3 + 4 * i], v[4 + 4 * i], v[5 + 4 * i], v[6 + 4 * i]}; c->q[fr * NB + i] = qnorm(q); }
        }
    }
    fclose(f); return 0;
}
const Clip *mref_find(const Motions *M, const char *name) { for (int k = 0; k < M->n; k++) if (!strcmp(M->c[k].name, name)) return M->c + k; return NULL; }
#endif

static BHD Quat qslerp(Quat a, Quat b, double t) {
    double d = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; if (d < 0) { b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z; d = -d; }
    Quat r = {a.w + t * (b.w - a.w), a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)}; (void)d; return qnorm(r);
}
BHD void mref_sample(const Clip *c, double t, Vec *root, Quat *world, Quat *rel) {
    const double T = (c->frames - 1) / c->fps; double u = fmod(t, T); if (u < 0) u += T;
    double x = u * c->fps; int f0 = (int)x; if (f0 >= c->frames - 1) f0 = c->frames - 2; double w = x - f0;
    Quat W[NB];
    for (int i = 0; i < NB; i++) W[i] = qslerp(c->q[f0 * NB + i], c->q[(f0 + 1) * NB + i], w);
    if (root) *root = vadd(vscale(c->root[f0], 1 - w), vscale(c->root[f0 + 1], w));
    if (world) memcpy(world, W, sizeof W);
    if (rel) { rel[0] = W[0]; for (int i = 1; i < NB; i++) rel[i] = qmul(qconj(W[BODY_DEFS[i].parent]), W[i]); }
}
