/* footlock.c: restore the travel that in-place clips lost. The shader page's retargeting keeps the hips near
 * the origin (walk, run), but a physical body can only move by pushing off its feet. For each frame the lower
 * foot is planted: the root moves horizontally by whatever keeps that foot still on the ground.
 *   usage: footlock in.bin out.bin clip [clip...]   (rewrites the named clips, copies the rest) */
#include "../mref.h"
#include <stdio.h>
#include <stdlib.h>
static int save(const char *path, const Motions *M) {
    FILE *f = fopen(path, "wb"); if (!f) return 1;
    fwrite("MREF", 1, 4, f); fwrite(&M->n, 4, 1, f);
    for (int k = 0; k < M->n; k++) {
        const Clip *c = M->c + k; char nm[16] = {0}; strncpy(nm, c->name, 16);
        fwrite(nm, 1, 16, f); fwrite(&c->frames, 4, 1, f); fwrite(&c->fps, 4, 1, f);
        for (int fr = 0; fr < c->frames; fr++) {
            float v[3 + 4 * NB] = {(float)c->root[fr].x, (float)c->root[fr].y, (float)c->root[fr].z};
            for (int i = 0; i < NB; i++) { Quat q = c->q[fr * NB + i]; v[3 + 4 * i] = q.w; v[4 + 4 * i] = q.x; v[5 + 4 * i] = q.y; v[6 + 4 * i] = q.z; }
            fwrite(v, sizeof v, 1, f);
        }
    }
    fclose(f); return 0;
}
int main(int argc, char **argv) {
    Motions M; if (argc < 4 || mref_load(argv[1], &M)) { fprintf(stderr, "usage: footlock in.bin out.bin clip...\n"); return 2; }
    World W; world_init(&W, 45, 20, 1.0 / 30);
    for (int a = 3; a < argc; a++) {
        Clip *c = (Clip *)mref_find(&M, argv[a]); if (!c) { fprintf(stderr, "no clip %s\n", argv[a]); return 1; }
        Vec off = v3(0, 0, 0), prevL = {0}, prevR = {0}; Vec *nr = malloc(sizeof(Vec) * c->frames);
        for (int fr = 0; fr < c->frames; fr++) {
            Quat rel[NB]; rel[0] = c->q[fr * NB];
            for (int i = 1; i < NB; i++) rel[i] = qmul(qconj(c->q[fr * NB + BODY_DEFS[i].parent]), c->q[fr * NB + i]);
            world_pose(&W, c->root[fr], rel[0], rel, v3(0, 0, 0));
            Vec L = W.b[L_FOOT].x, R = W.b[R_FOOT].x;
            if (fr > 0) {   /* planted foot: the lower one; move the root so it does not slide */
                Vec d = L.y < R.y ? vsub(prevL, L) : vsub(prevR, R); d.y = 0;
                off = vadd(off, d);
            }
            prevL = L; prevR = R;
            nr[fr] = vadd(c->root[fr], off);
        }
        Vec tr = vsub(nr[c->frames - 1], nr[0]);
        printf("%s: travel %.2f m over %.1f s (%.2f m/s), direction (%.2f, %.2f)\n", c->name, vlen(tr), (c->frames - 1) / c->fps, vlen(tr) / ((c->frames - 1) / c->fps), tr.x, tr.z);
        free(c->root); c->root = nr;
    }
    return save(argv[2], &M);
}
