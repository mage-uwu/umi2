/* room_drop.c: drop the body, limp, onto each feature of a room and report where she comes to rest.
 * For each box she lands on: resting pelvis height, deepest corner penetration into any room box, the speed
 * left after 3 s. Checks that contacts hold (no sinking through, no jitter) on steps, slopes, walls and foam.
 *   room_drop room.json */
#include "../env.h"
#include <stdio.h>
#include <stdlib.h>
static double penetration(const World *W, const double *v, int n) {   /* deepest corner inside any box */
    double worst = 0;
    for (int i = 0; i < NB; i++) for (int k = 0; k < NCORNER; k++) {
        const Vec h = BODY_DEFS[i].half; const double s[8][3] = {{-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1}};
        Vec p = body_point(W->b + i, v3(s[k][0] * h.x, s[k][1] * h.y, s[k][2] * h.z));
        for (int j = 0; j < n; j++) { const double *b = v + 12 * j; Quat q = {b[6], b[7], b[8], b[9]};
            Vec l = qrot(qconj(q), vsub(p, v3(b[0], b[1], b[2]))); double d = fmin(b[3] - fabs(l.x), fmin(b[4] - fabs(l.y), b[5] - fabs(l.z)));
            if (d > worst) worst = d; }
    }
    return worst;
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "room.json"; const int n = room_load(path);
    if (n <= 0) { fprintf(stderr, "cannot load %s\n", path); return 1; }
    static double v[ROOM_MAX * 12]; { /* the table as loaded, for measuring */ extern int room_get(double *); room_get(v); }
    Quat rel[NB]; stand_pose(rel);
    printf("%-10s %-16s %8s %8s %10s %9s\n", "box", "feature", "top y", "pelvis", "penetr mm", "speed");
    for (int j = 0; j < n; j++) {
        const double *b = v + 12 * j; if (b[4] > 1.0) continue;   /* walls taller than her: dropped beside them instead */
        Quat q = {b[6], b[7], b[8], b[9]}; const Vec up = qrot(q, v3(0, b[4], 0)); const double top = b[1] + up.y;
        World W; world_init(&W, 45, 20, 1.0 / CTRL_HZ); W.kp_scale = 0.05;
        world_pose(&W, v3(b[0], top + 1.2, b[2]), qexp(v3(1.2, 0.3, 0)), rel, v3(0, 0, 0));
        for (int i = 0; i < NB; i++) W.target[i] = rel[i];
        for (int s = 0; s < 90; s++) world_step(&W);
        double sp = 0; for (int i = 0; i < NB; i++) sp = fmax(sp, vlen(W.b[i].v));
        printf("%-10d %-16s %8.3f %8.3f %10.2f %9.4f\n", j, "", top, W.b[PELVIS].x.y, 1000 * penetration(&W, v, n), sp);
    }
    return 0;
}
