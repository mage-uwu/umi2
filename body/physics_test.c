/* physics_test.c: sanity checks for body.c.
 *   ./physics_test statue   joints very stiff: should stand (centre of mass is over the feet)
 *   ./physics_test stand    the page's stand pose held by the normal PD muscles: how long until a fall
 *   ./physics_test push     stand pose, a push on the chest at t = 1 s
 * Prints the pelvis height and the centre of mass over time, plus a final verdict.
 */
#include "body.h"
#include <stdio.h>
#include <stdlib.h>

static void stand_pose(Quat *rel) {
    for (int i = 0; i < NB; i++) rel[i] = QID;
    const double a = 68 * M_PI / 180;
    rel[L_UARM] = qexp(v3(0, 0, -a)); rel[R_UARM] = qexp(v3(0, 0, a));
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "stand";
    World W; world_init(&W, 45, 20, 1.0 / 30);
    Quat rel[NB]; stand_pose(rel);
    world_pose(&W, BODY_DEFS[PELVIS].center, QID, rel, v3(0, 0, 0));
    for (int i = 0; i < NB; i++) W.target[i] = rel[i];
    if (!strcmp(mode, "statue")) { W.kp_scale = atof(getenv("KP") ? getenv("KP") : "10"); W.kd_scale = atof(getenv("KD") ? getenv("KD") : "3"); }   /* nearly rigid joints */
    double fall = -1, maxv = 0;
    for (int step = 0; step < 30 * 10; step++) {
        if (!strcmp(mode, "push") && step == 30) { W.push_body = CHEST; W.push_force = v3(0, 0, 150); W.push_left = 0.1; }
        world_step(&W);
        for (int i = 0; i < NB; i++) { double s = vlen(W.b[i].v); if (s > maxv) maxv = s; }
        Vec cv, c = world_com(&W, &cv);
        if (step % 15 == 0) printf("t=%.2f pelvis_y=%.3f com=(%.3f %.3f %.3f) foot_low=%.4f\n", W.t, W.b[PELVIS].x.y, c.x, c.y, c.z, world_lowest(&W, L_FOOT));
        if (fall < 0 && W.b[PELVIS].x.y < 0.6) fall = W.t;
        if (!isfinite(c.y)) { printf("NaN at t=%.2f\n", W.t); return 1; }
    }
    printf("%s: %s, max body speed %.2f m/s\n", mode, fall < 0 ? "still up after 10 s" : "fell", maxv);
    if (fall >= 0) printf("fell at t=%.2f s\n", fall);
    return 0;
}
