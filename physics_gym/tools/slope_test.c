/* slope_test.c: does she hold her footing on the room's ramps? She is stood in the rest pose (muscles holding it, no
 * controller) on the steep slope, the gentle slope, a flat platform and the open floor of room.json, and the pelvis
 * drift over 3 s is printed (horizontal, m); then the same lying limp, which tests friction alone: with mu 0.9 an
 * 18 degree ramp must hold a lying body still (tan 18 deg = 0.32).
 *
 *   gcc -O2 -ffp-contract=off -I. -o slope_test tools/slope_test.c body.c room.c -lm && ./slope_test
 */
#include "body.h"
#include <stdio.h>
static void run(const char *name, double x, double z, double yaw, int lying) {
    World W; world_init(&W, 45, 20, 1.0 / 30); Quat rel[NB]; for (int i = 0; i < NB; i++) rel[i] = QID;
    const double y0 = support_y(0, x, z, 3.0);
    if (lying) {   /* limp, lying face up across the slope (yaw), so no part starts inside it: a pure friction test */
        world_pose(&W, v3(x, y0 + 0.25, z), qmul(qexp(v3(0, yaw, 0)), qexp(v3(-M_PI / 2, 0, 0))), rel, v3(0, 0, 0)); W.kp_scale = 0.02;
    } else world_pose(&W, v3(x, y0 + 0.915 + 0.01, z), qexp(v3(0, yaw, 0)), rel, v3(0, 0, 0));
    for (int i = 0; i < NB; i++) W.target[i] = rel[i];
    const Vec p0 = W.b[PELVIS].x; printf("%-20s ground %.3f", name, y0);
    for (int s = 1; s <= 90; s++) {
        world_step(&W);
        if (s % 30 == 0) { const Vec d = vsub(W.b[PELVIS].x, p0); printf("  t=%ds drift %.3f (y %.3f)", s / 30, sqrt(d.x * d.x + d.z * d.z), W.b[PELVIS].x.y); }
    }
    printf("\n");
}
int main(void) {
    if (room_load("room.json") < 0) return 1;
    for (int ly = 0; ly < 2; ly++) {
        printf(ly ? "lying limp (friction alone):\n" : "standing in the rest pose:\n");
        run("steep slope (18 deg)", 3.6, 3.4, ly ? M_PI / 2 : 0, ly);   /* lying: across the slope */
        run("gentle slope (8 deg)", -4.4, -5.15, ly ? 0 : M_PI / 2, ly);
        run("flat platform", 3.6, 5.0, 0, ly);
        run("open floor", 0, 0, 0, ly);
    }
    return 0;
}
