/* leg_test.c: the legs collide with each other. She stands in the rest pose, held up by the pelvis support, and her
 * hip muscles are driven toward crossing the legs (each thigh's target 0.35 rad toward the other side, full strength);
 * prints the left-right lateral separation of thighs, knees and feet (positive: left on the left) as they swing in.
 * With leg contacts they meet and stop (knees about two capsule radii apart); without, they pass through.
 *
 *   gcc -O2 -ffp-contract=off -I. -o leg_test tools/leg_test.c body.c -lm && ./leg_test
 */
#include "body.h"
#include <stdio.h>
int main(void) {
    World W; world_init(&W, 45, 20, 1.0 / 30); Quat rel[NB]; for (int i = 0; i < NB; i++) rel[i] = QID;
    world_pose(&W, v3(0, 1.0, 0.005), QID, rel, v3(0, 0, 0)); W.limit_k = 0;
    for (int i = 0; i < NB; i++) W.target[i] = rel[i];
    W.target[L_THIGH] = qexp(v3(0, 0, -0.35)); W.target[R_THIGH] = qexp(v3(0, 0, 0.35));   /* drive the knees inward, across */
    W.assist = 1; W.assist_x = W.b[PELVIS].x; W.assist_q = QID; W.assist_v = v3(0, 0, 0);
    for (int s = 0; s <= 30; s++) {
        if (s % 10 == 0) printf("t=%.2f knees %.3f feet %.3f thighs %.3f\n", s / 30.0, W.b[L_SHIN].x.x - W.b[R_SHIN].x.x, W.b[L_FOOT].x.x - W.b[R_FOOT].x.x, W.b[L_THIGH].x.x - W.b[R_THIGH].x.x);
        world_step(&W);
    }
    return 0;
}
