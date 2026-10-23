/* grip_test.c: the grip constraint, scripted. She hangs limp, arms straight up, under a bar, closes her hands, and hangs: with both hands, with one, with a weak grip (it slips and is
 * lost), and letting go after 1 s. Prints the pelvis height and the grip state at a few times.
 *
 *   gcc -O2 -ffp-contract=off -I. -o grip_test tools/grip_test.c body.c -lm && ./grip_test
 */
#include "body.h"
#include <stdio.h>

static void run(const char *name, int hands, double grip_max, double let_go, int limits) {
    World W; world_init(&W, 45, 20, 1.0 / 30); Quat rel[NB]; for (int i = 0; i < NB; i++) rel[i] = QID;
    rel[L_UARM] = qexp(v3(0, 0, M_PI / 2)); rel[R_UARM] = qexp(v3(0, 0, -M_PI / 2));   /* arms straight up, as in a hang */
    world_pose(&W, v3(0, 0.915 + 0.6, 0.005), QID, rel, v3(0, 0, 0)); W.kp_scale = 0.05;   /* 0.6 m up: her feet off the floor */
    for (int i = 0; i < NB; i++) W.target[i] = rel[i];
    const Vec hl = hand_point(&W, 0);   /* a bar (a handhold, 4 cm thick) just above both hands */
    const double box[ROOM_REC] = {0, hl.y + 0.021, hl.z, 0.6, 0.02, 0.02, 1, 0, 0, 0, 0.9, 0, GRIP_HOLD};
    room_set(box, 1); W.grip_max = grip_max;
    if (!limits) W.limit_k = 0;   /* without the joint ranges: arms overhead are outside the development clips' range */
    printf("%-22s", name);
    for (int s = 0; s <= 90; s++) {
        W.grip_cmd[0] = 1; W.grip_cmd[1] = hands > 1 ? 1 : 0;
        if (let_go > 0 && s >= let_go * 30) W.grip_cmd[0] = W.grip_cmd[1] = 0;
        if (s % 15 == 0) printf("  t=%.1f y=%.2f grip=%d%d f=%.0f", s / 30.0, W.b[PELVIS].x.y, W.grip_box[0] >= 0, W.grip_box[1] >= 0, W.grip_force[0] + W.grip_force[1]);
        world_step(&W);
    }
    printf("\n");
}

int main(void) {
    run("both hands", 2, GRIP_MAX, 0, 0);
    run("one hand", 1, GRIP_MAX, 0, 0);
    run("weak grip (150 N)", 2, 150, 0, 0);
    run("let go at 1 s", 2, GRIP_MAX, 1.0, 0);
    run("both, joint limits on", 2, GRIP_MAX, 0, 1);
    return 0;
}
