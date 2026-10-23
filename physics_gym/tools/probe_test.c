/* probe_test.c: the go-to probes (env.h OBS_PROBE) for her standing at a few spots of room.json, facing a few ways;
 * prints them (tools/probe_test.mjs prints the same from body.js, to diff).
 *
 *   gcc -O2 -ffp-contract=off -I. -o probe_test tools/probe_test.c env.c body.c body_x8.c mref.c room.c -lm && ./probe_test
 */
#include "env.h"
#include <stdio.h>
int main(void) {
    if (room_load("room.json") < 0) return 1;
    const double spot[8][2] = {{-3.6, 1.6}, {3.6, 1.6}, {0, -1.8}, {-1.4, 3.6}, {2.0, -0.4}, {-2.3, -1.8}, {5.5, -0.6}, {0, 6.5}};
    const double yaw[3] = {0, 0.6, -2.5};
    TaskCfg tc = {0}; tc.task = TASK_GOTO; tc.steps = 600; Env E; float o[OBS_DIM];
    for (int s = 0; s < 8; s++) for (int y = 0; y < 3; y++) {
        env_reset(&E, &tc, 1); Quat rel[NB]; for (int i = 0; i < NB; i++) rel[i] = QID;
        world_pose(&E.W, v3(spot[s][0], 0.915 + support_y(0, spot[s][0], spot[s][1], 3), spot[s][1]), qexp(v3(0, yaw[y], 0)), rel, v3(0, 0, 0));
        env_obs(&E, o); printf("%d %d", s, y);
        for (int k = OBS_DIM - OBS_PROBE; k < OBS_DIM; k++) printf(" %.4f", o[k]);
        printf("\n");
    }
    return 0;
}
