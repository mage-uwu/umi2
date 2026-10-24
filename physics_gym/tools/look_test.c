/* look_test.c: the go-to path and look inputs (env.h OBS_PATH, OBS_LOOK) for her standing at spots of the slope gym
 * (web/room_slope.json), facing a few ways, with goals across the room; prints them (tools/look_test.mjs prints the
 * same from body.js, to diff).
 *
 *   gcc -O2 -ffp-contract=off -I. -o look_test tools/look_test.c env.c body.c body_x8.c mref.c room.c -lm && ./look_test
 */
#include "env.h"
#include <stdio.h>
int main(void) {
    if (room_load("web/room_slope.json") < 0) return 1;
    const double goal[3][2] = {{4.5, 4.5}, {-5.0, 2.0}, {1.0, -5.5}}, yaw[2] = {0.3, -2.2};
    TaskCfg tc = {0}; tc.task = TASK_GOTO; tc.steps = 600; Env E; float o[OBS_DIM];
    for (int s = 0; s < 16; s++) for (int y = 0; y < 2; y++) {
        const double x = -6 + 0.8 * s, z = -5 + 0.65 * s * (s % 3 == 0 ? -1 : 1) + 5 * (s % 2);
        env_reset(&E, &tc, 1); Quat rel[NB]; for (int i = 0; i < NB; i++) rel[i] = QID;
        world_pose(&E.W, v3(x, 0.915 + support_y(0, x, z, 3), z), qexp(v3(0, yaw[y], 0)), rel, v3(0, 0, 0));
        const double *g = goal[(s + y) % 3]; env_set_goal(&E, v3(g[0], support_y(0, g[0], g[1], 3), g[1]));
        env_obs(&E, o); printf("%d %d", s, y);
        for (int k = OBS_DIM - OBS_PATH - OBS_LOOK; k < OBS_DIM; k++) printf(" %.4f", o[k]);
        printf("\n");
    }
    return 0;
}
