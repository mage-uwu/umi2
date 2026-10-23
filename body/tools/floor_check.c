/* floor_check.c: the floor tasks' start states and what doing nothing does from them.
 *   floor_check crawl|getup [episodes] */
#include "../env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    TaskCfg tc = {.task = !strcmp(argv[1], "crawl") ? TASK_CRAWL : TASK_GETUP, .steps = 60, .act_mul = 2.5};
    const int n = argc > 2 ? atoi(argv[2]) : 4; float a[ACT_DIM] = {0};
    for (int e = 0; e < n; e++) {
        Env E; env_reset(&E, &tc, 11 + e);
        printf("start %d: pelvis %.2f head %.2f  lowest:", e, E.W.b[PELVIS].x.y, E.W.b[HEAD].x.y);
        for (int i = 0; i < NB; i++) printf(" %.2f", world_lowest(&E.W, i));
        double R = 0; while (!E.done) R += env_step(&E, a);
        printf("\n   after %d steps of zero action: pelvis %.2f head %.2f chest-front.y %.2f  return %.1f\n", E.step, E.W.b[PELVIS].x.y, E.W.b[HEAD].x.y, qrot(E.W.b[CHEST].q, v3(0, 0, 1)).y, R);
    }
    return 0;
}
