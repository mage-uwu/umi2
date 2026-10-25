/* world_demo.c: the physics world on its own, no controller. physics_world.c embeds this as its main
 * (cc -DPHYSICS_WORLD_DEMO); built against the separate sources it prints the same numbers to the last digit:
 *
 *   cc -O2 -ffp-contract=off -I. -o world_demo tools/world_demo.c body.c room.c -lm && ./world_demo [room.json]
 *
 * She is dropped limp from 1.2 m above the spot (0, 0) (onto the floor, or whatever the room has there), lies
 * still, then her muscles tighten and hold the rest pose for a second (unbalanced: nothing steers her, so she
 * topples slowly). Every 0.25 s: the time, her pelvis height, the height of the surface under it, and the
 * contact force on the feet and on the pelvis as a share of her weight. */
#ifndef PHYSICS_WORLD_DEMO_EMBEDDED
#include "body.h"
#include <stdio.h>
#endif
int main(int argc, char **argv) {
    if (argc > 1 && room_load(argv[1]) < 0) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    World W; world_init(&W, 45, 20, 1.0 / 30);   /* 45 kg, 20 substeps per 1/30 s control step */
    Quat rel[NB]; for (int i = 0; i < NB; i++) rel[i] = QID;   /* the rest pose: standing, arms out (a T-pose) */
    const double top = support_y(0, 0, 0, 10);
    world_pose(&W, v3(0, top + 1.2 + BODY_DEFS[PELVIS].center.y, 0), qexp(v3(1.1, 0.3, 0)), rel, v3(0.5, 0, 0));
    for (int i = 0; i < NB; i++) W.target[i] = rel[i];
    const double wt = W.total_mass * 9.81;
    printf("    t  pelvis  under  feet  pelvis-touch\n");
    for (int s = 0; s <= 120; s++) {
        W.kp_scale = s < 75 ? 0.03 : 1.0;   /* limp for 2.5 s, then the muscles hold the rest pose */
        if (s % 8 == 0) {
            const Vec p = W.b[PELVIS].x;
            printf("%5.2f  %6.3f  %5.2f  %4.2f  %4.2f\n", W.t, p.y, support_y(W.room, p.x, p.z, p.y), (W.touch[L_FOOT] + W.touch[R_FOOT]) / wt, W.touch[PELVIS] / wt);
        }
        world_step(&W);
    }
    return 0;
}
