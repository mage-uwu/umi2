# clean warm starts

Policies trained only on clean data (the CMU motion capture; no Bandai Namco clips), for continuing training.

| file | what |
|---|---|
| walk_cmu.tss / .head | imitation of the CMU walk cycle (data/clean/cmu_gait.bin): 45 min on an A40 from scratch with the adaptive assist (0.5 down to about 0.02), then 60 min more from 0.06 down to 0 (gpu/pod_cmu_walk.sh, WALK_LOAD). Unassisted, mean actions, 64 episodes of 20 s on the open floor: **no falls**, 0.96 m/s (0.94 forward; the clip walks at 0.98). The M1 walk test passes. Training's survival figure (about 40%) counted the drift stop (pelvis 0.6 m from the clip's path), not falls. |
