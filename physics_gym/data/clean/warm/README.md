# clean warm starts

Policies trained only on clean data (the CMU motion capture; no Bandai Namco clips), for continuing training.

| file | what |
|---|---|
| walk_cmu.tss / .head | imitation of the CMU walk cycle (data/clean/cmu_gait.bin), 45 min on an A40 from scratch (gpu/pod_cmu_walk.sh), the adaptive assist taken from 0.5 down to about 0.02. Mean actions, 32 episodes of 10 s: no falls at assist 0.06; 56% end early at 0.02; all fall within about 2 s at 0. Not yet an unassisted walker. |
