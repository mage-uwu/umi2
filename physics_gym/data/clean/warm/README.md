# clean warm starts

Policies trained only on clean data (the CMU motion capture; no Bandai Namco clips), for continuing training.

| file | what |
|---|---|
| walk_cmu.tss / .head | **superseded: do not warm-start from it.** Trained on the CMU walk cycle as converted before the rest-direction fix in tools/cmu_to_ref.py, which crossed the legs (knees 0.27 m, feet 0.47 m past each other), and before leg-on-leg contact; it learned that crossed gait. Kept only as the record of the first clean run (unassisted, 64 x 20 s, no falls, 0.96 m/s). |
| walk_cmu2.tss / .head | **the clean walk prior.** Imitation of the corrected CMU walk cycle (legs no longer crossed), with leg-on-leg contact, CMU joint ranges and muscles: 90 min stage on an A40 from scratch, adaptive assist 0.5 -> 0 (this is the checkpoint at assist 0, about 70 min in). Unassisted, mean actions, 20 s on the open floor (drift stop off): 6% falls over 64 episodes (9% over 32), 0.98 m/s; knees 7.9 cm apart on average, crossed in 4% of steps (13% for the feet) against 31% and 53% for walk_cmu. |
