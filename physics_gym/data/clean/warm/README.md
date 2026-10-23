# clean warm starts

Policies trained only on clean data (the CMU motion capture; no Bandai Namco clips), for continuing training.

| file | what |
|---|---|
| walk_cmu.tss / .head | **superseded: do not warm-start from it.** Trained on the CMU walk cycle as converted before the rest-direction fix in tools/cmu_to_ref.py, which crossed the legs (knees 0.27 m, feet 0.47 m past each other), and before leg-on-leg contact; it learned that crossed gait. Kept only as the record of the first clean run (unassisted, 64 x 20 s, no falls, 0.96 m/s). |
