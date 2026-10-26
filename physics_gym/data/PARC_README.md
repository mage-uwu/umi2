# parc_train.bin, parc_heldout.bin (and their _bank.json course rooms)

Parkour motions from **PARC** (Michael Xu, Yi Shi, KangKang Yin, Xue Bin Peng: "PARC: Physics-based Augmentation with
Reinforcement Learning for Character Controllers", SIGGRAPH 2025), retargeted to the character with `tools/parc_to_ref.py`.
Source: huggingface.co/datasets/mxucg/PARC (`dec_release.zip`, `initial_aug/*/dm/`: one augmented take of each of the 137
source motions), released under the MIT licence; code github.com/mshoe/PARC, BSD-3-Clause (c) Xue Bin Peng. The paper credits
Beyond Capture Studios for the parkour capture; about 5.5 s of PARC's original data come from Epic Games' Game Animation
Sample Project, whose own terms may apply to those clips.

Each clip is one motion on the terrain it was made for (a 0.4 m heightfield, here its columns near the path as boxes):

| file | clips | what |
|---|---|---|
| parc_train | 64 | all 6 wall climbs (1.8 to 2.1 m walls, hands on the edge), all 11 vaults (speed, dash, reverse), 30 platform and 17 running clips |
| parc_heldout | 64 | 35 platform and 29 running clips not in the training set: motions and terrains the tracker never sees |

Clips are named p<n>; each room records its source file and PARC's contact labels (feet, hands) per frame.

Each clip's root is stored relative to the top of the column it starts on (course mode raises it onto that column). The
first conversion subtracted the column's height above the terrain's lowest column instead, which differs where a terrain's
floor is not at 0 (the running clips, 0.1 to 0.3 m too low); fixed in 372b0d8 and the files regenerated.

With a full hand-up toward each clip's root (tools/eval_course.py, jump_stick, 192 episodes from each clip's start), 124 of
192 trained-on episodes end where their clip does: wall climbs 17 of 19, platforms 72 of 91, vaults 15 of 32, runs 20 of 50
(the vaults and runs outpace the hand-up; the geometry matches). The viewer's PARC skill plays them from
web/parc_clips.json and web/parc_bank.json (tools/parc_web.py: both sets, the held-out clips renamed h<n>).
