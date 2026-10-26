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
