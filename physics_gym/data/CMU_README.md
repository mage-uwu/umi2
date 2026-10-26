# cmu_ref.bin

Reference clips for the floor skills' motion style prior (env.c, `TaskCfg.style`), converted with
`tools/cmu_to_ref.py` from the CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu/
(created with funding from NSF EIA-0196217).

| clip | source | what |
|---|---|---|
| crawl | subject 111, trial 03, frames 516-1188 (4.3-9.9 s) | crawling on hands and knees |
| getup_face1, getup_face2 | subject 140, trials 01, 02 | getting up from lying face down |
| getup_side1, getup_side2 | subject 140, trials 03, 04 | getting up from lying on the side |
| getup_back1, getup_back2 | subject 140, trials 08, 09 | getting up from lying on the back |

Terms, as stated on mocap.cs.cmu.edu: "This data is free for use in research projects. You may include this
data in commercially-sold products, but you may not resell this data directly, even in converted form."

# cmu_jump.bin

Jumps for the jump courses (data/jump_bank.json, tools/jump_course.mjs; gpu/pod_jump.sh), converted the same way at 30 Hz.

| clip | source | what |
|---|---|---|
| run_leap1, run_leap2 | subject 49, trials 04, 05 | run, leap (a dancer's running leap; 1.7 and 1.6 m from takeoff to landing) |
| fjump1 | subject 16, trial 07 | standing forward jump (1.1 m) |
| fjump2 | subject 13, trial 13 | standing forward jump (1.1 m) |
| run_jump1, run_jump2, run_jump3 | subject 75, trials 02, 10, 01 | run up and jump with both feet (2.0, 1.5 m; run_jump3 unused) |
| box_jump | subject 75, trial 12 | jump up onto a box (unused: no box under it yet) |
| run1, run2 | subject 35, trial 17; subject 16, trial 35 | run/jog (unused) |

# cmu_loco.bin

The locomotion reference for the learned style reward (gpu/ppo_gpu.py --amp; task goto_free), written by
`tools/amp_set.py` from the 30 Hz CMU library (data/clean/cmu30_library, tools/cmu_library.py --fps 30): 39 clips,
6215 frames (3.5 min) of plain walking, running and jogging, starting and stopping, turning, walking sideways and
backwards (subjects 2, 7, 8, 9, 35, 69, 77, 78, 83, 91, 104, 105, 141). Stylised walks, props, seats, stairs, uneven
ground, jumps and two-person trials are left out; frames the capture lost are dropped. Each clip is named
c<subject>_<trial>. Same terms as above.

# cmu_climb.bin

Climbs for the climbing courses (data/climb_bank.json, tools/climb_course.mjs; gpu/pod_climb.sh), fetched at 30 Hz with
`tools/cmu_fetch.py` (frame ranges at 30 Hz):

| clip | source | what |
|---|---|---|
| c13_35_30, c13_36_60 | subject 13, trials 35 (frames 30-200), 36 (60-240) | climb three steps (0.19, 0.36, 0.56 m) |
| c40_06_30 | subject 40, trial 06 (30-105) | step up a stepstool (0.19, 0.36 m) |
| c40_06_930, c40_06_1005 | subject 40, trial 06 (930-1000, 1005-1075) | standing jump over the stool, there and back |
| c81_14r, c81_15r | subject 81, trials 14 (255-510), 15 (255-507), played backwards | hoisting up a 0.6 m ledge with the hands (the capture: climbing backwards off it) |
| c88_02_15 | subject 88, trial 02 (15-66) | standing jump up onto a 0.7 m platform |
