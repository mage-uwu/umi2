# physics-gym roadmap

**North star.** A general neural rig: one physically simulated character driven by one learned policy that moves
through arbitrary geometry with Assassin's Creed fluidity (walk, run, jump, vault, climb edges and handholds, fall
and recover), commanded in real time by a game (go there, at this pace, with this style), and cheap enough to run
many NPCs at once. The gym trains it; the browser viewer is its proving ground.

Principles that hold at every stage:

- **One physics, everywhere.** body.c is the only implementation; the CUDA build is bit-identical to it, and body.js
  is a line-for-line twin. What the viewer draws is what is simulated (room.json boxes, nothing else).
- **Realism comes from the body, not the reward.** Joint ranges, muscle strength, how fast force rises and an energy
  cost are physics and observation; rewards say what to do, not how to look.
- **Feel, and glance.** Terrain is sensed through touch and muscle effort, plus 16 cheap probes (the ground at 6 points
  ahead, 10 short rays at knee and hip height, from the same room queries the physics uses); no height maps or
  rendered views. (Touch alone was not enough: in the cluttered room a blind walker trips over boxes it only learns
  about on contact, falling in 98% of episodes.)
- **Clean data for the true run.** Everything that ships is trained only on mocap we may use in products.
- **Short GPU runs with a parity check first.** Every pod job re-proves CPU/GPU bit-identity before training.

## Data

| source | licence | use | status |
|---|---|---|---|
| CMU Graphics Lab Motion Capture Database (2,435 trials: walk 617, run 156, jump 174, hop 42, dance 80, climb 36, cartwheel 14, flip 18, ...) | free for research; may ship in commercial products; may not be resold, even converted | **primary**: joint limits, muscle envelopes, skill clips | `tools/cmu_library.py` packs all of it (data/clean/, not in git) |
| 100STYLE (Mason et al., Zenodo 8127870, BVH) | CC BY 4.0 | locomotion styles (100 walk/run/idle styles) | planned: BVH converter |
| HDM05 (MPI) | CC BY-SA 3.0 | extra jumps, cartwheels, stairs | optional (share-alike) |
| Bandai Namco Research Motion Dataset | CC BY-NC 4.0 | **development only** (the current walk prior and the viewer's clips) | to be replaced; never in the true run |

The true run retrains the walk prior and every later skill from the clean set. Policies warm-started from the
Bandai walk (data/warm/walk and anything trained from it) are development checkpoints, not shippable.

## Milestones

Each has an exit test; the next starts only when it passes.

**M0: the body (done).** XPBD physics, 14 bodies, room contacts, touch and muscle sense, mocap joint limits,
muscle strength and slew, metabolic cost; GPU twin bit-identical; viewer renders the room.

**M1: clean foundations.**
- Joint limits and muscle envelopes recomputed from the whole CMU library (dance, acrobatics, climbing: the full
  range people actually use), mirrored.
- Walk, run and turn priors re-trained from CMU clips only (imitation, then the assist curriculum).
- Exit: the CMU walk prior walks 20 s in the open with under 10 % falls under the muscle limits, and go-to reaches
  goals ahead of her.

**M2: the gym, stage 1 (geometry).** room.json v2: grip labels on boxes (ledge: its top edges can be held; hold: a
labelled handhold), a climbing wall with holds, ledges at hang height, a parkour lane (gaps, vault blocks, jump
platforms, drops). Procedural variants per environment (each GPU world draws its own room from a seed) so nothing
is memorised. Exit: every feature in the physics matches the render; per-world rooms bit-identical CPU/GPU.

**M3: grip.** A hand can close on a hold or a ledge edge: a grip action per hand; a grip constraint (a ball joint
between hand and the touched point) while the hand is on a grippable surface and the action is held; it lets go
when the needed force exceeds grip strength (from published values, scaled like the muscles). Hand observations:
grip state and the nearest grippable point relative to each hand (two cheap probes, nothing rendered). Exit:
scripted tests (hang from a ledge, the grip breaking under load) in C, CUDA and JS, bit-identical.

**M4: skills as motion states.** One policy, conditioned on a skill command (walk, run, jump, climb, and later
vault, hang, drop) plus the goal, with each skill's style prior from its CMU clips. Curriculum per skill in the gym:
jump a gap, climb a few holds, mantle a ledge. A small high-level selector (or the game) switches commands;
transitions are trained by switching commands mid-episode. Exit: each skill passes its own gym test; switching
between any two works more than 90 % of the time.

**M5: the NPC engine.** Navigate the whole room to any goal choosing skills itself (learned selector over the M4
commands); real-time in the browser (WASM, many characters at once); robustness to pushes and unseen room seeds.
Exit: unseen procedural rooms, goals anywhere, more than 80 % reached without falling.

## Status

- **Go-to diagnosis.** Three problems behind go-to collapsing without assist, and their fixes: (1) a warm-start bug:
  inputs the walk never saw vary (goal, grip) had untrained weights and fed saturated values in go-to (open-floor
  falls 98% -> 4% once zeroed; ppo_gpu --fresh-inputs); (2) the walk falls in 98% of 20 s episodes under its own
  exploration noise, because the 0.6 m drift stop ended noisy episodes before they became falls (fix: free-heading
  imitation, ppo_gpu --free-heading 1, where only falls end episodes, plus pushes); (3) blind in clutter (fix: probes).

- **Fixed: CMU retargeting crossed the legs.** tools/cmu_to_ref.py copied each CMU bone's global rotation onto her
  body, but the two skeletons' rest poses point the bones differently (CMU's femurs slant 20 degrees outward, hers
  hang straight; the feet differ by 35 degrees in pitch), so every converted pose crossed the legs (in walks the knees
  0.27 m, the feet 0.47 m past each other). Each body now takes a fixed rest-direction alignment from the subject's
  ASF; the whole library was corrected in place (cmu_library.py --realign), and the joint limits, muscle envelopes
  (hip strength 188 -> 125 N m: the crossed legs had inflated it), gait cycles and get-up and crawl clips (also 0.9 m
  too high) regenerated. The first clean walk prior learned the crossed gait and is superseded.
- **Leg-on-leg contact.** The legs were not colliding with each other at all; each leg body is now a capsule and the
  two legs push apart where they meet (thighs, shins, feet, shin against foot), in C, CUDA and JS
  (tools/leg_test.c and .mjs: driven to cross, the knees stop 0.09 m apart; without contact they pass 0.24 m through).

- M1: the CMU library is packed (2,431 trials, 348,504 poses at 10 Hz, about 9.7 h). **Joint limits now come from
  it** (joint_limits.h, web/joint_limits.js): per-trial ranges with robust statistics over trials, joined with the
  CMU range-of-motion trials, after anatomical sanity caps drop capture errors (feet flipped half a turn in about
  4 % of trials). Shoulders reach 87 degrees above the chest's horizontal and hips abduct to 132 (splits): an
  overhead hang is inside the range, which the development clips did not allow. Still to do: muscle envelopes from
  CMU (inverse dynamics needs 30 Hz clips: a second, smaller conversion), and the CMU walk/run priors.
- M2: room.json v2 with grip labels; the stage-1 room (climbing wall with 76 holds, mantle block, high ledge,
  parkour lane, stepping stones). Per-world rooms: a bank of 32 seeded variants, each world in its own (C, CUDA; one room plays bit-identically to before). Still to do: a spatial grid for the broadphase (the
  stage-1 room's 125 boxes cost about 25 % of CPU step time).
- M3: the grip constraint is in C, CUDA and JS (tools/grip_test.c and .mjs agree on every grip state and force; the
  JS twin drifts in the last bits once she tumbles). Still to do: GPU parity on the next pod.
- Go-to run 3 (development, Bandai walk prior), stopped early: at assist 0.15 she walks 13 s episodes and reaches
  goals ahead of her; every fixed step-down of the assist (0.15 -> 0.05 -> 0) collapsed to falls within 1-2 s under
  the muscle limits. So the curriculum now adapts (ppo_gpu --assist-auto): the assist comes down only while she
  lasts most of each episode, and goes back up when she falls.
- M1 muscles: strength and slew from inverse dynamics of 557 CMU trials at 30 Hz. The CMU gait cycles
  (tools/cmu_cycle.py: walk 16_16, 1.17 s at 0.98 m/s; run 35_20, 0.73 s at 3.1 m/s) are the clean prior's
  reference; gpu/pod_cmu_walk.sh trains it (imitation, then go-to) with no Bandai data.
