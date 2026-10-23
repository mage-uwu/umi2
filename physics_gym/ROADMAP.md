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
- **Feel, don't see.** Terrain is sensed through touch, muscle effort and (later) a handful of cheap probes; no
  height maps or rendered views in the policy's input.
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

## Now

- Go-to run 3 (development, Bandai walk prior): goal cone curriculum and a linear move reward, with the new
  muscle physics. Its job is to validate the go-to task and the realism physics, not to ship.
- M1: CMU library download and conversion; joint limits from it.
- M2: room.json v2 with grip labels, the climbing wall, ledges and the parkour lane.
