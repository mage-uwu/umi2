# physics-gym

A fork of `body/`: the same 14-body XPBD character, its tasks, controllers, CPU/GPU trainers and viewer, in a
room with things to climb, crawl through and fall onto.

## Stage 0: the room

`room.py` writes `room.json` (and the viewer's copy, `web/room.json`). Every feature is expanded into
static oriented boxes, and the file holds only those boxes: centre `c`, half extents `h`, rotation `q` (w, x, y, z),
friction `mu`, contact softness `soft` (XPBD compliance in m/N; 0 is rigid, foam is 3e-4) and a colour.

| feature | boxes | what |
|---|---|---|
| outer walls | 4 | 12 x 12 m room, 2.4 m high (drawn see-through) |
| stairway | 7 | 6 solid steps, 0.12 m rise, 0.30 m run, 1.4 m wide, up to a landing at 0.72 m |
| steep slope | 2 | an 18 degree ramp, 2.2 m long, then a landing |
| gentle slope | 2 | an 8 degree ramp, 2.6 m long, then a landing |
| crawl tunnel | 3 | 0.75 m clear height, 1.0 m wide, 2.0 m long |
| walk tunnel | 3 | 2.05 m clear height, 1.0 m wide, 2.2 m long |
| walls | 2 | free-standing partitions, 1.1 and 1.4 m high |
| foam | 6 | soft blocks and a mat; she sinks in about 1.5 cm, with high friction |

The spawn area (|x|, |z| < 1.8 m around the origin, where every task starts her, facing +z) is kept clear.
`python3 room.py --seed N --out f.json` writes variants with the features moved, turned and resized. Variants
that overlap, leave the room or reach into the spawn area are drawn again.

## Stage 1: climbing and parkour (room.json v2, the default)

`python3 room.py` now writes stage 1: a 16 x 16 m room (outer walls 4 m high) with all of stage 0 in the middle
and a band around it. `python3 room.py --stage 0 --out room_stage0.json` writes the stage-0 room.

| feature | boxes | what |
|---|---|---|
| climbing wall | 81 | 4.4 m wide, 3.6 m high, against the north wall; 76 handholds (small jugs 8 cm proud of the face, about 0.42 m apart, jittered) |
| mantle block, high ledge | 2 | solid blocks 1.55 m and 2.1 m high: hang from the edge and pull up |
| parkour lane | 7 | east side: a step, four platforms (0.6 to 0.9 m) across gaps of 0.5, 0.8, 1.1 and 0.7 m, a 1 m vault block, a 1.25 m drop platform |
| stepping stones | 6 | south side: low blocks 0.18 to 0.30 m high |

Every box carries a `grip` label for the hands: 0 nothing to hold, 1 a ledge (its top edges can be held), 2 a
handhold. The physics does not read it yet; it is for the grip constraint (ROADMAP.md, M3). Starts and goals are
drawn from the whole room (the outer walls less 0.6 m).

### Per-world rooms (the room bank)

`python3 room.py --bank 32 --out rooms_bank.json` writes 32 seeded variants of the stage-1 room (features moved,
turned and resized; 125 to 135 boxes each) in one file. Given a bank (`--room rooms_bank.json`), each training world
is in its own room, drawn from its episode seed at every reset, so a policy cannot memorise one layout. The physics
holds up to 32 rooms of up to 256 boxes (ROOM_BANK, ROOM_MAX); with a single room.json everything plays exactly as
before. The viewer shows one room.

### What is drawn is what is simulated

The physics in C (`body.c`), in CUDA (`gpu/tssenv_kernels.cu`, which compiles the same `body.c`) and in the browser
(`web/body.js`) all collide against the boxes in `room.json`. The viewer (`web/index.html`) draws those same boxes:
same centre, extents and rotation. Every consumer reads identical doubles, because `room.py` writes Python's
shortest round-trip decimals, which `strtod` and `JSON.parse` read back exactly.

Contacts: each corner of each body box inside a room box is pushed out through the box face it is nearest to,
using the XPBD position solve with the box's compliance. The velocity pass then removes the approach speed and
applies Coulomb friction along that face. Floor contacts are unchanged. With no room loaded, the solver gives the
same results as `body/`, bit for bit (`tools/bench_physics` checksum 662.0449413496583 in both).

The CPU's 8-wide solver (`body_x8.c`) steps worlds with the scalar solver while a room is loaded. The GPU runs
the scalar solver anyway.

### Tools

* `tools/room_drop room.json` drops her limp onto every box and reports where she rests and how deep any corner
  went: rigid boxes under 1 mm, foam 12 to 17 mm.
* `gpu/parity_env.py` (on a GPU) compares the CUDA environments with the CPU ones, including a room case: every
  environment dropped onto a different box, the same state given to both, stepped side by side.
* `gpu/ppo_gpu.py --room room.json` (the default) trains in the room. `--room ''` gives the empty floor.
* In the viewer, `umiBody.place([x, y, z], yaw)` starts her at a point of the room, `umiBody.view({ yaw, pitch, dist })`
  moves the camera, and `umiBody.room` returns the room.

### Not yet

* Heights are still measured from y = 0: the observation (pelvis height), the fall test, the foot-contact flags
  and the rewards. On a platform the flat-floor controllers see themselves "floating" and lose balance. Stage 1:
  heights relative to the support under her.
* Only body-box corners collide. A room-box edge, such as a stair nosing, can press into the middle of a long
  body face without a contact.
* One room per process (all worlds share it).
