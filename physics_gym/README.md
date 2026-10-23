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
