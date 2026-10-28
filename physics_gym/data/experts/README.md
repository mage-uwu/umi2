# Rooftops move experts

Why: one ternary policy over all 57 parkour clips, with a hand-up curriculum, stalled at 2% of clips unaided over
three runs (about 1.3 billion steps). One clip alone, on a plain float MLP, learned a climb in 30 minutes
(`data/clean/warm/climb_p4_mlp`). But it was brittle. It trained with no start noise and one fixed room, so it failed
with the block 20 cm higher, or with a building 0.6 m behind the block, and its way back down failed most of the
time. So: one expert per Rooftops move, trained from the start with what broke that one.

## What is here

| File | What |
|---|---|
| `rooftops.json` | The manifest: each expert's clip, its variation, and the Rooftops obstacles it is judged on |
| `<name>_train.json`, `<name>_heldout.json` | 48 training and 16 held-out variants of the clip's room (`tools/expert_bank.py`, seeded) |
| `../../tools/expert_bank.py` | Builds the banks: heights, gaps, distances, wall depths, and unrelated boxes off her path |
| `../../gpu/pod_experts.sh` | The pod job: 4 experts side by side on one GPU, in waves; snapshots; deadline watchdog |
| `../../tools/expert_eval.py` | Judges the experts: held-out variants, and Rooftops itself (`clip_in_world.py --align --t0 auto`) |

| Expert | Rooftops move | Clip | Its obstacle | Trained on | Judged in Rooftops on |
|---|---|---|---|---|---|
| step_up | step 0.4–0.5 | p38 (PARC platform) | 0.41 m | 0.30–0.52 m | garden crate 0.5 from the street; courtyard step 0.4 from the landing |
| hoist | hoist 0.5–0.7 | p44 (PARC platform) | 0.57 m | 0.43–0.71 m | market crate (0.5 over the market roof); bell tower (0.6 over the central roof) |
| vault_low | vault 0.5 | p10 (reverse vault) | 0.50 m | 0.40–0.60 m, 0.25–0.9 m deep | alley wall 0.5 |
| vault_mid | vault 0.8 | p13 (speed vault) | 0.70 m | 0.66–0.85 m, 0.25–0.9 m deep | alley wall 0.8 |
| vault_high | vault 1.0 | p15 (speed vault) | 0.93 m | 0.88–1.05 m, 0.25–0.9 m deep | alley wall 1.0 (from the east) |
| gap_jump | jump 0.9 | fjump2 (CMU) | 0.9 m gap | gaps ±0.15 m, platforms 0.3–1.2 m high | west roof → west roof across the gap |
| leap | leap 1.4 | run_leap1 (CMU) | 1.48 m gap | gaps ±0.15 m, landing up to 0.25 m lower | west roof across the gap → leap roof |
| climb_block | step and block | p4 | 1.81 m | ±0.12 m, ±0.2 m | (the block put into Rooftops: `clip_in_world.py --own`) |

Training, per expert:
- a float MLP (3×512, ELU) from scratch;
- start noise of 0.03 rad;
- unaided, at x2 strength;
- half the episodes from the clip's start, half mid-clip;
- the variants drawn by how often she fails each (`--clip-focus 0.5`);
- 40 minutes, about 100M steps at about 45k steps/s with 4 experts sharing one A100.

**Baseline to beat** (the current climber on climb_block's held-out variants, 64 episodes, 0.02 rad start noise): on
top 57/64, whole clip 7/64.

## Not covered (no motion data for it)

- drops of 1.0–1.2 m;
- the inclines;
- the bare 1.2 m walls (west roof and courtyard step from the street: her climb stands on a 0.54 m step);
- the leap of faith.

The route planner (`web/planner.js`) can route around the walls. The rest needs clips.

## Launch (when funded)

Commit and push first. The pod checks out the commit given.

One A100 SXM, Community Cloud $1.39/hr or Secure $1.79/hr. Create the pod with this body (RunPod `create-pod`), at the
pushed commit:

```json
{"name": "physics-gym-experts-a100", "cloud": "COMMUNITY", "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
 "gpu": {"id": "NVIDIA A100-SXM4-80GB", "count": 1, "minCudaVersion": "12.8"}, "disk": 20, "ports": ["8000/http"],
 "entrypoint": ["/bin/bash", "-c"],
 "cmd": ["mkdir -p /w/out; (python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &); { which gcc || (apt-get update && apt-get install -y gcc); git clone https://github.com/mage-uwu/umi2 /w/repo && cd /w/repo && git checkout COMMIT && git log --oneline -1; } > /w/out/setup.log 2>&1; cd /w/repo && DEADLINE=6000 bash physics_gym/gpu/pod_experts.sh >> /w/out/setup.log 2>&1; sleep infinity"]}
```

| Run | Pod time | Community ($1.39/hr) | Secure ($1.79/hr) |
|---|---|---|---|
| All 8 experts, 40 min each, 4 at a time (2 waves) | ~1 h 30 min (watchdog at 1 h 40 min) | ~$2.10 (max $2.32) | ~$2.70 (max $2.98) |
| 4 experts (`EXPERTS=step_up,vault_low,gap_jump,climb_block`) | ~47 min | ~$1.10 | ~$1.40 |
| All 8, 80 min each (`SECS=4800 DEADLINE=10200`) | ~2 h 50 min | ~$3.90 | ~$5.00 |

The pod serves `/w/out` on port 8000:
- `<name>.mlp.pt` / `.head`, and `<name>_s<n>` snapshots every 30 min;
- `<name>.log` (`fell_share`, and `focus`: the hardest variants);
- `stages.txt`, `parity.txt` (GPU/CPU match, including the start-noise reset), then `DONE`.

Pull everything before the watchdog's deadline (a stopped pod loses its disk), then terminate.

## After

```
python3 tools/expert_eval.py PULLED_DIR --n 64 --replays web/replays
```

This gives the table (`PULLED_DIR/report.md`), and one replay per Rooftops placement. The viewer shows those as
buttons via `web/replays/index.json`. An expert is ready for the course when it reaches its obstacle in most of its
held-out variants and in Rooftops.

Then phase 2, chaining:
- the planner (`web/planner.js`) picks each move;
- the expert for it starts from her state near its takeoff (`EVAL_T0` / `fixed_start` start an expert mid-clip;
  experts train half their episodes mid-clip);
- later, the experts distilled into one policy that sees the next move.
