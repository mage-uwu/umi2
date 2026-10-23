"""parity_env.py: GPU environments (GpuPool) against the CPU build (CpuPool) under identical actions.

Both pools start from the same seeds; each control step both get the same random actions; observations,
rewards, done flags and resets are compared. The physics is chaotic, so a last-bit difference (a libm call)
grows over an episode; the report gives the first step where anything differs at all and how far apart the
two are after that.

    python3 parity_env.py [--n 64] [--steps 600]
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env as ge

D = os.path.join(ge.BODY, "data")
CASES = {
    "balance": dict(task="balance", push=150),
    "walk": dict(task="walk", assist=0.3),
    "crawl": dict(task="crawl", style=f"{D}/cmu_ref.bin:crawl"),
    "goto": dict(task="goto"),
    "grip": dict(task="goto", grip=True),
    "bank": dict(task="goto", room=os.path.join(ge.BODY, "rooms_bank.json")),   # per-world rooms (python3 room.py --bank 32 --out rooms_bank.json)   # go-to with both hands always trying to close (the grip constraint exercised)
    "getup": dict(task="getup", style=f"{D}/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2", style_start=0.3),
}

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=64); ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--cases", default=",".join(CASES)); ap.add_argument("--room", default=os.path.join(ge.BODY, "room.json")); a = ap.parse_args()
for name in a.cases.split(","):
    kw = dict(CASES[name]); grip = kw.pop("grip", False); held = 0; room = kw.pop("room", a.room)
    g = ge.GpuPool(a.n, seed=3, room=room, **kw); c = ge.CpuPool(a.n, seed=3, room=room, **kw)
    gen = torch.Generator().manual_seed(11)
    first = None; at = {}; exact_steps = 0; worst_obs = 0.0; done_mismatch = 0; rew_diff = 0.0; episodes = 0
    o0 = (g.obs().cpu() - c.obs()).abs().max().item()
    for t in range(a.steps):
        og, oc = g.obs().cpu(), c.obs()
        dobs = (og - oc).abs().max().item(); worst_obs = max(worst_obs, dobs)
        if t in (1, 3, 10, 30, 100, 300): at[t] = dobs
        act = (torch.rand(a.n, g.ACT, generator=gen) * 2 - 1)
        if grip: act[:, -2:] = 1; held += int((oc[:, -10] > 0).sum() + (oc[:, -5] > 0).sum())   # grip inputs: holding, left and right
        rg, rc = g.step(act.cuda()), c.step(act)
        dr = (rg["rew"].cpu() - rc["rew"]).abs().max().item(); rew_diff = max(rew_diff, dr)
        dm = int((rg["done"].cpu() != rc["done"]).sum()); done_mismatch += dm; episodes += int(rc["done"].sum())
        if dobs == 0 and dr == 0 and dm == 0: exact_steps += 1
        elif first is None: first = t
    g._check()
    print(f"{name:8s} {a.n} envs x {a.steps} steps: initial obs diff {o0:.1e}; bit-identical steps {exact_steps}/{a.steps}"
          f" (first difference at step {first}); worst obs diff {worst_obs:.2e}, worst reward diff {rew_diff:.2e};"
          f" done flags differing {done_mismatch} of {episodes} episode ends" + (f"; hand-steps holding {held}" if grip else ""), flush=True)
    print("         max obs difference after step " + ", ".join(f"{k}: {v:.1e}" for k, v in sorted(at.items())), flush=True)

# the room: every environment dropped limp onto a different room box, the same state given to both builds
# (the Env layout is shared), then stepped with zero actions; the CPU's per-env physics is the scalar solver
if a.room:
    import ctypes
    nb = len(ge.load_room(a.room)); n = max(nb, 1)
    c = ge.CpuPool(n, task="getup", seed=5, room=a.room); g = ge.GpuPool(n, task="getup", seed=5, room=a.room)
    C = c.L; C.pool_drop.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; C.pool_env_copy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    ES = g.L.tss_env_size(); buf = np.zeros((n, ES), np.uint8)
    for e in range(n): C.pool_drop(c.p, e, e % nb); C.pool_env_copy(c.p, e, buf[e].ctypes.data_as(ctypes.c_void_p))
    g.envs.copy_(torch.from_numpy(buf.reshape(-1)).to(g.envs.device))
    exact = 0; first = None; worst = 0.0
    for t in range(90):
        rg, rc = g.step(torch.zeros(n, g.ACT, device="cuda")), c.step(torch.zeros(n, g.ACT))
        d = (g.obs().cpu() - c.obs()).abs().max().item(); worst = max(worst, d)
        if d == 0 and (rg["rew"].cpu() == rc["rew"]).all(): exact += 1
        elif first is None: first = t
    g._check()
    print(f"room     {n} envs dropped on the {nb} boxes x 90 steps: bit-identical steps {exact}/90 (first difference at step {first}); worst obs diff {worst:.2e}", flush=True)
