"""eval_goto.py: how well a go-to policy (prefix.tss + prefix.head) does in the room, with its mean actions.

Runs N environments for one episode each (600 steps, 20 s); every goal reached draws the next, as in training.
Reports goals reached per episode, the share of episodes that end in a fall and when, the speed toward the goal
while walking, and how often a goal on a raised surface (steps, a landing, foam) was reached.

    python3 eval_goto.py prefix [--n 64] [--device cuda|cpu] [--room ../room.json]
"""
import argparse, json, math, os, struct, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nanotss_torch as nt
import gpu_env as ge

ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--room", default=os.path.join(ge.BODY, "room.json"))
ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--assist", type=float, default=0); ap.add_argument("--muscles", action="store_true", help="CPU: how often each joint is at its strength"); a = ap.parse_args()
dev = torch.device(a.device)
Pool = ge.GpuPool if dev.type == "cuda" else ge.CpuPool
pool = Pool(a.n, task="goto", room=a.room, seed=a.seed, device=dev, assist=a.assist)
m = nt.TSS.load(a.prefix + ".tss", device=dev); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(a.prefix + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); assert hd == d and hact == ACT and hobs <= OBS   # a 216-input head: the new inputs unused
o = 16
def take(n):
    global o; v = np.frombuffer(raw, np.float32, n, o).copy(); o += 4 * n; return torch.tensor(v, device=dev)
Win = torch.zeros(d, OBS, device=dev); Win[:, :hobs] = take(d * hobs).view(d, hobs)
bin_, Wo, bo = take(d), take(ACT * d).view(ACT, d), take(ACT); take(ACT)
mean, sd = torch.zeros(OBS, device=dev), torch.ones(OBS, device=dev); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(a.n)
goals = torch.zeros(a.n, dtype=torch.long, device=dev); fell_at = torch.full((a.n,), -1, dtype=torch.long, device=dev)
alive = torch.ones(a.n, dtype=torch.bool, device=dev); prev_rew = None
speed_sum = torch.zeros(a.n, device=dev, dtype=torch.float64); speed_n = torch.zeros(a.n, device=dev, dtype=torch.float64)
steps = pool.steps
import re as _re
MAXT = np.array([float(v) for v in _re.search(r"MUSCLE_MAX\[NB\] = \{([^}]*)\}", open(os.path.join(ge.BODY, "muscle.h")).read()).group(1).split(",")])
with torch.no_grad():
    for t in range(steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + bin_)[:, None], state=state, stream=True)[:, 0]
        u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T)
        dist = ob[:, 219]   # the goal distance input (at most 4 m)
        r = pool.step(act.clamp(-1, 1))
        if a.muscles:
            import ctypes
            if t == 0: pool.L.pool_mtau.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]; sat = np.zeros(14); frac = np.zeros(14); cnt = 0
            buf = np.zeros(14)
            for e in range(a.n):
                if not bool(alive[e]): continue
                pool.L.pool_mtau(pool.p, e, buf.ctypes.data_as(ctypes.c_void_p)); sat += buf >= 0.98 * MAXT; frac += buf / np.maximum(MAXT, 1e-9); cnt += 1
        got = (r["rew"] > 1.9) & alive   # the arrival bonus (+2) on top of a step reward below 1
        goals += got.long()
        done = r["done"].bool() & alive
        fell_now = done & r["fell"].bool(); fell_at = torch.where(fell_now, torch.full_like(fell_at, t), fell_at)
        alive &= ~done
        # speed toward the goal: the goal input's distance shrinking, while far from it (per step, at 30 Hz)
        if t > 0:
            dd = (prev_d - dist) * pool.HZ; far = (prev_d > 0.6) & (dist < prev_d + 1.0) & alive
            speed_sum += torch.where(far, dd.double(), torch.zeros_like(dd.double())); speed_n += far.double()
        prev_d = dist
g = goals.cpu().numpy(); f = fell_at.cpu().numpy()
if a.muscles:
    names = ["pelvis", "spine", "chest", "head", "l_uarm", "l_farm", "r_uarm", "r_farm", "l_thigh", "l_shin", "l_foot", "r_thigh", "r_shin", "r_foot"]
    print("at strength (share of steps), mean torque / strength:", {names[i]: (round(float(sat[i] / cnt), 2), round(float(frac[i] / cnt), 2)) for i in range(1, 14)})
res = {"policy": os.path.basename(a.prefix), "assist": a.assist, "episodes": a.n, "goals_per_episode": round(float(g.mean()), 2),
       "episodes_reaching_a_goal": round(float((g > 0).mean()), 2), "fall_rate": round(float((f >= 0).mean()), 2),
       "mean_fall_time_s": round(float(f[f >= 0].mean() / pool.HZ), 1) if (f >= 0).any() else None,
       "speed_toward_goal_m_s": round(float(speed_sum.sum() / max(speed_n.sum(), 1)), 2)}
print(json.dumps(res))
