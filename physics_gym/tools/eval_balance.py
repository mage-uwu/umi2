"""Balance on the room bank (CPU physics): python3 tools/eval_balance.py PRE ROOM.json N TOPS [SECONDS]
TOPS: share of episodes starting on the top of a room box (env.c top_start: ramps, landings, platforms, stones).
Prints the share that fell, mean time up, and how far the pelvis moved (stepping or sliding off)."""
import sys, os, struct, numpy as np, torch
sys.path.insert(0, os.environ.get('GPU_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpu'))); import nanotss_torch as nt, gpu_env as ge
pre, room, n, tops = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]); secs = float(sys.argv[5]) if len(sys.argv) > 5 else 10
dev = torch.device('cpu'); pool = ge.CpuPool(n, task="balance", room=room, seed=4242, steps=int(secs * 30), device=dev, goal_tops=tops)
m = nt.TSS.load(pre + ".tss", device=dev); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(n); alive = torch.ones(n, dtype=torch.bool); fell = torch.zeros(n, dtype=torch.bool); t_alive = torch.zeros(n)
with torch.no_grad():
    for t in range(pool.steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T)
        r = pool.step(act.clamp(-1, 1)); dn = r["done"].bool() & alive
        fell |= dn & r["fell"].bool(); alive &= ~dn; t_alive += alive.float()
print({"episodes": n, "tops": tops, "fell": round(float(fell.float().mean()), 2), "mean_up_s": round(float(t_alive.mean()) / pool.HZ, 1), "episode_s": secs})
