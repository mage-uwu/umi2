"""eval_walk.py: a walk prior with mean actions (or sampled, sigma), no assist, on the open floor, for 20 s: falls, survival,
speed. Uses the gpu/ build given first on sys.path below; for the M1 exit test build it with the imitation task's drift
stop off; or, as now, with free_heading=1, which has no drift stop (only falls end an episode).

    python3 tools/eval_walk.py prefix mref.bin [n] [sigma]"""
import sys, struct, numpy as np, torch
sys.path.insert(0, __import__('os').environ.get('GPU_DIR', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'gpu'))); import nanotss_torch as nt, gpu_env as ge
pre, mref = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 32; dev = torch.device('cpu')
pool = ge.CpuPool(n, task="walk", mref=mref, room='', seed=777, steps=600, device=dev, free_heading=1); m = nt.TSS.load(pre + ".tss", device=dev); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(n); alive = torch.ones(n, dtype=torch.bool); fell = torch.zeros(n, dtype=torch.bool); t_alive = torch.zeros(n)
x0 = None; import ctypes; SIG = float(sys.argv[4]) if len(sys.argv) > 4 else 0; torch.manual_seed(0)
with torch.no_grad():
    for t in range(pool.steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T) + (SIG * torch.randn(n, ACT) if SIG else 0)
        vz = ob[:, 11]; vx = ob[:, 9]; spd = (vx * vx + vz * vz).sqrt(); globals().setdefault('S_', []).append(float(spd[alive].mean()) if alive.any() else 0.0); globals().setdefault('F_', []).append(float(vz[alive].mean()) if alive.any() else 0.0)
        r = pool.step(act.clamp(-1, 1)); dn = r["done"].bool() & alive
        fell |= dn & r["fell"].bool(); alive &= ~dn; t_alive += alive.float()
pass
print({"episodes": n, "fell": round(float(fell.float().mean()), 2), "mean_survival_s": round(float(t_alive.mean()) / pool.HZ, 1), "episode_s": pool.steps / pool.HZ, "mean_speed_m_s": round(sum(S_)/len(S_), 2), "mean_forward_m_s": round(sum(F_)/len(F_), 2)})
