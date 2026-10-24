"""Getting up (CPU physics, the gym's): python3 tools/eval_getup.py PRE N [SECONDS]
Prints the share that stood up (pelvis 0.8 m and head 1.3 m above the floor, both feet down) within the time, the
mean time to stand, and two twitch measures: the mean change of the actions per control step and the mean of the
largest joint angular speed (rad/s) while getting up (a person's get-up: smooth, under about 6 rad/s)."""
import sys, os, struct, numpy as np, torch
sys.path.insert(0, os.environ.get('GPU_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpu'))); import nanotss_torch as nt, gpu_env as ge
pre, n = sys.argv[1], int(sys.argv[2]); secs = float(sys.argv[3]) if len(sys.argv) > 3 else 8
dev = torch.device('cpu'); pool = ge.CpuPool(n, task="getup", room='', seed=99, steps=int(secs * 30), device=dev)
m = nt.TSS.load(pre + ".tss", device=dev); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(n); alive = torch.ones(n, dtype=torch.bool); stood = torch.zeros(n, dtype=torch.bool); t_stand = torch.zeros(n)
prev = None; dact, wmax, cnt = 0.0, 0.0, 0
with torch.no_grad():
    for t in range(pool.steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        live = alive & ~stood
        if live.any():
            if prev is not None: dact += float((act - prev).abs().mean(1)[live].mean()); cnt += 1
            w = ob[:, :15 * 14].view(n, 14, 15)[:, :, 12:15] * 5   # angular velocities (obs scales them by 0.2)
            wmax += float(w.norm(dim=2).max(1).values[live].mean())
        prev = act
        # standing: pelvis height above the lowest foot (obs[1]) and the head above the pelvis
        up = (ob[:, 1] > 0.8) & (ob[:, 3 * 15 + 1] > 0.45) & (ob[:, 14 * 15] > 0) & (ob[:, 14 * 15 + 1] > 0)
        newly = up & ~stood & alive; t_stand[newly] = t / pool.HZ; stood |= newly
        r = pool.step(act); alive &= ~r["done"].bool()
print({"episodes": n, "stood_up": round(float(stood.float().mean()), 2), "mean_time_s": round(float(t_stand[stood].mean()), 1) if stood.any() else None,
       "action_change_per_step": round(dact / max(cnt, 1), 3), "peak_joint_speed_rad_s": round(wmax / max(cnt, 1), 1)})
