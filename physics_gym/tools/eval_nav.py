"""eval_nav.py: does a go-to policy follow the path helper or head straight for the goal, and does it slow at edges?
    python3 tools/eval_nav.py PREFIX [N]   (N episodes of 30 s over rooms_nav.json, the 5% hand-up of the viewer)
On steps where the path's waypoint and the goal lie in different directions (more than 60 deg apart: a detour, round
an edge to a ramp, a wall), compares her velocity with each. Also: how fast she meets drops and rises ahead."""
import sys, os, struct, math, numpy as np, torch
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(B, 'gpu')); import nanotss_torch as nt, gpu_env as ge
pre = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 64

pool = ge.CpuPool(n, task="goto", mref=B + "/data/clean/cmu_gait.bin", room=B + "/rooms_nav.json", seed=777, device="cpu", assist=0.05, goal_tops=0.5, steps=900)
m = nt.TSS.load(pre + ".tss", device='cpu'); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(n); alive = torch.ones(n, dtype=torch.bool)
cw, cg, nd, sp_all, sp_det, falls, goals = [], [], 0, [], [], 0, 0
drop_fast = 0; drop_steps = 0; rise_fast = 0; rise_steps = 0
with torch.no_grad():
    for t in range(900):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        v = ob[:, [9, 11]]; sp = v.norm(dim=1); g = ob[:, [216, 218]]; w = ob[:, [273, 275]]; plen = ob[:, 276]
        gn, wn = g.norm(dim=1), w.norm(dim=1)
        cosgw = (g * w).sum(1) / (gn * wn + 1e-9)
        det = alive & (cosgw < 0.5) & (plen > 1.0) & (wn > 0.2) & (sp > 0.3)
        for e in torch.nonzero(det).flatten().tolist():
            cw.append(float((v[e] * w[e]).sum() / (sp[e] * wn[e]))); cg.append(float((v[e] * g[e]).sum() / (sp[e] * gn[e]))); sp_det.append(float(sp[e]))
        sp_all += sp[alive & (ob[:, 219] > 0.6)].tolist()
        pr = ob[:, 257:259]   # the ground 0.3 and 0.6 m ahead, from her lowest foot
        dr = alive & (pr.min(1).values < -0.3); drop_steps += int(dr.sum()); drop_fast += int((dr & (sp > 1.2)).sum())
        rs = alive & (pr.max(1).values > 0.3); rise_steps += int(rs.sum()); rise_fast += int((rs & (sp > 1.2)).sum())
        r = pool.step(act); goals += int(((r["rew"] > 1.9) & alive).sum())
        dn = r["done"].bool() & alive; falls += int((dn & r["fell"].bool()).sum()); alive &= ~dn
cw, cg = np.array(cw), np.array(cg)
print(f"{pre}: {n} episodes of 30 s: goals {goals}, falls {falls}; speed while going {np.mean(sp_all):.2f} m/s (90th pct {np.percentile(sp_all, 90):.2f})")
print(f"detour steps {len(cw)}: velocity toward the waypoint cos {cw.mean():.2f}, toward the goal {cg.mean():.2f}; closer to the waypoint's direction in {np.mean(cw > cg) * 100:.0f}%; speed {np.mean(sp_det):.2f}")
print(f"a drop of more than 0.3 m within 0.6 m ahead: {drop_steps} steps, {100 * drop_fast / max(drop_steps, 1):.0f}% of them at over 1.2 m/s; a rise: {rise_steps} steps, {100 * rise_fast / max(rise_steps, 1):.0f}% at over 1.2 m/s")
