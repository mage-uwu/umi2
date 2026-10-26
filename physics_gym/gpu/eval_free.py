"""eval_free.py: a goto_free policy on its own, mean actions, no hand-up unless given.

    python3 eval_free.py PREFIX [N] [seconds] [assist]

N episodes on the open floor (goals 1 to 6 m away, any direction; no starts from motion capture: she begins standing).
Reports falls and how long she stayed up, goals reached, her speed while going, and (with PREFIX.amp.pt) the style
reward the discriminator gives her moving, against what it gives the motion capture.
"""
import sys, os, struct, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nanotss_torch as nt, gpu_env as ge

pre = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 256; secs = float(sys.argv[3]) if len(sys.argv) > 3 else 30
assist = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Pool = ge.GpuPool if dev.type == "cuda" else ge.CpuPool
steps = int(secs * 30)
pool = Pool(n, task="goto_free", room="", seed=4242, device=dev, assist=assist, steps=steps)
m = nt.TSS.load(pre + ".tss", device=dev); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v, device=dev)
Win = torch.zeros(d, OBS, device=dev); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d)
Wo = torch.zeros(ACT, d, device=dev); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT, device=dev); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS, device=dev), torch.ones(OBS, device=dev); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
D = None
if os.path.exists(pre + ".amp.pt"):
    ck = torch.load(pre + ".amp.pt", map_location=dev); F = 14 * 15
    D = torch.nn.Sequential(torch.nn.Linear(2 * F, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    D.load_state_dict(ck["D"]); dmu, dsd = ck["mu"].to(dev), ck["sd"].to(dev)
    style = lambda x: (1 - 0.25 * (D(((x - dmu) / dsd).clamp(-5, 5)).squeeze(-1) - 1) ** 2).clamp(min=0)
state = m.zero_state(n); alive = torch.ones(n, dtype=torch.bool, device=dev); up = torch.zeros(n, device=dev)
goals = 0; falls = 0; speeds = []; styles = []; prev = None
with torch.no_grad():
    for t in range(steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        f = ob[:, :14 * 15]
        if D is not None and prev is not None and alive.any(): styles.append(style(torch.cat([prev, f], -1))[alive].mean().item())
        prev = f
        going = alive & (ob[:, 219] > 0.6)   # goal over 0.6 m away
        if going.any(): speeds += ob[going][:, [9, 11]].norm(dim=1).tolist()
        r = pool.step(act); goals += int(((r["rew"] > 1.9) & alive).sum())
        dn = r["done"].bool() & alive; falls += int((dn & r["fell"].bool()).sum()); up += alive.float() / 30; alive &= ~dn
print(f"{pre}: {n} episodes of {secs:.0f} s, hand-up {assist}: falls {falls} ({100 * falls / n:.0f}%), up {up.mean():.1f} s on average; "
      f"goals {goals} ({goals / n:.2f} each); speed while going {np.mean(speeds):.2f} m/s (90th pct {np.percentile(speeds, 90):.2f})")
if D is not None:
    real = torch.cat([torch.cat([torch.tensor(x[:-1]), torch.tensor(x[1:])], 1) for x in ge.amp_reference(os.path.join(HERE, "../data/cmu_loco.bin"), starts=1)[0]]).to(dev)
    print(f"style reward: hers {np.mean(styles):.3f}, the motion capture's {style(real).mean().item():.3f}")
