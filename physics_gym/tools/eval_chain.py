"""eval_chain.py: a jump tracker on chained jump courses (env.h TaskCfg.chain), with no hand-up.
    python3 tools/eval_chain.py PREFIX COURSES.json [N] [seconds] [chain_yaw]
Each of N environments plays one episode from the start of its room's clip (as the viewer's first jump does), each jump
after the first from where the last one landed, aimed by her feet (give or take chain_yaw, 0 by default: the viewer's
aim). Counts the jumps landed before she drops into a gap or falls, per clip: their mean, the share of episodes reaching
2, 4, 6 and 8, and how many of the episodes end down. A jump is landed when the next one starts (she stood steady on the
far platform), so the last one of an episode cut off by its time limit is not counted."""
import sys, os, json, struct, numpy as np, torch
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(B, 'gpu')); import nanotss_torch as nt, gpu_env as ge
pre, rooms_file = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 128
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 30; yaw = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
doc = json.load(open(rooms_file)); clips = doc["clips"]; ST = os.path.join(B, "data", "cmu_jump.bin") + ":" + ",".join(clips)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); Pool = ge.GpuPool if dev.type == "cuda" else ge.CpuPool
steps = int(secs * 30)
pool = Pool(n, task="getup_imitate", style=ST, room=rooms_file, seed=17, device=dev, assist=0, steps=steps, chain=1, chain_yaw=yaw, fixed_start=1)
m = nt.TSS.load(pre + ".tss", device=dev); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v, device=dev)
Win = torch.zeros(d, OBS, device=dev); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d)
Wo = torch.zeros(ACT, d, device=dev); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT, device=dev); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS, device=dev), torch.ones(OBS, device=dev); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(n); alive = torch.ones(n, dtype=torch.bool, device=dev); down = torch.zeros(n, dtype=torch.bool, device=dev)
J = np.zeros(n)
with torch.no_grad():
    for t in range(steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        if t % 15 == 14 or t == steps - 1: J = np.where(alive.cpu().numpy(), pool.jumps(), J)   # the count, until her episode ends
        r = pool.step(act)
        dn = r["done"].bool().to(dev) & alive; down |= dn & r["fell"].bool().to(dev); alive &= ~dn
room = pool.rooms(); clip_of = np.array([clips[r % len(clips)] for r in room])
print(f"{pre}: {n} episodes of {secs:.0f} s from each clip's start, aiming error up to {yaw} rad: jumps landed {J.mean():.2f} on average "
      f"(reaching 2: {np.mean(J >= 2) * 100:.0f}%, 4: {np.mean(J >= 4) * 100:.0f}%, 6: {np.mean(J >= 6) * 100:.0f}%, 8: {np.mean(J >= 8) * 100:.0f}%); "
      f"ended down {int(down.sum())} of {n}")
for c in clips:
    k = clip_of == c
    if k.any(): print(f"  {c}: {k.sum()} episodes, {J[k].mean():.2f} jumps on average, max {int(J[k].max())}")
