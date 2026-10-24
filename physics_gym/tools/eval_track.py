"""eval_track.py: the get-up tracker (task getup_imitate) with no hand-up: python3 tools/eval_track.py PRE [FALLS]
FALLS: share of episodes starting from a real fall with the nearest clip frame taking over (env.c matched_start; 1 =
all), the rest from a random frame of a clip. Counts the episodes that started low (lying, sitting, on all fours) and
reached standing height, and those that started high and kept tracking the whole 8 s."""
import sys, os, struct, numpy as np, torch
sys.path.insert(0, os.environ.get('GPU_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpu'))); import nanotss_torch as nt, gpu_env as ge
pre = sys.argv[1]; n = 96
ST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cmu_ref.bin") + ":getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2"
pool = ge.CpuPool(n, task="getup_imitate", style=ST, room="", seed=7, device="cpu", assist=0, style_start=float(sys.argv[2]) if len(sys.argv) > 2 else 0)
m = nt.TSS.load(pre + ".tss", device='cpu'); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
state = m.zero_state(n); ob = pool.obs(); h0 = ob[:, 1].clone(); hmax = ob[:, 1].clone(); alive = torch.ones(n, dtype=torch.bool); life = torch.zeros(n)
with torch.no_grad():
    for t in range(240):
        on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        r = pool.step(act); ob = pool.obs()
        hmax = torch.where(alive, torch.maximum(hmax, ob[:, 1]), hmax); life += alive.float(); alive &= ~r["done"].bool()
low = h0 < 0.45
print(f"started low (lying/sitting): {int(low.sum())}; of them reached standing height (pelvis 0.8 m above the feet): {int((hmax[low] > 0.8).sum())}; mean time tracked {float(life[low].mean())/30:.1f} s")
print(f"started high: {int((~low).sum())}; kept tracking 8 s: {int((life[~low] >= 239).sum())}")
