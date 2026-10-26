"""eval_course.py: a tracker on a course bank (env.c course mode: climbing, jumping over), with no hand-up.
    python3 tools/eval_course.py PREFIX COURSES.json CLIPS.bin [N] [hand-up]
Each of N environments plays its room's clip once from its start. It counts as done when, a second after the clip ends,
she is still up and her pelvis is within 0.5 m (across) and 0.2 m (in height) of where the clip's ends: on top of the
box, or past the obstacle. Per clip, and how many episodes ended down."""
import sys, os, json, ctypes, struct, numpy as np, torch
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(B, 'gpu')); import nanotss_torch as nt, gpu_env as ge
pre, rooms_file, clip_file = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 192; hand = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
doc = json.load(open(rooms_file)); names = doc["clips"]; clips = ge.load_mref(clip_file)
T = {k: (clips[k]["frames"] - 1) / clips[k]["fps"] for k in names}; steps = int(max(T.values()) * 30) + 40
pool = ge.CpuPool(n, task="getup_imitate", style=clip_file + ":" + ",".join(names), room=rooms_file, seed=23, device="cpu", assist=hand, steps=steps, fixed_start=1)
L = pool.L; L.pool_pelvis.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
m = nt.TSS.load(pre + ".tss", device="cpu"); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d)
bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact); mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
room = pool.rooms(); clip_of = [names[r % len(names)] for r in room]
goal = {k: clips[k]["root"][-1] - np.array([clips[k]["root"][0][0], 0, clips[k]["root"][0][2]]) for k in names}   # the clip's end, in its room (course: root at frame 0 over the origin,
lift = np.zeros(n)                                                                                                      # raised by what it starts on: measured from where she is put)
check = np.array([int(T[k] * 30) + 30 for k in clip_of]); done = np.zeros(n, bool); down = np.zeros(n, bool); alive = np.ones(n, bool); buf = np.zeros(3)
state = m.zero_state(n)
with torch.no_grad():
    for t in range(steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        if t == 0:
            for e in range(n): L.pool_pelvis(pool.p, int(e), buf.ctypes.data_as(ctypes.c_void_p)); lift[e] = buf[1] - clips[clip_of[e]]["root"][0][1]
        for e in np.nonzero(alive & (check == t))[0]:
            L.pool_pelvis(pool.p, int(e), buf.ctypes.data_as(ctypes.c_void_p)); g = goal[clip_of[e]]
            done[e] = np.hypot(buf[0] - g[0], buf[2] - g[2]) < 0.5 and abs(buf[1] - lift[e] - g[1]) < 0.2
        r = pool.step(act); dn = r["done"].numpy().astype(bool) & alive; down |= dn & r["fell"].numpy().astype(bool); alive &= ~dn
print(f"{pre}: {n} episodes from each clip's start, hand-up {hand}: done {int(done.sum())} ({100 * done.mean():.0f}%), ended down {int(down.sum())}")
for k in names:
    s = np.array([c == k for c in clip_of])
    if s.any(): print(f"  {k}: {int(done[s].sum())} of {int(s.sum())} done, {int(down[s].sum())} down")
