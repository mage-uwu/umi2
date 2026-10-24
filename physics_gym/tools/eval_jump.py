"""eval_jump.py: a jump tracker (task getup_imitate over a jump course bank, env.c course mode) with no hand-up.
    python3 tools/eval_jump.py PREFIX COURSES.json [N]
Each of N environments plays one episode (8 s) from a random moment of its room's clip. Of those that start on the
takeoff platform (and of those with at least 0.5 s of run-up there: a full jump), counts the ones that get across and stand on the landing platform (pelvis over it, upright, for a second), per
clip; and how many drop into the gap (start over)."""
import sys, os, json, struct, ctypes, numpy as np, torch
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(B, 'gpu')); import nanotss_torch as nt, gpu_env as ge
pre, rooms_file = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 192
doc = json.load(open(rooms_file)); clips = doc["clips"]; ST = os.path.join(B, "data", "cmu_jump.bin") + ":" + ",".join(clips)
pool = ge.CpuPool(n, task="getup_imitate", style=ST, room=rooms_file, seed=11, device="cpu", assist=0)
L = pool.L; L.pool_pelvis.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]; L.pool_room.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.pool_room.restype = ctypes.c_int
m = nt.TSS.load(pre + ".tss", device='cpu'); d = m.c.d; OBS, ACT = pool.OBS, pool.ACT
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d); bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact)
mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
def on_box(bx, p):   # (x, z) inside the box's footprint
    q = bx["q"]; yaw = 2 * np.arctan2(q[2], q[0]); dx, dz = p[0] - bx["c"][0], p[2] - bx["c"][2]
    lx, lz = np.cos(yaw) * dx - np.sin(yaw) * dz, np.sin(yaw) * dx + np.cos(yaw) * dz
    return abs(lx) < bx["h"][0] and abs(lz) < bx["h"][2]
room = [L.pool_room(pool.p, e) for e in range(n)]; buf = np.zeros(3)
def pelvis(e): L.pool_pelvis(pool.p, e, buf.ctypes.data_as(ctypes.c_void_p)); return buf.copy()
start_a = [on_box(doc["rooms"][room[e]]["boxes"][0], pelvis(e)) for e in range(n)]
landed = np.zeros(n, bool); dropped = np.zeros(n, bool); alive = np.ones(n, bool); state = m.zero_state(n)
left_a = np.full(n, -1)   # the step her pelvis first leaves the takeoff platform
on_b = np.zeros(n, int)   # consecutive steps standing on the landing platform (30: a second, landed)
with torch.no_grad():
    for t in range(240):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5)
        x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); act = torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
        r = pool.step(act); done = r["done"].numpy().astype(bool)
        for e in range(n):
            if not alive[e]: continue
            if done[e]: dropped[e] |= bool(r["fell"][e]); alive[e] = False; continue
            p = pelvis(e); R = doc["rooms"][room[e]]
            if left_a[e] < 0 and not on_box(R["boxes"][0], p): left_a[e] = t
            on_b[e] = on_b[e] + 1 if on_box(R["boxes"][1], p) and p[1] - R["height"] > 0.6 else 0
            if on_b[e] >= 30: landed[e] = True
print(f"{pre}: {n} episodes; ended early (dropped or drifted off the clip) {int(dropped.sum())}")
for k, c in enumerate(clips):
    idx = [e for e in range(n) if room[e] % len(clips) == k and start_a[e]]
    print(f"  {c:10s} gap {doc['rooms'][k]['gap']:.2f} m: started on the takeoff platform {len(idx)}, landed and stood on the far one {int(sum(landed[e] for e in idx))}")
idx = [e for e in range(n) if start_a[e]]; print(f"  all: {int(sum(landed[e] for e in idx))} of {len(idx)}")
full = [e for e in range(n) if start_a[e] and (left_a[e] < 0 or left_a[e] >= 15)]   # at least 0.5 s of run-up on the takeoff platform: the takeoff is hers
print(f"  full jumps (0.5 s or more of run-up before leaving the takeoff platform): {int(sum(landed[e] for e in full))} of {len(full)}")
for k, c in enumerate(clips):
    f = [e for e in full if room[e] % len(clips) == k]; print(f"    {c:10s} {int(sum(landed[e] for e in f))} of {len(f)}")
