"""eval_course.py: a tracker on a course bank (env.c course mode: climbing, jumping over), with no hand-up.
    python3 tools/eval_course.py PREFIX COURSES.json CLIPS.bin [N] [hand-up] [strength] [pos-only] [RECORD.json]
Each of N environments plays its room's clip once from its start. It counts as done when she is still up and her pelvis
is within 0.5 m (across) and 0.2 m (in height) of where the clip's ends: on top of the box, or past the obstacle. Judged
a second after the clip ends if it ends at rest (under 0.5 m/s: she should be standing there), else as it ends (most PARC
clips end at a run, and no one stops dead). Per clip, and how many episodes ended down.
RECORD.json: environment 0's bodies every control step (30 Hz) with its room and the clip's pelvis path, for the viewer to
replay (web/index.html #replay=RECORD.json): what it shows is exactly what was judged. An episode ended early (down, or
more than 1 m off the clip's path) is played on past that in a second run with no path limit (the same start, the same
actions up to there: checked), until she is down or the clip is over; judged_end marks where the judged one ended.
EVAL_NOISE=r (environment): each joint's start rotation off the clip's by up to r rad (TaskCfg.init_noise), so the episodes
differ; imitation's default is none, every episode from a clip's start then the same."""
import sys, os, json, ctypes, struct, numpy as np, torch
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(B, 'gpu')); import nanotss_torch as nt, gpu_env as ge
pre, rooms_file, clip_file = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 192; hand = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
strength = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # her muscles times this (TaskCfg.strength; 0 = 1)
pos_only = int(sys.argv[7]) if len(sys.argv) > 7 else 0   # 1: the hand-up along the path only (TaskCfg.assist_pos_only)
record = sys.argv[8] if len(sys.argv) > 8 else ""
noise = float(os.environ["EVAL_NOISE"]) if os.environ.get("EVAL_NOISE") else None
doc = json.load(open(rooms_file)); names = doc["clips"]; clips = ge.load_mref(clip_file)
T = {k: (clips[k]["frames"] - 1) / clips[k]["fps"] for k in names}; steps = int(max(T.values()) * 30) + 40
pool = ge.CpuPool(n, task="getup_imitate", style=clip_file + ":" + ",".join(names), room=rooms_file, seed=23, device="cpu", assist=hand, steps=steps, fixed_start=1, strength=strength, assist_pos_only=pos_only, init_noise=noise)
L = pool.L; L.pool_pelvis.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]; L.pool_bodies.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
frames, fb, ended = [], np.zeros(14 * 7), None
raw = open(pre + ".head", "rb").read(); hd, hobs, hact = struct.unpack("<3i", raw[4:16]); o = 16
m = nt.TSS.load(pre + ".tss", device="cpu") if os.path.exists(pre + ".tss") else None; d = m.c.d if m else hd; OBS, ACT = pool.OBS, pool.ACT   # (an --mlp actor needs no stack)
def take(k):
    global o; v = np.frombuffer(raw, np.float32, k, o).copy(); o += 4 * k; return torch.tensor(v)
Win = torch.zeros(d, OBS); Win[:, :hobs] = take(d * hobs).view(d, hobs); b = take(d); Wo = torch.zeros(ACT, d); Wo[:hact] = take(hact * d).view(hact, d)
bo = torch.zeros(ACT); bo[:hact] = take(hact); take(hact); mean, sd = torch.zeros(OBS), torch.ones(OBS); mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
actor = None
if os.path.exists(pre + ".mlp.pt"):   # ppo_gpu.py --mlp: a float MLP actor (the head file still holds the normaliser)
    from ppo_gpu import mlp_actor
    ck = torch.load(pre + ".mlp.pt", map_location="cpu"); actor = mlp_actor(ck["obs"], ck["width"], ck["act"]); actor.load_state_dict(ck["actor"]); actor.eval()
room = pool.rooms(); clip_of = [names[r % len(names)] for r in room]
goal = {k: clips[k]["root"][-1] - np.array([clips[k]["root"][0][0], 0, clips[k]["root"][0][2]]) for k in names}   # the clip's end, in its room (course: root at frame 0 over the origin,
lift = np.zeros(n)                                                                                                      # raised by what it starts on: measured from where she is put)
end_v = {k: np.linalg.norm((clips[k]['root'][-1] - clips[k]['root'][-4])[[0, 2]]) * clips[k]['fps'] / 3 for k in names}
tall = [max((bx for bx in doc["rooms"][r]["boxes"] if bx["kind"] != "outer"), key=lambda bx: bx["c"][1] + bx["h"][1], default=None) for r in room]   # each room's tallest box
ontop = np.zeros(n, bool)   # her pelvis 0.6 m over its top, above it, at some point (got up onto it)
check = np.array([int(T[k] * 30) + (30 if end_v[k] < 0.5 else 0) for k in clip_of]); done = np.zeros(n, bool); down = np.zeros(n, bool); alive = np.ones(n, bool); buf = np.zeros(3)
def act_of(on, state):
    if actor is not None: return torch.tanh(actor(on)).clamp(-1, 1)
    x = m.forward((on @ Win.T + b)[:, None], state=state, stream=True)[:, 0]; u, _ = nt.rmsnorm(x, m.P["norm_f"]); return torch.tanh(bo + u @ Wo.T).clamp(-1, 1)
state = m.zero_state(n) if m else None
with torch.no_grad():
    for t in range(steps):
        ob = pool.obs(); on = ((ob - mean) / sd).clamp(-5, 5); act = act_of(on, state)
        if t == 0:
            for e in range(n): L.pool_pelvis(pool.p, int(e), buf.ctypes.data_as(ctypes.c_void_p)); lift[e] = buf[1] - clips[clip_of[e]]["root"][0][1]
        for e in np.nonzero(alive & (check == t))[0]:
            L.pool_pelvis(pool.p, int(e), buf.ctypes.data_as(ctypes.c_void_p)); g = goal[clip_of[e]]
            done[e] = np.hypot(buf[0] - g[0], buf[2] - g[2]) < 0.5 and abs(buf[1] - lift[e] - g[1]) < 0.2
        for e in np.nonzero(alive & ~ontop)[0]:
            bx = tall[e]
            if bx is None: continue
            L.pool_pelvis(pool.p, int(e), buf.ctypes.data_as(ctypes.c_void_p))
            ontop[e] = buf[1] > bx["c"][1] + bx["h"][1] + 0.6 and abs(buf[0] - bx["c"][0]) < bx["h"][0] + 0.1 and abs(buf[2] - bx["c"][2]) < bx["h"][2] + 0.1
        if record and ended is None: L.pool_bodies(pool.p, 0, fb.ctypes.data_as(ctypes.c_void_p)); frames.append([round(float(v), 5) for v in fb])
        r = pool.step(act); dn = r["done"].numpy().astype(bool) & alive; down |= dn & r["fell"].numpy().astype(bool); alive &= ~dn
        if record and ended is None and dn[0]: ended = t + 1   # (the episode over: its environment starts another)
print(f"{pre}: {n} episodes from each clip's start{f' (start noise {noise:g} rad)' if noise else ''}, hand-up {hand}{' along the path only' if pos_only else ''}{f', strength x{strength:g}' if strength else ''}: done {int(done.sum())} ({100 * done.mean():.0f}%), ended down {int(down.sum())}, up on its tallest box {int(ontop.sum())}")
for k in names:
    s = np.array([c == k for c in clip_of])
    if s.any(): print(f"  {k}: {int(done[s].sum())} of {int(s.sum())} done, {int(down[s].sum())} down, {int(ontop[s].sum())} up on its tallest box")
if record:
    R = doc["rooms"][room[0]]; c = clips[clip_of[0]]; r0 = np.array(c["root"][0])
    ref = [(np.array(c["root"][min(int(round(k / 30 * c["fps"])), c["frames"] - 1)]) - [r0[0], 0, r0[2]] + [0, lift[0], 0]).round(4).tolist() for k in range(int(T[clip_of[0]] * 30) + 1)]   # the clip's pelvis, where she is judged against
    judged, why = len(frames), ""
    if ended is not None and ended < len(frames) + 1 and not done[0]:   # ended early: on past it with no path limit, to see what she does
        F = np.array(frames[-1][:3]); off = float(np.linalg.norm(F - ref[min(len(frames) - 1, len(ref) - 1)]))
        why = f"more than 1 m off the clip's path, {off:.1f} m from its pelvis" if off > 0.95 else "down"
        p2 = ge.CpuPool(n, task="getup_imitate", style=clip_file + ":" + ",".join(names), room=rooms_file, seed=23, device="cpu", assist=hand, steps=steps, fixed_start=1, strength=strength, assist_pos_only=pos_only, drift=100, init_noise=noise)
        st2, again = m.zero_state(n) if m else None, []   # (all n as before: the start draws and the stack's batch the same)
        with torch.no_grad():
            for t in range(steps):
                o2 = p2.obs(); L.pool_bodies(p2.p, 0, fb.ctypes.data_as(ctypes.c_void_p)); again.append([round(float(v), 5) for v in fb])
                if p2.step(act_of(((o2 - mean) / sd).clamp(-5, 5), st2))["done"][0]: break
        same = all(np.allclose(a, b_, atol=1e-4) for a, b_ in zip(again[:judged], frames))
        print(f"  the second run (no path limit) the same to step {judged}: {same}; on to step {len(again)}")
        if same: frames = again
    json.dump({"clip": clip_of[0], "source": R.get("source", ""), "prefix": os.path.basename(pre), "hand_up": hand, "strength": strength, "fps": 30, "done": bool(done[0]), "down": bool(down[0]),
               "judged_end": judged, "why": why, "ref": ref, "boxes": R["boxes"], "frames": frames}, open(record, "w"), separators=(",", ":"))
    print(f"{record}: {len(frames)} frames of {clip_of[0]} ({'done' if done[0] else 'ended at step ' + str(judged) + ': ' + why})")
