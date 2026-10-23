#!/usr/bin/env python3
"""cmu_cycle.py: the cleanest loopable gait cycles in the CMU library, as reference clips (MREF) for the priors.

Searches the 30 Hz CMU trials (data/clean/cmu30_library.npz) whose description is a plain gait (walk, run, jog; not
turning, backwards, stairs, carrying, ...) for one gait cycle: frames i < j, j - i within the gait's period range,
where the pose at j matches the pose at i (joint rotations relative to the pelvis, pelvis tilt and height) and the
pelvis travelled straight (little change of heading, steady speed). The best cycle is written with its heading
turned to +z, its start at the origin, and its last frame replaced by the first (shifted by the cycle's travel),
so it loops exactly; and grounded: shifted up or down so the feet in stance (the lowest foot corner, its 10th
percentile over the cycle) are at y = 0, since the go-to start uses the clip's pelvis height as it is. A short
report names the trial and the fit.

    python3 tools/cmu_cycle.py [--out data/clean/cmu_gait.bin]   (clips: walk, run)

Data: CMU Graphics Lab Motion Capture Database, mocap.cs.cmu.edu (terms in data/clean/cmu_library.json).
"""
import argparse, json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from joint_limits import qmul, qlog, PARENT
import muscle_envelope as me   # the body's boxes (from web/body.js) and its forward kinematics pieces

GAITS = {"walk": dict(words=("walk",), period=(0.85, 1.4), speed=(0.6, 1.8)),
         "run": dict(words=("run", "jog"), period=(0.5, 0.9), speed=(1.8, 5.0))}
AVOID = ("turn", "back", "stair", "carry", "veer", "stop", "uneven", "swing", "stretch", "sideways", "curve", "around",
         "circle", "zig", "slow", "drunk", "limp", "jump", "leap", "hop", "dog", "navigate", "wander", "start", "cat",
         "sneak", "duck", "with", "and", "/ ", "left", "right", "shoulder", "march", "exagger", "stiff", "punch", "kick")
CONJ = np.array([1, -1, -1, -1])


def yaw_of(q):   # heading of a rotation: where it sends +z, about +y
    u = q[..., 1:]; w = q[..., :1]; v = np.array([0, 0, 1.0]); t = 2 * np.cross(u, v); d = v + w * t + np.cross(u, t)
    return np.arctan2(d[..., 0], d[..., 2])


def qyaw(a):
    return np.stack([np.cos(a / 2), 0 * a, np.sin(a / 2), 0 * a], -1)


def features(q, root):
    rel = np.stack([qlog(qmul(q[:, PARENT[i]] * CONJ, q[:, i])) for i in range(1, 14)], 1).reshape(len(q), -1)
    tilt = qlog(qmul(qyaw(-yaw_of(q[:, 0])), q[:, 0]))   # pelvis rotation without its heading
    return np.concatenate([rel, 2 * tilt, 4 * root[:, 1:2]], 1)


def best_cycle(q, root, fps, g):
    F = features(q, root); yaw = np.unwrap(yaw_of(q[:, 0])); best = None
    lo, hi = int(g["period"][0] * fps), int(g["period"][1] * fps)
    for i in range(len(q) - lo):
        for j in range(i + lo, min(i + hi, len(q) - 1) + 1):
            d = root[j] - root[i]; dist = np.hypot(d[0], d[2]); v = dist * fps / (j - i)
            if not g["speed"][0] <= v <= g["speed"][1]: continue
            turn = abs(yaw[j] - yaw[i]) + np.abs(np.diff(yaw[i:j + 1])).sum() * 0.1
            step = np.linalg.norm(np.diff(root[i:j + 1, [0, 2]], axis=0), axis=1) * fps; wobble = step.std() / max(v, 1e-6)
            err = np.linalg.norm(F[j] - F[i]) / np.sqrt(F.shape[1]) + 0.5 * turn + 0.2 * wobble
            if best is None or err < best[0]: best = (err, i, j, v)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "data", "clean", "cmu_gait.bin")); a = ap.parse_args()
    z = np.load(os.path.join(ROOT, "data", "clean", "cmu30_library.npz")); man = json.load(open(os.path.join(ROOT, "data", "clean", "cmu30_library.json")))
    fps = man["fps"]; Q = z["q"].astype(np.float64); Q /= np.linalg.norm(Q, axis=-1, keepdims=True); R = z["root"].astype(np.float64)
    cuts = np.flatnonzero(np.diff(z["trial"])) + 1; trials = list(zip(np.split(Q, cuts), np.split(R, cuts), np.split(z["trial"], cuts)))
    clips = []; report = {}
    for name, g in GAITS.items():
        found = None
        for q, r, t in trials:
            m = man["trials"][int(t[0])]; d = m["desc"].lower()
            if not any(w in d for w in g["words"]) or any(x in d for x in AVOID): continue
            b = best_cycle(q, r, fps, g)
            if b and (found is None or b[0] < found[0][0]): found = (b, q, r, m)
        (err, i, j, v), q, r, m = found
        q = q[i:j + 1].copy(); r = r[i:j + 1].copy()
        d = r[-1] - r[0]; turn = qyaw(np.array(-np.arctan2(d[0], d[2])))   # travel along +z
        q = qmul(np.broadcast_to(turn, q.shape), q)
        u = turn[1:]; t2 = 2 * np.cross(u, r - r[0]); r = r[0] * np.array([0, 1, 0]) + (r - r[0]) + turn[0] * t2 + np.cross(u, t2)
        for k in range(14):   # keep each track continuous, then close the loop exactly
            s = np.sign((q[1:, k] * q[:-1, k]).sum(-1)); s = np.concatenate([[1], np.cumprod(np.where(s == 0, 1, s))]); q[:, k] *= s[:, None]
        q[-1] = np.where((q[-1] * q[0]).sum(-1, keepdims=True) < 0, -q[0], q[0]); r[-1] = r[0] + np.array([r[-1, 0] - r[0, 0], 0, r[-1, 2] - r[0, 2]])
        x = np.zeros((len(q), 14, 3))
        for b in range(14):   # forward kinematics (body.c world_pose), then the feet's lowest corners
            if PARENT[b] < 0: x[:, b] = r
            else: p = PARENT[b]; an = x[:, p] + me.rot(q[:, p], me.ANCHOR[b] - me.CENTER[p]); x[:, b] = an + me.rot(q[:, b], me.CENTER[b] - me.ANCHOR[b])
        low = np.min([x[:, f, 1] + me.rot(q[:, f], np.array([sx, sy, sz]) * me.HALF[f])[:, 1] for f in me.FEET
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], 0)
        lift = -np.percentile(low, 10); r[:, 1] += lift
        clips.append((name, q, r)); report[name] = dict(trial=m["id"], desc=m["desc"], frames=int(len(q)), period_s=round((len(q) - 1) / fps, 2),
                                                        speed_m_s=round(float(v), 2), loop_error=round(float(err), 3), grounding_shift_m=round(float(lift), 3))
    with open(a.out, "wb") as f:
        f.write(b"MREF"); f.write(struct.pack("<i", len(clips)))
        for name, q, r in clips:
            f.write(name.encode().ljust(16, b"\0")); f.write(struct.pack("<if", len(q), float(fps)))
            for k in range(len(q)):
                f.write(struct.pack("<3f", *r[k]))
                for b in range(14): f.write(struct.pack("<4f", *q[k, b]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
