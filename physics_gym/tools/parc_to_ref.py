#!/usr/bin/env python3
"""parc_to_ref.py: PARC parkour clips (Xu, Shi, Yin, Peng: "PARC: Physics-based Augmentation with Reinforcement Learning
for Character Controllers", SIGGRAPH 2025; data: huggingface.co/datasets/mxucg/PARC, MIT; code: github.com/mshoe/PARC,
BSD-3) as clips and course rooms for the character.

    python3 tools/parc_to_ref.py OUT.bin OUT_bank.json file.pkl [...]

Each PARC file holds a motion (root position and rotation, the 14 joints' rotations in their parents, contact labels for
its 15 bodies, 30 fps) on the AMP humanoid (z up, x forward, y left; at rest a T-pose with arms along +-y) and the terrain
it was made for: a 2.5D heightfield of 0.4 m cells (hf[i, j] the top of the column at min_point + (i, j) * dx).
  The motion: each body's world rotation by forward kinematics, the axes turned into hers (y up, z forward, x left; her
  rest pose is a T-pose too, so the body frames line up), the torso as her chest and halfway between it and the pelvis as
  her spine, the hands left out (her forearm boxes carry them). The root scaled by her pelvis height over theirs (0.915 m
  over 0.881 m from the sole), and lowered by the height of the column under its start, since course mode raises a clip by
  what it starts on.
  The terrain: the columns within 1.6 m of the root's path, each a box from the floor (the lowest of them at 0) to its top,
  equal neighbours merged into rectangles, in the clip's own frame (its root at frame 0 over the origin).
The files are pickles: they are read with an unpickler that builds numpy arrays and plain containers only.
Writes an MREF file of the clips (named p<n>) and a room bank of the same order (env.c course mode: room r for clip r).
"""
import io, json, pickle, struct, sys
import numpy as np

S = 0.915 / 0.881   # her pelvis height from the sole over the AMP humanoid's
PARENT = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13]   # pelvis, torso, head, r_uarm, r_larm, r_hand, l_uarm, l_larm, l_hand, r_thigh, r_shin, r_foot, l_thigh, l_shin, l_foot
# her bodies (body.h): PELVIS SPINE CHEST HEAD L_UARM L_FARM R_UARM R_FARM L_THIGH L_SHIN L_FOOT R_THIGH R_SHIN R_FOOT, from theirs
MAP = [0, None, 1, 2, 6, 7, 3, 4, 12, 13, 14, 9, 10, 11]


class _Safe(pickle.Unpickler):
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
          ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("collections", "OrderedDict")}
    def find_class(self, mod, name):
        if (mod, name) in self.OK: return getattr(__import__(mod, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"not loading {mod}.{name}")


def load(path):
    top = _Safe(open(path, "rb")).load(); return {k: _Safe(io.BytesIO(top[k])).load() for k in ("motion_data", "terrain_data")}


def qmul(a, b):   # (..., 4) w x y z
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def slerp(a, b, t):
    d = np.sum(a * b, -1, keepdims=True); b = np.where(d < 0, -b, b); d = np.abs(d)
    om = np.arccos(np.clip(d, -1, 1)); so = np.sin(om); small = so < 1e-6
    wa = np.where(small, 1 - t, np.sin((1 - t) * om) / np.where(small, 1, so)); wb = np.where(small, t, np.sin(t * om) / np.where(small, 1, so))
    q = wa * a + wb * b; return q / np.linalg.norm(q, axis=-1, keepdims=True)


AX = np.array([0.5, -0.5, -0.5, -0.5])   # the rotation taking their axes to hers: x -> z, y -> x, z -> y (-120 deg about (1, 1, 1))
AXc = AX * np.array([1, -1, -1, -1])


def motion(md):
    rr = np.asarray(md["root_rot"], np.float64)[:, [3, 0, 1, 2]]   # x y z w -> w x y z
    jr = np.asarray(md["joint_rot"], np.float64)[..., [3, 0, 1, 2]]
    T = len(rr); W = np.zeros((T, 15, 4)); W[:, 0] = rr
    for b in range(1, 15): W[:, b] = qmul(W[:, PARENT[b]], jr[:, b - 1])
    W = qmul(qmul(AX, W), AXc)   # in her axes
    Q = np.zeros((T, 14, 4))
    for i, k in enumerate(MAP): Q[:, i] = W[:, k] if k is not None else slerp(W[:, 0], W[:, 1], 0.5)
    for f in range(1, T): Q[f] *= np.where(np.sum(Q[f] * Q[f - 1], -1, keepdims=True) < 0, -1, 1)   # no sign flips between frames
    rp = np.asarray(md["root_pos"], np.float64); root = S * np.stack([rp[:, 1], rp[:, 2], rp[:, 0]], -1)
    return root, Q


def terrain(td, root_parc):
    hf = np.asarray(td["hf"], np.float64); dx = float(td["dx"]); mn = np.asarray(td["min_point"], np.float64)
    I, J = np.meshgrid(np.arange(hf.shape[0]), np.arange(hf.shape[1]), indexing="ij")
    cx, cy = mn[0] + I * dx, mn[1] + J * dx
    d = np.min(np.hypot(cx[..., None] - root_parc[None, None, :, 0], cy[..., None] - root_parc[None, None, :, 1]), -1)
    use = d < 1.6
    i0 = np.clip(np.round((root_parc[0, 0] - mn[0]) / dx).astype(int), 0, hf.shape[0] - 1); j0 = np.clip(np.round((root_parc[0, 1] - mn[1]) / dx).astype(int), 0, hf.shape[1] - 1)
    return hf, use, mn, dx, hf[i0, j0]


def boxes_of(hf, use, mn, dx, base, start):   # merged columns, in her axes and scale, the clip's start over the origin
    H = np.where(use, np.round(hf - base, 3), np.nan); done = np.zeros_like(use); out = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if done[i, j] or not use[i, j] or H[i, j] <= 0.005: continue
            h = H[i, j]; j1 = j
            while j1 + 1 < H.shape[1] and not done[i, j1 + 1] and use[i, j1 + 1] and H[i, j1 + 1] == h: j1 += 1
            i1 = i
            while i1 + 1 < H.shape[0] and all(not done[i1 + 1, k] and use[i1 + 1, k] and H[i1 + 1, k] == h for k in range(j, j1 + 1)): i1 += 1
            done[i:i1 + 1, j:j1 + 1] = True
            x0, x1 = mn[0] + (i - 0.5) * dx, mn[0] + (i1 + 0.5) * dx; y0, y1 = mn[1] + (j - 0.5) * dx, mn[1] + (j1 + 0.5) * dx
            c = S * np.array([(y0 + y1) / 2, h / 2, (x0 + x1) / 2]) - np.array([start[0], 0, start[2]]); hh = S * np.array([(y1 - y0) / 2, h / 2, (x1 - x0) / 2])
            out.append({"kind": "platform", "feature": f"parc {S * h:.2f}", "c": c.round(4).tolist(), "h": hh.round(4).tolist(), "q": [1, 0, 0, 0], "mu": 0.9, "soft": 0, "grip": 1, "color": "#b9a083"})
    return out


def main():
    out, bank_out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    clips, rooms = [], []
    for n, path in enumerate(files):
        D = load(path); md, td = D["motion_data"], D["terrain_data"]
        assert int(md["fps"]) == 30, path
        rp = np.asarray(md["root_pos"], np.float64)
        hf, use, mn, dx, h0 = terrain(td, rp)
        base = np.nanmin(np.where(use, hf, np.nan))   # the lowest column it passes is the floor
        root, Q = motion(md); root[:, 1] -= S * (h0 - base) + 0.0   # course mode raises the clip by what it starts on
        start = root[0].copy()
        bx = boxes_of(hf, use, mn, dx, base, S * np.array([rp[0, 1], 0, rp[0, 0]]))
        if len(bx) > 250: print(f"skip {path}: {len(bx)} boxes"); continue
        name = f"p{len(clips)}"
        clips.append((name, root.astype(np.float32), Q.astype(np.float32)))
        bc = np.asarray(md["body_contacts"]) > 0.5   # their labels: l_foot, r_foot, l_hand, r_hand as bits 0..3 per frame
        lab = (bc[:, 14] * 1 + bc[:, 11] * 2 + bc[:, 8] * 4 + bc[:, 5] * 8).astype(int).tolist()
        rooms.append({"name": f"parc_{name}", "clip": name, "source": path.split("__")[-1], "bounds": [-8, 8, -8, 8], "boxes": bx, "contacts": lab})
        print(f"{name}: {path.split('__')[-1]}: {len(root)} frames, {len(bx)} boxes, start {S * (h0 - base):.2f} m up, rise {root[-1, 1] - root[0, 1]:+.2f} m, travel {np.hypot(*(root[-1, [0, 2]] - root[0, [0, 2]])):.1f} m")
    with open(out, "wb") as f:
        f.write(b"MREF"); f.write(struct.pack("<i", len(clips)))
        for name, r, q in clips:
            f.write(name.encode().ljust(16, b"\0")[:16]); f.write(struct.pack("<if", len(r), 30.0)); f.write(np.concatenate([r, q.reshape(len(q), -1)], 1).astype(np.float32).tobytes())
    json.dump({"version": 1, "source": "PARC (huggingface.co/datasets/mxucg/PARC, MIT)", "clips": [c[0] for c in clips], "rooms": rooms}, open(bank_out, "w"))
    print(f"{out}, {bank_out}: {len(clips)} clips")


if __name__ == "__main__":
    main()
