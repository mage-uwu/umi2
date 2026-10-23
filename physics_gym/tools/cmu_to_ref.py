#!/usr/bin/env python3
"""Convert CMU Graphics Lab motion capture (ASF skeleton + AMC motion) into reference clips (MREF, see
motions_to_ref.py) for the physics character.

CMU's rest pose is a T-pose with Y up, the subject's left along +X and facing +Z, as the character's is, and
every bone's global rotation is the identity in it; so a bone's global rotation from forward kinematics is the
world rotation of the body it drives. The root is scaled by the ratio of the character's hip height to the
subject's and placed at the pelvis body centre.

Data: mocap.cs.cmu.edu. "This data is free for use in research projects. You may include this data in
commercially-sold products, but you may not resell this data directly, even in converted form."

usage: python3 cmu_to_ref.py out.bin name=dir/subject.asf:dir/trial.amc[:first:last] ...   (frames at 120 Hz)
       python3 cmu_to_ref.py --plot out.png subject.asf trial.amc   (stick figures, for choosing clips)
"""
import struct, sys
import numpy as np

UNIT = (1.0 / 0.45) * 2.54 / 100            # ASF length units to metres (CMU FAQ)
BODY_BONES = ["root", "lowerback", "thorax", "head", "lhumerus", "lradius", "rhumerus", "rradius",
              "lfemur", "ltibia", "lfoot", "rfemur", "rtibia", "rfoot"]
PELVIS_CENTER = np.array([0, 0.915, 0.005]); HIP_HEIGHT = 0.888   # the character's pelvis centre, hip joints


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def euler(vals, order):   # applied in the listed order: the first rotation is innermost
    m = np.eye(3)
    for ax, v in zip(order, vals): m = rot(ax, v) @ m
    return m


def parse_asf(path):
    bones, hier, root = {}, {}, {}
    lines = open(path).read().split("\n"); i = 0; sec = None
    while i < len(lines):
        t = lines[i].strip(); i += 1
        if t.startswith(":"):
            sec = t.split()[0]
            if sec == ":root":
                while i < len(lines) and not lines[i].strip().startswith(":"):
                    w = lines[i].split(); i += 1
                    if w and w[0] == "order": root["order"] = [x.lower() for x in w[1:]]
                    if w and w[0] == "axis": root["axis"] = w[1].lower()
                    if w and w[0] == "orientation": root["orient"] = [float(x) for x in w[1:4]]
            continue
        if sec == ":bonedata" and t == "begin":
            b = {"dof": []}
            while lines[i].strip() != "end":
                w = lines[i].split(); i += 1
                if w[0] == "name": b["name"] = w[1]
                elif w[0] == "direction": b["dir"] = np.array([float(x) for x in w[1:4]])
                elif w[0] == "length": b["len"] = float(w[1])
                elif w[0] == "axis": b["axis"] = [float(x) for x in w[1:4]]; b["axis_order"] = w[4].lower()
                elif w[0] == "dof": b["dof"] = [x.lower() for x in w[1:]]
            i += 1
            b["C"] = euler(b["axis"], b["axis_order"]); bones[b["name"]] = b
        elif sec == ":hierarchy" and t and t not in ("begin", "end"):
            w = t.split()
            for c in w[1:]: hier[c] = w[0]
    return bones, hier, root


def parse_amc(path):
    frames, cur = [], None
    for line in open(path):
        t = line.strip()
        if not t or t.startswith("#") or t.startswith(":"): continue
        if t.isdigit(): cur = {}; frames.append(cur); continue
        w = t.split(); cur[w[0]] = [float(x) for x in w[1:]]
    return frames


def fk(bones, hier, root, fr):
    """Global rotations and joint end positions (metres, CMU world) of every bone."""
    rv = fr["root"]; order = root.get("order", ["tx", "ty", "tz", "rx", "ry", "rz"])
    tr = np.array([rv[order.index(a)] for a in ("tx", "ty", "tz")]) * UNIT
    rots = [(a[1], rv[order.index(a)]) for a in order if a.startswith("r")]
    G = {"root": euler([v for _, v in rots], [a for a, _ in rots])}; P = {"root": tr}
    done = {"root"}; todo = [b for b in hier]
    while todo:
        for b in list(todo):
            p = hier[b]
            if p not in done: continue
            B = bones[b]; vals = fr.get(b, [0] * len(B["dof"]))
            M = euler(vals, [d[1] for d in B["dof"]]) if B["dof"] else np.eye(3)
            G[b] = G[p] @ B["C"] @ M @ B["C"].T
            P[b] = P[p] + G[b] @ (B["dir"] * B["len"] * UNIT)
            done.add(b); todo.remove(b)
    return G, P


def mat2quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2; q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0: s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1: s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else: s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def convert(asf, amc, first=0, last=None, step=4):
    bones, hier, root = parse_asf(asf); frames = parse_amc(amc)[first:last:step]   # 120 Hz -> 30 Hz
    # the subject's hip height in the rest pose: root to the femur's end, vertically
    _, P0 = fk(bones, hier, root, {"root": [0] * 6})
    hip = -min(P0["ltibia"][1], P0["lfoot"][1], P0["rfoot"][1]) + 0   # rest: root at 0, feet below
    scale = HIP_HEIGHT / hip
    out, qprev = [], [None] * 14
    for fr in frames:
        G, P = fk(bones, hier, root, fr)
        qs = []
        for k, b in enumerate(BODY_BONES):
            q = mat2quat(G[b])
            if qprev[k] is not None and q.dot(qprev[k]) < 0: q = -q   # keep each track continuous
            qprev[k] = q; qs.append(q)
        out.append((P["root"] * scale, qs))
    # ground: the lowest point the subject's body reaches (any joint) sits at the floor over the clip
    return out, 30.0, scale


def main():
    if sys.argv[1] == "--plot":   # side views (z right, y up) of 8 frames across the clip
        from PIL import Image, ImageDraw
        png, asf, amc = sys.argv[2:5]; bones, hier, root = parse_asf(asf); frames = parse_amc(amc)
        idx = np.linspace(0, len(frames) - 1, 8).astype(int); W = 260
        img = Image.new("RGB", (W * 8, W), "white"); d = ImageDraw.Draw(img)
        for k, i in enumerate(idx):
            G, P = fk(bones, hier, root, frames[i]); lo = min(p[1] for p in P.values()); zc = P["root"][2]
            xy = lambda p: (k * W + W / 2 + (p[2] - zc) * 120, W - 20 - (p[1] - lo) * 120)
            d.line([(k * W, W - 20), (k * W + W - 4, W - 20)], fill=(200, 200, 200))
            for b, pa in hier.items():
                d.line([xy(P[pa]), xy(P[b])], fill=(200, 40, 40) if b[0] == "l" else (40, 60, 200) if b[0] == "r" else (0, 0, 0), width=3)
            d.text((k * W + 6, 4), f"{i / 120:.1f} s", fill=(0, 0, 0))
        img.save(png); return
    path = sys.argv[1]; clips = []
    for spec in sys.argv[2:]:
        name, files = spec.split("=", 1); parts = files.split(":")
        first = int(parts[2]) if len(parts) > 2 else 0; last = int(parts[3]) if len(parts) > 3 else None
        frames, fps, scale = convert(parts[0], parts[1], first, last)
        lo = min(f[0][1] for f in frames)   # (root height range, for the report)
        clips.append((name, frames, fps)); print(f"{name}: {len(frames)} frames at {fps} Hz, scale {scale:.3f}, root height {lo:.2f}..{max(f[0][1] for f in frames):.2f} m")
    with open(path, "wb") as f:
        f.write(b"MREF"); f.write(struct.pack("<i", len(clips)))
        for name, frames, fps in clips:
            f.write(name.encode()[:16].ljust(16, b"\0")); f.write(struct.pack("<if", len(frames), fps))
            for root, qs in frames:
                f.write(struct.pack("<3f", *(root + PELVIS_CENTER - [0, 0, 0] - [0, 0, 0])))
                for q in qs: f.write(struct.pack("<4f", *q))


if __name__ == "__main__":
    main()
