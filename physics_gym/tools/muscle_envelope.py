"""muscle_envelope.py: how strong and how quick each joint's muscles are in the motion capture, on this body.

Inverse dynamics of the capture clips (data/motions_ref.bin: walk, run, dance, bows, waves; her feet are the
only contacts) on the simulated body (its 14 boxes, 45 kg, masses and inertias as body.c computes them): each
frame's pose by forward kinematics (as body.c world_pose), velocities and accelerations by central differences,
the ground's push shared between the feet on the ground, then the Newton-Euler recursion from the hands, head and
feet up to the pelvis gives the torque each joint's muscles must make, and how fast it changes. Every clip is
used as captured and mirrored left to right.

Prints, per joint, the 50th / 95th / 99th percentile torque (N m) and rate of change (N m/s), and writes
muscle_envelope.json and the muscles' limits for the physics (muscle.h for body.c, web/muscle.js for body.js):

  strength (N m): the larger of the joint's maximum voluntary torque (published adult values for women, the
      stronger direction, scaled to 45 kg; the capture shows typical use, and getting up needs more arm than any
      clip here) and the capture's 95th percentile (its 99th is landing and flight noise in the run and dance);
  slew (N m/s): how fast the joint's torque may rise (activation dynamics): the larger of 10 x strength per
      second (about 0.1 s from rest to full force) and the capture's 95th percentile rate.

    python3 tools/muscle_envelope.py
"""
import json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "gpu"))
import gpu_env as ge

# BODY_DEFS from body.js (the same table as body.c): name, bone, parent, anchor, center, half, mass_frac, kp, kd, hinge
src = open(os.path.join(ROOT, "web", "body.js")).read()
rows = re.findall(r"\['(\w+)', '(\w+)', (-?\d+), \[([^\]]+)\], \[([^\]]+)\], \[([^\]]+)\], ([\d.]+), (\d+), (\d+), (\d)\]", src)
NB = len(rows); assert NB == 14
NAMES = [r[0] for r in rows]; PARENT = [int(r[2]) for r in rows]
ANCHOR = np.array([[float(x) for x in r[3].split(",")] for r in rows]); CENTER = np.array([[float(x) for x in r[4].split(",")] for r in rows])
HALF = np.array([[float(x) for x in r[5].split(",")] for r in rows]); FRAC = np.array([float(r[6]) for r in rows]); KP = np.array([float(r[7]) for r in rows])
MASS = 45 * FRAC / FRAC.sum()
X, Y, Z = 2 * HALF[:, 0], 2 * HALF[:, 1], 2 * HALF[:, 2]
IB = np.stack([MASS * (Y * Y + Z * Z) / 12, MASS * (X * X + Z * Z) / 12, MASS * (X * X + Y * Y) / 12], 1)   # body-frame diagonal inertia
MIRROR = [0, 1, 2, 3, 6, 7, 4, 5, 11, 12, 13, 8, 9, 10]
# maximum voluntary joint torque (N m), adult women, the stronger direction, times 45/60 for this body: trunk (the
# spine joint takes the lumbar share, the chest joint the thoracic), neck, shoulder, elbow, hip, knee, ankle
MVC = {k: round(v * 45 / 60, 1) for k, v in {"spine": 150, "chest": 100, "head": 20, "uarm": 40, "farm": 35, "thigh": 160, "shin": 135, "foot": 95}.items()}
G = np.array([0, -9.81, 0]); FEET = (10, 13)


def rot(q, v):   # q [..., 4] applied to v [..., 3]
    u = q[..., 1:]; t = 2 * np.cross(u, v); return v + q[..., :1] * t + np.cross(u, t)


def qmul(a, b):
    return np.stack([a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1),
                     *(a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])).transpose(-1, *range(a.ndim - 1))], -1)


def qlog(q):
    q = np.where(q[..., :1] < 0, -q, q); s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    return np.where(s < 1e-12, 2 * q[..., 1:], q[..., 1:] * (2 * np.arctan2(s, q[..., :1]) / np.maximum(s, 1e-12)))


def mirror_clip(root, q):   # reflection in the x = 0 plane: positions x -> -x, rotations (w, x, y, z) -> (w, x, -y, -z), left <-> right
    return root * np.array([-1, 1, 1]), (q * np.array([1, 1, -1, -1]))[:, MIRROR]


def smooth(a, sigma):   # Gaussian along time (axis 0), edges held
    r = int(3 * sigma) + 1; k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return sum(k[j] * p[j:j + len(a)] for j in range(2 * r + 1))


def inverse_dynamics(root, q, fps):
    F = len(q); dt = 1 / fps; sig = 0.05 * fps   # about 50 ms of smoothing before differencing (capture jitter)
    root = smooth(root, sig); qs = smooth(q * np.where((q * q[:1]).sum(-1, keepdims=True) < 0, -1, 1), sig); q = qs / np.linalg.norm(qs, axis=-1, keepdims=True)
    x = np.zeros((F, NB, 3))
    for i in range(NB):   # forward kinematics (body.c world_pose)
        if PARENT[i] < 0: x[:, i] = root
        else:
            p = PARENT[i]; a = x[:, p] + rot(q[:, p], ANCHOR[i] - CENTER[p]); x[:, i] = a + rot(q[:, i], CENTER[i] - ANCHOR[i])
    v = np.gradient(x, dt, axis=0); acc = np.gradient(v, dt, axis=0)
    dq = qmul(q[2:], q[:-2] * np.array([1, -1, -1, -1])); w = np.zeros((F, NB, 3)); w[1:-1] = qlog(dq) / (2 * dt); w[0], w[-1] = w[1], w[-2]
    alpha = np.gradient(w, dt, axis=0)
    # world inertia I = R diag(IB) R^T applied to vectors
    Iw = lambda i, vec: rot(q[:, i], IB[i] * rot(q[:, i] * np.array([1, -1, -1, -1]), vec))
    need = MASS[None, :, None] * (acc - G)   # force each body needs (N)
    # the ground's push: all of it (sum of needs), shared between the feet on the ground (the lower ones, and still)
    total = need.sum(1)
    fy = np.stack([x[:, f, 1] - HALF[f, 1] for f in FEET], 1); fspeed = np.stack([np.linalg.norm(v[:, f], axis=-1) for f in FEET], 1)
    lowest = fy.min(1, keepdims=True); on = (fy < lowest + 0.03) & (fspeed < 1.5); on[on.sum(1) == 0] = True
    share = on / on.sum(1, keepdims=True); grf = {f: total * share[:, k:k + 1] for k, f in enumerate(FEET)}
    cop = {f: x[:, f] - np.array([0, HALF[f, 1], 0]) for f in FEET}   # the push acts under the foot's centre
    # Newton-Euler from the leaves: force and torque each joint passes up to its parent
    order = sorted(range(1, NB), key=lambda i: -depth(i))
    Fj = np.zeros((F, NB, 3)); Tj = np.zeros((F, NB, 3)); anchor = np.zeros((F, NB, 3))
    for i in range(1, NB):
        p = PARENT[i]; anchor[:, i] = x[:, p] + rot(q[:, p], ANCHOR[i] - CENTER[p])
    for i in order:
        f = need[:, i].copy(); t = Iw(i, alpha[:, i]) + np.cross(w[:, i], Iw(i, w[:, i])) + np.cross(x[:, i] - anchor[:, i], need[:, i])
        if i in FEET: f = f - grf[i]; t = t - np.cross(cop[i] - anchor[:, i], grf[i])
        for c in range(1, NB):
            if PARENT[c] == i: f = f + Fj[:, c]; t = t + Tj[:, c] + np.cross(anchor[:, c] - anchor[:, i], Fj[:, c])
        Fj[:, i] = f; Tj[:, i] = t
    return Tj   # torque at each joint (world frame), what the muscles there must provide


def depth(i):
    d = 0
    while PARENT[i] >= 0: i = PARENT[i]; d += 1
    return d


def main():
    clips = ge.load_mref(os.path.join(ROOT, "data", "motions_ref.bin")); tau = []; rate = []
    for name, c in clips.items():
        for root, q in ((c["root"], c["q"]), mirror_clip(c["root"], c["q"])):
            T = inverse_dynamics(root, q, c["fps"])[3:-3]   # the ends lack proper differences
            m = np.linalg.norm(T, axis=-1); tau.append(m); rate.append(np.abs(np.gradient(m, 1 / c["fps"], axis=0)))
    tau = np.concatenate(tau); rate = np.concatenate(rate)
    out = {}
    print(f"{'joint':8s} {'tau50':>7s} {'tau95':>7s} {'tau99':>7s} {'rate50':>8s} {'rate95':>8s} {'rate99':>8s}   (N m, N m/s)")
    for i in range(1, NB):
        t50, t95, t99 = np.percentile(tau[:, i], [50, 95, 99]); r50, r95, r99 = np.percentile(rate[:, i], [50, 95, 99])
        out[NAMES[i]] = {k: round(float(v), 2) for k, v in (("tau50", t50), ("tau95", t95), ("tau99", t99), ("rate50", r50), ("rate95", r95), ("rate99", r99))}
        print(f"{NAMES[i]:8s} {t50:7.1f} {t95:7.1f} {t99:7.1f} {r50:8.0f} {r95:8.0f} {r99:8.0f}")
    json.dump({"source": "data/motions_ref.bin, inverse dynamics on the 45 kg body, mirrored", "frames": int(len(tau)), "joints": out},
              open(os.path.join(ROOT, "muscle_envelope.json"), "w"), indent=1)
    strength = [0.0] + [max(MVC[n.split("_")[-1]], out[n]["tau95"]) for n in NAMES[1:]]
    slew = [0.0] + [max(10 * strength[i], out[NAMES[i]]["rate95"]) for i in range(1, NB)]
    with open(os.path.join(ROOT, "muscle.h"), "w") as f:
        f.write("/* muscle.h: written by tools/muscle_envelope.py. Per body, its joint's muscles: strength (largest torque, N m) and\n"
                " * slew (fastest rise of torque, N m/s), from published maximum torques scaled to 45 kg and the capture's inverse dynamics. */\n")
        f.write("static BCONST double MUSCLE_MAX[NB] = {" + ", ".join(repr(v) for v in strength) + "};\n")
        f.write("static BCONST double MUSCLE_SLEW[NB] = {" + ", ".join(repr(v) for v in slew) + "};\n")
    with open(os.path.join(ROOT, "web", "muscle.js"), "w") as f:
        f.write("// muscle.js: written by tools/muscle_envelope.py (the same numbers as muscle.h): per body, its joint's strength (N m) and slew (N m/s).\n")
        f.write("export const MUSCLE_MAX = [" + ", ".join(repr(v) for v in strength) + "];\n")
        f.write("export const MUSCLE_SLEW = [" + ", ".join(repr(v) for v in slew) + "];\n")
    print("\nstrength (N m):", {n: round(strength[i]) for i, n in enumerate(NAMES) if i})
    print("slew (N m/s):   ", {n: round(slew[i]) for i, n in enumerate(NAMES) if i})


if __name__ == "__main__":
    main()
