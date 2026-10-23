"""joint_limits.py: natural joint ranges from the motion capture, as a table the physics enforces.

Every frame of the clean library (data/clean/cmu_library.npz: the whole CMU database, 2,435 trials of walking,
running, jumping, dancing, climbing, acrobatics, ..., at 10 Hz; tools/cmu_library.py), or with --dev the development
clips (data/motions_ref.bin: Bandai Namco, CC BY-NC; data/cmu_ref.bin: CMU crawling and getting up), and its
left-right mirror image, gives each joint's rotation (child in parent, as a rotation vector in the parent's rest
frame). Each joint's range per axis is the 0.25 to 99.75 percentile of those, widened by a margin. Elbows and knees
are hinges here, so only their flexion axis is limited (the capture's forearm twist is not a motion this body has).

Writes joint_limits.h (body.c includes it; every build: CPU, CUDA) and web/joint_limits.js (body.js imports it), from
the same numbers. Python's shortest round-trip decimals, so all builds hold identical doubles.

    python3 tools/joint_limits.py [--margin-deg 10] [--dev]
"""
import argparse, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "gpu"))
import gpu_env as ge

NAMES = ["pelvis", "spine", "chest", "head", "l_uarm", "l_farm", "r_uarm", "r_farm", "l_thigh", "l_shin", "l_foot", "r_thigh", "r_shin", "r_foot"]
PARENT = [-1, 0, 1, 2, 2, 4, 2, 6, 0, 8, 9, 0, 11, 12]
MIRROR = [0, 1, 2, 3, 6, 7, 4, 5, 11, 12, 13, 8, 9, 10]   # left <-> right
HINGE = {5: 1, 7: 1, 9: 0, 12: 0}                         # body: its hinge axis (elbows y, knees x), as in body.c BODY_DEFS


def qmul(a, b):
    return np.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3],
                     a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0] + a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2],
                     a[..., 0] * b[..., 2] - a[..., 1] * b[..., 3] + a[..., 2] * b[..., 0] + a[..., 3] * b[..., 1],
                     a[..., 0] * b[..., 3] + a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1] + a[..., 3] * b[..., 0]], -1)


def qlog(q):
    q = np.where(q[..., :1] < 0, -q, q); s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    return np.where(s < 1e-12, 2 * q[..., 1:], q[..., 1:] * (2 * np.arctan2(s, q[..., :1]) / np.maximum(s, 1e-12)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--margin-deg", type=float, default=10.0); ap.add_argument("--pct", type=float, default=0.25)
    ap.add_argument("--dev", action="store_true", help="the development clips instead of the clean CMU library")
    a = ap.parse_args(); margin = math.radians(a.margin_deg)
    if a.dev: Q = np.concatenate([c["q"] for f in ("data/motions_ref.bin", "data/cmu_ref.bin") for c in ge.load_mref(os.path.join(ROOT, f)).values()])
    else:
        Q = np.load(os.path.join(ROOT, "data", "clean", "cmu_library.npz"))["q"].astype(np.float64)
        Q /= np.linalg.norm(Q, axis=-1, keepdims=True)
    src = "the development clips" if a.dev else "the CMU library"
    E = np.stack([qlog(qmul(Q[:, PARENT[i]] * np.array([1, -1, -1, -1]), Q[:, i])) if i else np.zeros((len(Q), 3)) for i in range(14)], 1)
    Em = E[:, MIRROR] * np.array([1, -1, -1])   # the mirror image: rotation vectors (x, y, z) -> (x, -y, -z), left and right swapped
    E2 = np.concatenate([E, Em])
    lo = np.percentile(E2, a.pct, axis=0) - margin; hi = np.percentile(E2, 100 - a.pct, axis=0) + margin
    lo[0] = -np.pi; hi[0] = np.pi                      # the pelvis has no parent
    for i, ax in HINGE.items():                        # hinges: flexion only; the other axes are held by the hinge itself
        for k in range(3):
            if k != ax: lo[i, k], hi[i, k] = -np.pi, np.pi
    lo = np.maximum(lo, -np.pi); hi = np.minimum(hi, np.pi)
    rows = [(NAMES[i], [float(x) for x in lo[i]], [float(x) for x in hi[i]]) for i in range(14)]
    with open(os.path.join(ROOT, "joint_limits.h"), "w") as f:
        f.write(f"/* joint_limits.h: written by tools/joint_limits.py from {len(Q)} capture frames of {src} and their mirror images (margin {a.margin_deg} deg).\n"
                " * Per body, the range of its joint's rotation vector (child in parent, rest frame), lo and hi per axis (rad). */\n")
        f.write("static BCONST double JLIM_LO[NB][3] = {\n" + ",\n".join(f"    {{{lo_[0]!r}, {lo_[1]!r}, {lo_[2]!r}}}   /* {n} */" for n, lo_, _ in rows) + "\n};\n")
        f.write("static BCONST double JLIM_HI[NB][3] = {\n" + ",\n".join(f"    {{{hi_[0]!r}, {hi_[1]!r}, {hi_[2]!r}}}   /* {n} */" for n, _, hi_ in rows) + "\n};\n")
    with open(os.path.join(ROOT, "web", "joint_limits.js"), "w") as f:
        f.write(f"// joint_limits.js: written by tools/joint_limits.py (the same numbers as joint_limits.h): per body, its joint's range\n"
                f"// (rotation vector, child in parent, rest frame), lo and hi per axis (rad).\n")
        f.write("export const JLIM_LO = [\n" + ",\n".join(f"  [{lo_[0]!r}, {lo_[1]!r}, {lo_[2]!r}]" for n, lo_, _ in rows) + "\n];\n")
        f.write("export const JLIM_HI = [\n" + ",\n".join(f"  [{hi_[0]!r}, {hi_[1]!r}, {hi_[2]!r}]" for n, _, hi_ in rows) + "\n];\n")
    for n, l, h in rows[1:]:
        print(f"{n:8s} " + "  ".join(f"{'xyz'[k]}[{math.degrees(l[k]):5.0f},{math.degrees(h[k]):5.0f}]" for k in range(3)))


if __name__ == "__main__":
    main()
