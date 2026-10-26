#!/usr/bin/env python3
"""cmu_fetch.py: named CMU trials (mocap.cs.cmu.edu) as an MREF clip file at 30 Hz, one clip per trial (or per frame range).

    python3 tools/cmu_fetch.py out.bin 127_26 40_06:300-420 81_14:255-510:rev ...
        (id; id:first-last, frames at 30 Hz; :rev plays the range backwards, as climbing down a ledge played back climbs up)

Converted with cmu_to_ref.convert (the character's 14 body rotations, root scaled to her hip height), as
tools/cmu_library.py does. Clips are named c<id> (with a frame range: c<id>_<first>; reversed: c<id>r).

Terms, as stated on mocap.cs.cmu.edu: "This data is free for use in research projects. You may include this data in
commercially-sold products, but you may not resell this data directly, even in converted form."
"""
import os, struct, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cmu_to_ref as c2r
from cmu_library import get, SITE


def fetch(tid, tmp):
    s = tid.split("_")[0]; asf = os.path.join(tmp, s + ".asf"); amc = os.path.join(tmp, tid + ".amc")
    if not os.path.exists(asf) and not get(f"{SITE}/subjects/{s}/{s}.asf", asf): raise RuntimeError("no asf " + s)
    if not get(f"{SITE}/subjects/{s}/{tid}.amc", amc): raise RuntimeError("no amc " + tid)
    frames, fps, scale = c2r.convert(asf, amc, step=4)   # 120 Hz -> 30 Hz
    q = np.array([[qq for qq in f[1]] for f in frames], np.float32); r = np.array([f[0] for f in frames], np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for f in range(1, len(q)):   # each quaternion on the side of the previous frame's
        sg = np.sign(np.sum(q[f] * q[f - 1], -1, keepdims=True)); sg[sg == 0] = 1; q[f] *= sg
    return r, q


def main():
    out, specs = sys.argv[1], sys.argv[2:]; clips = []
    with tempfile.TemporaryDirectory() as tmp:
        for sp in specs:
            tid, _, rng = sp.partition(":"); rng, _, flag = rng.partition(":"); r, q = fetch(tid, tmp); name = "c" + tid
            if rng:
                a, b = map(int, rng.split("-")); r, q = r[a:b + 1], q[a:b + 1]; name += f"_{a}"
            if flag == "rev": r, q = r[::-1].copy(), q[::-1].copy(); name = "c" + tid + "r"
            clips.append((name, r, q)); print(f"{name}: {len(r)} frames", flush=True)
    with open(out, "wb") as f:
        f.write(b"MREF"); f.write(struct.pack("<i", len(clips)))
        for name, r, q in clips:
            f.write(name.encode().ljust(16, b"\0")[:16]); f.write(struct.pack("<if", len(r), 30.0))
            f.write(np.concatenate([r, q.reshape(len(q), -1)], 1).astype(np.float32).tobytes())


if __name__ == "__main__":
    main()
