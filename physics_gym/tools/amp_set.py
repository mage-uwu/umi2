#!/usr/bin/env python3
"""amp_set.py: the locomotion reference for the motion discriminator (gpu/ppo_gpu.py --amp), from the 30 Hz CMU library.

Plain walking, running and jogging, starting and stopping, turning, walking sideways and backwards, picked from
data/clean/cmu30_library.{json,npz} (tools/cmu_library.py --fps 30) by their descriptions: stylised walks (stealthy,
"weird", Frankenstein, baby, drunk...), anything with props, seats, stairs, uneven ground, jumps or other people's
hands left out. Each trial is one clip, named c<subject>_<trial>, world rotations and root as the library holds them.
Writes data/cmu_loco.bin (MREF, as mref_load reads it).

Terms, as stated on mocap.cs.cmu.edu: "This data is free for use in research projects. You may include this data
in commercially-sold products, but you may not resell this data directly, even in converted form."

    python3 tools/amp_set.py [--list]
"""
import argparse, json, os, re, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
INC = re.compile(r"walk|run|jog|turn|veer|stop|start|forward")
EXC = re.compile(r"sit|climb|ladder|stair|carry|swing|playground|dance|jump|hop|uneven|slope|box|bench|lie |lying|crawl|kick|punch|ball|"
                 r"soccer|drink|wash|vignette|mime|zombie|drunk|injur|limp|stool|chair|crouch|sneak|wipe|window|lift|push|pull|throw|catch|"
                 r"fight|duck|bend|stretch|yawn|cartwheel|flip|roll|skip|leap|gymnast|salsa|story|pick|wheel|monkey|dog|animal|bear|chicken|"
                 r"elephant|cat |stealth|franken|mickey|weird|baby|pregnant|range of motion|scared|shy|achey|macho|toes|arms out|bouncy|"
                 r"high-five|scramble|basketball|golf|tai chi|yoga|exercise|boxing|sword|martial|bow|salute|mop|sweep|paint|clean|dribble|"
                 r"march|pump|spin|squat|get up|getting up|fall|stagger|stumble|trip|twist|rotate|wave|point|gesture|talk|conversation|"
                 r"shake|clap|cheer|celebrate|hand|arm|leg|elbow|shoulder|head|body|motorcycle|bicycle|steering|drive|swim|fly|row")


def pick(trials):
    return [k for k, t in enumerate(trials) if INC.search(t["desc"].lower()) and not EXC.search((t["desc"] + " " + t["group"]).lower())]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--out", default=os.path.join(ROOT, "data", "cmu_loco.bin"))
    a = ap.parse_args()
    lib = json.load(open(os.path.join(ROOT, "data", "clean", "cmu30_library.json"))); T = lib["trials"]
    ks = pick(T)
    if a.list:
        for k in ks: print(T[k]["id"], T[k]["desc"], T[k]["frames"])
        return
    z = np.load(os.path.join(ROOT, "data", "clean", "cmu30_library.npz")); tri = z["trial"]
    clips = []
    for k in ks:
        idx = np.nonzero(tri == k)[0]
        if len(idx) < 8: continue
        q = z["q"][idx].astype(np.float32); root = z["root"][idx].astype(np.float32)
        # frames the capture lost (root and rotations all zero) out, and a clip cut where the root jumps (over 8 m/s)
        ok = (np.abs(root).sum(1) > 0) & (np.linalg.norm(q, axis=-1).min(1) > 0.5)
        q, root = q[ok], root[ok]; q /= np.linalg.norm(q, axis=-1, keepdims=True)
        cut = np.nonzero(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1) * lib["fps"] > 8)[0] + 1
        for part, (s0, s1) in enumerate(zip(np.r_[0, cut], np.r_[cut, len(root)])):
            if s1 - s0 < 8: continue
            qq = q[s0:s1].copy()
            for f in range(1, len(qq)):   # each quaternion on the side of the previous frame's (no sign flips between frames)
                s = np.sign(np.sum(qq[f] * qq[f - 1], -1, keepdims=True)); s[s == 0] = 1; qq[f] *= s
            clips.append(("c" + T[k]["id"] + ("" if not part else chr(96 + part)), root[s0:s1], qq))
    with open(a.out, "wb") as f:
        f.write(b"MREF"); f.write(struct.pack("<i", len(clips)))
        for name, root, q in clips:
            f.write(name.encode().ljust(16, b"\0")[:16]); f.write(struct.pack("<if", len(root), float(lib["fps"])))
            f.write(np.concatenate([root, q.reshape(len(q), -1)], 1).astype(np.float32).tobytes())
    n = sum(len(r) for _, r, _ in clips)
    print(f"{a.out}: {len(clips)} clips, {n} frames ({n / lib['fps'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
