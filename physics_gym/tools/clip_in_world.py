#!/usr/bin/env python3
"""clip_in_world.py: a tracker's clip played in a world (a Rooftops layout, data/ac_course_bank.json), not in the room it
was made for: the clip's start put at (X, Z) on the world's ground facing YAW, every box of the world around her. The
C environment (course mode) wants the room in the clip's own frame (its frame 0 over the origin), so the world's boxes are
moved into it, tools/eval_course.py judges N episodes (EVAL_NOISE as it reads it) and records the first, and the recording
is moved back: OUT.json replays in the world itself (web/index.html, with the world's boxes).

    python3 tools/clip_in_world.py PREFIX CLIP WORLD.json ROOM X Z YAW_DEG TARGET OUT.json [N] [+ROOM_DTOP]

ROOM: the world's room index; TARGET: the name (its feature; or part of it) of the box she is to climb, counted as got up on when her
pelvis is 0.6 m over its top above it. In the clip's frame she faces -z: YAW turns that onto the world (180: she faces +z).
A clip's motion is fixed to its own terrain (its pelvis heights are absolute; the env only raises it by what is under its
start point). +ROOM_DTOP (e.g. +0 or +-0.1): the clip's own room is put into the world with it (its tallest box that much
taller, as a variation; TARGET is then that box, "climbing block").
"""
import json, math, os, subprocess, sys, tempfile

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def main():
    pre, clip, world, room, X, Z, yaw, target, out = sys.argv[1:10]; n = sys.argv[10] if len(sys.argv) > 10 else "32"
    W = json.load(open(world))["rooms"][int(room)]
    own = sys.argv[11] if len(sys.argv) > 11 else ""; t = math.radians(float(yaw)); c, s = math.cos(t), math.sin(t); O = [float(X), 0.0, float(Z)]
    qy = lambda a: [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]   # turned a about +y
    to_w = lambda p: [c * p[0] + s * p[2] + O[0], p[1], -s * p[0] + c * p[2] + O[2]]   # clip frame -> world
    to_c = lambda p: [c * (p[0] - O[0]) - s * (p[2] - O[2]), p[1], s * (p[0] - O[0]) + c * (p[2] - O[2])]   # world -> clip frame
    if own:   # the clip's own room into the world: its boxes (clip frame) moved out to the world, the tallest dtop taller
        mine = [dict(b) for b in [r for r in json.load(open(os.path.join(B, "data/parkour_fit_bank.json")))["rooms"] if r["clip"] == clip][0]["boxes"]]
        k = max(range(len(mine)), key=lambda i: mine[i]["c"][1] + mine[i]["h"][1]); top = mine[k]["c"][1] + mine[k]["h"][1] + float(own[1:])
        mine[k] = dict(mine[k], c=[mine[k]["c"][0], top / 2, mine[k]["c"][2]], h=[mine[k]["h"][0], top / 2, mine[k]["h"][2]], feature="climbing block")
        for i, b in enumerate(mine):
            if i != k: b["feature"] = "climbing step"
        W = dict(W, boxes=W["boxes"] + [dict(b, c=to_w(b["c"]), q=qmul(qy(t), b["q"]), color="#c9a27a") for b in mine])
    boxes = [dict(b, c=to_c(b["c"]), q=qmul(qy(-t), b["q"])) for b in W["boxes"]]
    name = lambda b: str(b.get("feature", b.get("name", "")))
    ti = [i for i, b in enumerate(W["boxes"]) if name(b) == target] or [i for i, b in enumerate(W["boxes"]) if target in name(b)]; assert len(ti) == 1, f"{target}: {len(ti)} boxes"
    clips = os.path.join(B, "data/parkour_fit.bin")
    src = {r["clip"]: r.get("source", "") for r in json.load(open(os.path.join(B, "data/parkour_fit_bank.json")))["rooms"]}
    bank = {"version": 1, "source": f"{W['name']} with clip {clip} at ({X}, {Z}) facing {yaw} deg (tools/clip_in_world.py)", "clips": [clip],
            "rooms": [{"name": W["name"], "clip": clip, "source": src.get(clip, ""), "bounds": [-30, 30, -30, 30], "boxes": boxes, "contacts": [], "target": ti[0]}]}
    with tempfile.TemporaryDirectory() as d:
        bf, rf = os.path.join(d, "bank.json"), os.path.join(d, "rec.json")
        json.dump(bank, open(bf, "w"))
        r = subprocess.run([sys.executable, os.path.join(B, "tools/eval_course.py"), pre, bf, clips, n, "0", "2", "0", rf], capture_output=True, text=True)
        print(r.stdout.strip().replace(bf, "(the world in the clip's frame)")); assert r.returncode == 0, r.stderr[-2000:]
        R = json.load(open(rf))
    fr = []
    for f in R["frames"]:   # each body's centre and rotation, back in the world
        g = []
        for k in range(len(f) // 7): g += [round(v, 5) for v in to_w(f[7 * k:7 * k + 3]) + qmul(qy(t), f[7 * k + 3:7 * k + 7])]
        fr.append(g)
    R.update(frames=fr, ref=[[round(v, 4) for v in to_w(p)] for p in R["ref"]], boxes=W["boxes"], world=W["name"], target=name(W["boxes"][ti[0]]),
             source=f"{R['source']}, in {W['name']} at the {name(W['boxes'][ti[0]])}")
    json.dump(R, open(out, "w"), separators=(",", ":")); print(f"{out}: {len(fr)} frames in the world")


if __name__ == "__main__":
    main()
