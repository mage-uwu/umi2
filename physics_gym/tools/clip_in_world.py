#!/usr/bin/env python3
"""clip_in_world.py: a tracker's clip played in a world (a Rooftops layout, data/ac_course_bank.json), not in the room it
was made for, every box of the world around her. The C environment (course mode) wants the room in the clip's own frame
(its frame 0 over the origin), so the world's boxes are moved into it, tools/eval_course.py judges N episodes
(EVAL_NOISE as it reads it) and records the first, and the recording is moved back: the replay is in the world itself
(web/index.html, with the world's boxes).

    python3 tools/clip_in_world.py PREFIX CLIP WORLD.json ROOM (--at X Z YAW | --align BOX HEADING) [--target BOX]
        [--own DTOP] [--n N] [--clips CLIPS.bin --bank BANK.json] [--replay OUT.json]

--at X Z YAW: the clip's start put at (X, Z) on the world's ground; in the clip's frame she faces -z, YAW turns that onto
  the world (180: she faces +z).
--align BOX HEADING: the clip's obstacle (its own room's: the tallest one not under her start) put on the world's box BOX,
  the near faces of the two together, her way through the clip turned to HEADING (deg: 0 = +z, 90 = +x). The target
  (what she is to get up on, or land on) is then BOX.
--t0 S|auto: each episode from S seconds into the clip (eval_course.py EVAL_T0); auto (with --align): the earliest moment
  from which her way to the obstacle (the clip's pelvis) runs on one surface and through no other box, at least 1/3 s
  before she reaches it. The env raises the clip by what is under its frame-0 point (behind her, with a later start), so
  that is made the height her start needs (the surface under her start less the clip's own there): a 0.1 m post put under
  it if the world is lower there.
--target BOX: the world box she is to get up on (pelvis 0.6 m over its top, above it), by name (feature, or part of it).
--own=DTOP: the clip's own room put into the world with it, its tallest box DTOP taller (TARGET "climbing block").
A clip's motion is fixed to its own terrain: a PARC clip's heights are its terrain's (the env only raises it by what is
under its start point); a CMU clip is on flat ground, raised onto what it starts on.
"""
import argparse, json, math, os, subprocess, sys, tempfile

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(B, 'tools')); sys.path.insert(0, os.path.join(B, 'gpu'))


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def qrot(q, v):
    w, x, y, z = q; u = (x, y, z)
    cr = lambda a, b: (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    t = [2 * c for c in cr(u, v)]; tt = cr(u, t)
    return [v[k] + w * t[k] + tt[k] for k in range(3)]


name = lambda b: str(b.get("feature", b.get("name", "")))


def near_face(b, d):   # the centre of a box's face toward -d in plan (d: unit (x, z), her way)
    ext = sum(abs(qrot(b["q"], ax)[0] * d[0] + qrot(b["q"], ax)[2] * d[1]) * h for ax, h in (((1, 0, 0), b["h"][0]), ((0, 0, 1), b["h"][2])))
    return b["c"][0] - d[0] * ext, b["c"][2] - d[1] * ext


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); A = ap.add_argument
    A("prefix"); A("clip"); A("world"); A("room", type=int)
    A("--at", nargs=3, type=float, metavar=("X", "Z", "YAW")); A("--align", nargs=2, metavar=("BOX", "HEADING"))
    A("--target"); A("--own"); A("--n", type=int, default=32); A("--replay"); A("--t0", default="0")
    A("--clips", default="data/parkour_fit.bin"); A("--bank", default="data/parkour_fit_bank.json")
    a = ap.parse_args()
    W = json.load(open(a.world))["rooms"][a.room]
    find = lambda boxes, t: [i for i, b in enumerate(boxes) if name(b) == t] or [i for i, b in enumerate(boxes) if t in name(b)]
    own_room = [r for r in json.load(open(os.path.join(B, a.bank)))["rooms"] if r["clip"] == a.clip][0]
    import expert_bank as eb, gpu_env as ge
    clip = ge.load_mref(os.path.join(B, a.clips))[a.clip]; r0 = clip["root"][0]
    at = lambda k: clip["root"][min(int(round(k / 30 * clip["fps"])), clip["frames"] - 1)]   # the clip's pelvis at control step k (its own frame)
    rel = lambda k: (at(k)[0] - r0[0], at(k)[1], at(k)[2] - r0[2])
    T = int((clip["frames"] - 1) / clip["fps"] * 30)
    if a.align:   # the clip's obstacle onto the world's box, her way onto HEADING
        bx = own_room["boxes"]; groups = eb.obstacles(bx)
        cand = [g for g in groups if not eb.under_start(bx, g)] or groups
        g = max(cand, key=lambda g: eb.top(bx[g[0]])); d = eb.way(eb.path_of(clip), bx, groups)
        nx = min(near_face(bx[i], d)[0] * d[0] + near_face(bx[i], d)[1] * d[1] for i in g)   # the obstacle's near face, along her way
        kn = next((k for k in range(T + 1) if rel(k)[0] * d[0] + rel(k)[2] * d[1] >= nx - 0.3), T)   # when she reaches it
        across = rel(kn)[0] * -d[1] + rel(kn)[2] * d[0]   # her way's across-coordinate there: the world box's middle goes under it
        cc = (nx * d[0] + across * -d[1], nx * d[1] + across * d[0])   # her way at its near face, in the clip's frame
        wi = find(W["boxes"], a.align[0]); assert len(wi) == 1, f"{a.align[0]}: {len(wi)} boxes"
        H = math.radians(float(a.align[1])); dw = (math.sin(H), math.cos(H)); wc = near_face(W["boxes"][wi[0]], dw)
        t = H - math.atan2(d[0], d[1]); c, s = math.cos(t), math.sin(t)
        O = [wc[0] - (c * cc[0] + s * cc[1]), 0.0, wc[1] - (-s * cc[0] + c * cc[1])]; a.target = a.target or a.align[0]
    else:
        t = math.radians(a.at[2]); c, s = math.cos(t), math.sin(t); O = [a.at[0], 0.0, a.at[1]]
    qy = lambda ang: [math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0]   # turned ang about +y
    to_w = lambda p: [c * p[0] + s * p[2] + O[0], p[1], -s * p[0] + c * p[2] + O[2]]   # clip frame -> world
    to_c = lambda p: [c * (p[0] - O[0]) - s * (p[2] - O[2]), p[1], s * (p[0] - O[0]) + c * (p[2] - O[2])]   # world -> clip frame
    if a.align: assert math.dist(to_w([cc[0], 0, cc[1]])[::2], wc) < 1e-9, "the obstacle's face not on the box's"
    def support(boxes, x, z):   # the highest top under (x, z) in plan (0: the floor)
        tops = [b["c"][1] + b["h"][1] for b in boxes if b["kind"] != "outer" and all(abs(l) < h for l, h in zip(qrot([b["q"][0], -b["q"][1], -b["q"][2], -b["q"][3]], [x - b["c"][0], 0, z - b["c"][2]])[::2], b["h"][::2]))]
        return max(tops, default=0.0)
    k0, pad = 0, None
    if a.t0 == "auto":
        assert a.align, "--t0 auto goes with --align"
        tw = lambda k: to_w(rel(k)); bw = [b for b in W["boxes"] if b["kind"] != "outer" and name(b) != a.align[0]]
        own0 = support(own_room["boxes"], 0, 0)
        def inbox(b, p):
            l = qrot([b["q"][0], -b["q"][1], -b["q"][2], -b["q"][3]], [p[0] - b["c"][0], p[1] - b["c"][1], p[2] - b["c"][2]]); return all(abs(v) < h for v, h in zip(l, b["h"]))
        def ok(k):   # from step k to reaching the obstacle: wherever the clip runs on the surface it starts from, so does the world; no other box in her way
            s0, sk = support(W["boxes"], *tw(k)[::2]), support(own_room["boxes"], rel(k)[0], rel(k)[2]); lift = s0 - sk + own0
            for j in range(k, kn + 1):
                p = tw(j)
                if abs(support(own_room["boxes"], rel(j)[0], rel(j)[2]) - sk) < 0.05 and all(abs(support(W["boxes"], p[0] - e * dw[0], p[2] - e * dw[1]) - s0) > 0.1 for e in (0, 0.15)):   # (0.15 m of slack at an edge)
                    if os.environ.get("CIW_DEBUG"): print(f"t0 {k / 30:.2f}: at {j / 30:.2f} s the world's surface {support(W['boxes'], p[0], p[2]):.2f} m, her start's {s0:.2f}")
                    return False
                hit = [name(b) for b in bw if inbox(b, (p[0], p[1] + lift - 0.5, p[2]))]
                if hit:
                    if os.environ.get("CIW_DEBUG"): print(f"t0 {k / 30:.2f}: at {j / 30:.2f} s through {hit}")
                    return False
            return True
        k0 = next((k for k in range(0, max(1, kn - 10)) if ok(k)), None)   # (at least a third of a second before she reaches it)
        assert k0 is not None, f"no clear run-up to {a.align[0]} within the clip"
    else:
        k0 = int(round(float(a.t0) * 30))
    if k0 > 0:   # the env raises the clip by what is under its frame-0 point: made the height her start needs
        need = support(W["boxes"], *to_w(rel(k0))[::2]) - support(own_room["boxes"], rel(k0)[0], rel(k0)[2]) + support(own_room["boxes"], 0, 0)
        o = to_w((0, 0, 0)); have = support(W["boxes"], o[0], o[2])
        assert have <= need + 0.02, f"the clip's frame-0 point is over {have:.2f} m, her start needs {need:.2f} m"
        if need > have + 0.02: pad = {"kind": "platform", "feature": "post (the clip's height)", "c": [o[0], need / 2, o[2]], "h": [0.05, need / 2, 0.05], "q": [1, 0, 0, 0], "mu": 0.9, "soft": 0, "grip": 0, "color": "#777777"}
    if pad: W = dict(W, boxes=W["boxes"] + [pad])
    if a.own is not None:   # the clip's own room into the world: its boxes (clip frame) moved out to the world, the tallest dtop taller
        mine = [dict(b) for b in own_room["boxes"]]
        k = max(range(len(mine)), key=lambda i: mine[i]["c"][1] + mine[i]["h"][1]); top = mine[k]["c"][1] + mine[k]["h"][1] + float(a.own)
        mine[k] = dict(mine[k], c=[mine[k]["c"][0], top / 2, mine[k]["c"][2]], h=[mine[k]["h"][0], top / 2, mine[k]["h"][2]], feature="climbing block")
        for i, b in enumerate(mine):
            if i != k: b["feature"] = "climbing step"
        W = dict(W, boxes=W["boxes"] + [dict(b, c=to_w(b["c"]), q=qmul(qy(t), b["q"]), color="#c9a27a") for b in mine]); a.target = a.target or "climbing block"
    boxes = [dict(b, c=to_c(b["c"]), q=qmul(qy(-t), b["q"])) for b in W["boxes"]]
    ti = find(W["boxes"], a.target or ""); assert len(ti) == 1, f"{a.target}: {len(ti)} boxes"
    bank = {"version": 1, "source": f"{W['name']} with clip {a.clip} (tools/clip_in_world.py)", "clips": [a.clip],
            "rooms": [{"name": W["name"], "clip": a.clip, "source": own_room.get("source", ""), "bounds": [-30, 30, -30, 30], "boxes": boxes, "contacts": [], "target": ti[0]}]}
    with tempfile.TemporaryDirectory() as d:
        bf, rf = os.path.join(d, "bank.json"), os.path.join(d, "rec.json")
        json.dump(bank, open(bf, "w"))
        r = subprocess.run([sys.executable, os.path.join(B, "tools/eval_course.py"), a.prefix, bf, os.path.join(B, a.clips), str(a.n), "0", "2", "0", rf], capture_output=True, text=True,
                           env=dict(os.environ, EVAL_T0=str(k0 / 30)))
        assert r.returncode == 0, r.stderr[-2000:]
        print(r.stdout.strip().split("\n")[0].replace(bf, "(the world in the clip's frame)") + f" [{W['name']}, {name(W['boxes'][ti[0]])}" + (f", from {k0 / 30:.1f} s into the clip" if k0 else "") + "]")
        R = json.load(open(rf))
    if a.replay:
        fr = []
        for f in R["frames"]:   # each body's centre and rotation, back in the world
            g = []
            for k in range(len(f) // 7): g += [round(v, 5) for v in to_w(f[7 * k:7 * k + 3]) + qmul(qy(t), f[7 * k + 3:7 * k + 7])]
            fr.append(g)
        R.update(frames=fr, ref=[[round(v, 4) for v in to_w(p)] for p in R["ref"]], boxes=W["boxes"], world=W["name"], target=name(W["boxes"][ti[0]]),
                 source=f"{R['source']}, in {W['name']} at the {name(W['boxes'][ti[0]])}")
        json.dump(R, open(a.replay, "w"), separators=(",", ":")); print(f"{a.replay}: {len(fr)} frames in the world")


if __name__ == "__main__":
    main()
