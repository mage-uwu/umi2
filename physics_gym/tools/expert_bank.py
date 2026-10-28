#!/usr/bin/env python3
"""expert_bank.py: the room banks of the Rooftops move experts (data/experts/rooftops.json): for each, variants of its clip's
own room (data/experts/<name>_train.json, <name>_heldout.json), each a room of env.c's bank (ROOM_BANK 64 at most), all
built round the one clip. Seeded: the same manifest gives the same banks.

    python3 tools/expert_bank.py [MANIFEST] [NAME ...]

A room's boxes fall into obstacles: boxes of one top height that touch (PARC builds a platform from several). Per variant:
  lift   all boxes raised together, by U(lift) (the env raises the clip with what is under its start point, so a CMU clip on
         flat ground, a jump from platform to platform, stays on them; a PARC clip's heights are its terrain's own);
  dtop   each obstacle not under her start: its top by U(-dtop, dtop) (or U(dtop) for a range);
  shift  each such obstacle along her way and across it, by U(-a, a), U(-c, c);
  depth  each such obstacle's extent along her way U(depth) (its near face kept: a vault box thinned to a wall);
  boxes  U(boxes) unrelated boxes (0.3-1.5 m wide, 0.3-2.6 m tall) kept 0.9 m off her path and 0.3 m off the obstacles.
Each room's "target" is the obstacle she is to get onto, over or across to (the tallest not under her start; its largest
box): tools/eval_course.py counts an episode as reaching it when her pelvis is 0.6 m over its top, above it.
Room 0 of the training bank is the clip's own room as it is (and stays in: she keeps what she has). Her way is the clip's
pelvis path; when it ends within 1 m of its start (a climb up and down), the direction to the tallest obstacle.
"""
import json, math, os, random, sys

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(B, 'gpu'))
import gpu_env as ge


def top(b): return b["c"][1] + b["h"][1]


def obstacles(boxes):
    """Boxes of one top height that touch or overlap in plan: one obstacle (a list of box indices)."""
    idx = [i for i, b in enumerate(boxes) if b["kind"] != "outer"]; seen, groups = set(), []
    near = lambda a, b: abs(top(a) - top(b)) < 0.01 and all(abs(a["c"][k] - b["c"][k]) <= a["h"][k] + b["h"][k] + 0.01 for k in (0, 2))
    for i in idx:
        if i in seen: continue
        g, stack = [], [i]; seen.add(i)
        while stack:
            j = stack.pop(); g.append(j)
            for k in idx:
                if k not in seen and near(boxes[j], boxes[k]): seen.add(k); stack.append(k)
        groups.append(sorted(g))
    return groups


def u(rng, v):   # a number from a spec: x (U(-x, x)) or [lo, hi]
    return rng.uniform(*v) if isinstance(v, list) else rng.uniform(-v, v)


def path_of(clip):
    """The clip's pelvis path in its room (frame 0 over the origin), every 0.1 s, (x, z)."""
    r = clip["root"]; x0, z0 = r[0][0], r[0][2]; step = max(1, int(clip["fps"] / 10))
    return [(p[0] - x0, p[2] - z0) for p in r[::step]] + [(r[-1][0] - x0, r[-1][2] - z0)]


def way(path, boxes, groups):
    dx, dz = path[-1][0] - path[0][0], path[-1][1] - path[0][1]
    if math.hypot(dx, dz) < 1.0:   # up and back down: toward the tallest obstacle
        g = max(groups, key=lambda g: top(boxes[g[0]])); dx = sum(boxes[i]["c"][0] for i in g) / len(g); dz = sum(boxes[i]["c"][2] for i in g) / len(g)
    n = math.hypot(dx, dz) or 1.0
    return dx / n, dz / n


def under_start(boxes, g):   # an obstacle she starts on: within 0.35 m of the origin (her pelvis at frame 0) in plan (p4 starts her on its step's edge)
    return any(abs(boxes[i]["c"][0]) <= boxes[i]["h"][0] + 0.35 and abs(boxes[i]["c"][2]) <= boxes[i]["h"][2] + 0.35 for i in g)


def variant(room, clip, spec, rng):
    boxes = [json.loads(json.dumps(b)) for b in room["boxes"]]; groups = obstacles(boxes); path = path_of(clip)
    wx, wz = way(path, boxes, groups); sx, sz = -wz, wx   # along her way, and across it
    lift = u(rng, spec["lift"])
    for g in groups:
        start = under_start(boxes, g)
        dt = 0.0 if start else u(rng, spec["dtop"])
        a, c = (0.0, 0.0) if start else (u(rng, spec["shift"][0]), u(rng, spec["shift"][1]))
        dep = None if start or not spec.get("depth") else rng.uniform(*spec["depth"])
        for i in g:
            b = boxes[i]
            if dep is not None:   # its extent along her way (PARC's boxes are axis-aligned; a vault's obstacle is one box): its far side cut back
                ax = 0 if abs(wx) > abs(wz) else 2; sgn = 1 if (wx if ax == 0 else wz) > 0 else -1   # her way along x or z, and which way
                near = b["c"][ax] - b["h"][ax] if sgn > 0 else b["c"][ax] + b["h"][ax]
                far = near + sgn * min(dep, 2 * b["h"][ax]); b["c"][ax] = (near + far) / 2; b["h"][ax] = abs(far - near) / 2
            t = top(b) + dt + lift; b["h"][1] = max(0.02, t / 2); b["c"][1] = t / 2
            b["c"][0] += a * wx + c * sx; b["c"][2] += a * wz + c * sz
    n, placed = rng.randint(*spec["boxes"]), 0
    for _ in range(200):   # unrelated boxes, off her path and the obstacles
        if placed >= n: break
        hx, hz, ht = rng.uniform(0.15, 0.75), rng.uniform(0.15, 0.75), rng.uniform(0.3, 2.6)
        px, pz = rng.choice(path); r = rng.uniform(1.0, 5.0); a = rng.uniform(0, 2 * math.pi); cx, cz = px + r * math.cos(a), pz + r * math.sin(a)
        gap = lambda x, z, mx, mz: math.hypot(max(0.0, abs(x - cx) - mx), max(0.0, abs(z - cz) - mz))
        if min(gap(x, z, hx, hz) for x, z in path) < 0.9: continue
        if any(gap(b["c"][0], b["c"][2], hx + b["h"][0], hz + b["h"][2]) < 0.3 for b in boxes if b["kind"] != "outer"): continue
        boxes.append({"kind": "platform", "feature": "unrelated box", "c": [cx, ht / 2, cz], "h": [hx, ht / 2, hz], "q": [1, 0, 0, 0], "mu": 0.9, "soft": 0, "grip": 1, "color": "#9a8f84"})
        placed += 1
    return boxes


def build(spec, out_dir):
    bank = json.load(open(os.path.join(B, spec["bank"]))); room = [r for r in bank["rooms"] if r["clip"] == spec["clip"]][0]
    clip = ge.load_mref(os.path.join(B, spec["clips"]))[spec["clip"]]
    res = {}
    for part, n, seed in (("train", spec["variants"], 1), ("heldout", spec["heldout"], 2)):
        rng = random.Random(f"{spec['name']}/{part}/{seed}"); rooms = []
        for k in range(n):
            boxes = room["boxes"] if part == "train" and k == 0 else variant(room, clip, spec, rng)
            gs = obstacles(boxes); gs = [g for g in gs if not any(boxes[i].get("feature") == "unrelated box" for i in g)]
            main = max([g for g in gs if not under_start(boxes, g)] or gs, key=lambda g: top(boxes[g[0]]))   # what she is to get over, onto or across to
            rooms.append({"name": f"{spec['name']} {part} {k}", "clip": spec["clip"], "source": room.get("source", ""), "bounds": room.get("bounds", [-8, 8, -8, 8]), "boxes": boxes, "contacts": [],
                          "target": max(main, key=lambda i: boxes[i]["h"][0] * boxes[i]["h"][2])})
        assert len(rooms) <= 64, "env.c holds 64 rooms (ROOM_BANK)"
        f = os.path.join(out_dir, f"{spec['name']}_{part}.json")
        json.dump({"version": 1, "source": f"tools/expert_bank.py: {spec['name']} ({spec['move']}), {part}: variants of {spec['clip']}'s room", "clips": [spec["clip"]], "rooms": rooms}, open(f, "w"), separators=(",", ":"))
        tops = sorted({round(top(b), 2) for r in rooms for b in r["boxes"] if b.get("feature") != "unrelated box" and b["kind"] != "outer"})
        res[part] = (f, len(rooms), tops[0] if tops else None, tops[-1] if tops else None)
    return res


def experts(manifest):
    M = json.load(open(manifest))
    return [dict(M["defaults"], **e) for e in M["experts"]]


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else os.path.join(B, "data/experts/rooftops.json")
    names = [a for a in sys.argv[1:] if not a.endswith(".json")]
    for spec in experts(manifest):
        if names and spec["name"] not in names: continue
        r = build(spec, os.path.dirname(manifest))
        print(f"{spec['name']:12s} {spec['clip']:10s} " + "  ".join(f"{p}: {n} rooms, obstacle tops {lo}-{hi} m" for p, (f, n, lo, hi) in r.items()))


if __name__ == "__main__":
    main()
