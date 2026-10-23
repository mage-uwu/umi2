"""room.py: the gym room, written as room.json.

Every feature (foam block, wall, tunnel, slope, stairway) is expanded here into static oriented boxes, and
room.json holds only those boxes: centre c, half extents h, rotation q (w, x, y, z), friction mu, contact
softness (XPBD compliance, m/N: 0 rigid, foam about 3e-4) and a colour. The physics (body.c, the CUDA build,
body.js) collides against exactly these boxes and the viewer draws exactly these boxes, so what is seen is
what is simulated. Numbers are written in Python's shortest round-trip form, so C (strtod), JavaScript
(JSON.parse) and Python read the same doubles.

The floor (y = 0) is not a box; it stays the ground plane of body.c. The spawn area (|x|, |z| < spawn_clear,
where every task starts her, facing +z) is kept free.

    python3 room.py                  (the stage-0 room, room.json)
    python3 room.py --seed 3 --out room_s3.json      (a shuffled variant: features moved and resized)
"""
import argparse, json, math, random

FOAM = ["#7fa8f0", "#f2c75c", "#f08a7f", "#8fd6a8", "#c39ff0"]
COL = {"wall": "#d9dce8", "outer": "#c9cde0", "tunnel": "#6fb7b0", "stair": "#d8b48a", "slope": "#c9a27a", "platform": "#d2ae86"}


def qaxis(ax, ang):
    s = math.sin(ang / 2); n = math.sqrt(sum(a * a for a in ax))
    return [math.cos(ang / 2), ax[0] / n * s, ax[1] / n * s, ax[2] / n * s]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def rot(q, v):   # q v q*
    u = q[1:]; t = [2 * (u[1] * v[2] - u[2] * v[1]), 2 * (u[2] * v[0] - u[0] * v[2]), 2 * (u[0] * v[1] - u[1] * v[0])]
    return [v[i] + q[0] * t[i] + (u[(i + 1) % 3] * t[(i + 2) % 3] - u[(i + 2) % 3] * t[(i + 1) % 3]) for i in range(3)]


class Room:
    def __init__(self): self.boxes = []

    def box(self, kind, c, h, q=(1, 0, 0, 0), mu=0.9, soft=0.0, color=None, feature=None):
        self.boxes.append(dict(kind=kind, feature=feature or kind, c=[float(x) for x in c], h=[float(x) for x in h], q=[float(x) for x in q],
                               mu=float(mu), soft=float(soft), color=color or COL.get(kind, "#cccccc")))

    # local frame of a feature: origin o on the floor, yaw about +y; (u, v, w) = (right, up, forward)
    def place(self, o, yaw, p):
        q = qaxis([0, 1, 0], yaw); d = rot(q, p); return [o[0] + d[0], o[1] + d[1], o[2] + d[2]], q

    def foam(self, o, size, yaw=0.0, color=FOAM[0]):
        c, q = self.place(o, yaw, [0, size[1] / 2, 0])
        self.box("foam", c, [s / 2 for s in size], q, mu=1.0, soft=3e-4, color=color)

    def wall(self, o, length, height, yaw=0.0, thick=0.2, kind="wall"):
        c, q = self.place(o, yaw, [0, height / 2, 0]); self.box(kind, c, [length / 2, height / 2, thick / 2], q, mu=0.6)

    def tunnel(self, o, length, inner_w, clear_h, yaw=0.0, wall=0.15, name="tunnel"):
        """two side walls and a roof along the feature's forward axis; clear_h is the free height inside"""
        for s in (-1, 1):
            c, q = self.place(o, yaw, [s * (inner_w / 2 + wall / 2), clear_h / 2, 0])
            self.box("tunnel", c, [wall / 2, clear_h / 2, length / 2], q, mu=0.6, feature=name)
        c, q = self.place(o, yaw, [0, clear_h + wall / 2, 0])
        self.box("tunnel", c, [inner_w / 2 + wall, wall / 2, length / 2], q, mu=0.6, feature=name)

    def stairs(self, o, steps, rise, run, width, landing, yaw=0.0, name="stairway"):
        """solid steps from the floor up along the forward axis, then a landing at the top"""
        for k in range(1, steps + 1):
            c, q = self.place(o, yaw, [0, k * rise / 2, (k - 0.5) * run])
            self.box("stair", c, [width / 2, k * rise / 2, run / 2], q, mu=0.9, feature=name)
        top = steps * rise
        c, q = self.place(o, yaw, [0, top / 2, steps * run + landing / 2])
        self.box("platform", c, [width / 2, top / 2, landing / 2], q, mu=0.9, feature=name)

    def slope(self, o, angle_deg, length, width, landing, yaw=0.0, thick=0.2, name="slope"):
        """a ramp rising along the forward axis from the floor at o, length measured along the incline, then a landing"""
        a = math.radians(angle_deg); rise = length * math.sin(a); depth = length * math.cos(a)
        # the ramp: a slab whose top face is the incline (its top front edge touches the floor at o)
        qy = qaxis([0, 1, 0], yaw); qr = qmul(qy, qaxis([1, 0, 0], -a))
        mid_top = [0, rise / 2, depth / 2]; nrm = [0, math.cos(a), -math.sin(a)]
        p = [mid_top[i] - nrm[i] * thick / 2 for i in range(3)]
        d = rot(qy, p); self.box("slope", [o[0] + d[0], o[1] + d[1], o[2] + d[2]], [width / 2, thick / 2, length / 2], qr, mu=0.9, feature=name)
        c, q = self.place(o, yaw, [0, rise / 2, depth + landing / 2])
        self.box("platform", c, [width / 2, rise / 2, landing / 2], q, mu=0.9, feature=name)


def problems(boxes, clear=1.8, half=5.9):
    """features overlapping one another (in plan), reaching past the walls, or into the spawn area"""
    def aabb(b):
        cs = [[b["c"][i] + rot(b["q"], [sx * b["h"][0], sy * b["h"][1], sz * b["h"][2]])[i] for i in range(3)] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return [min(c[i] for c in cs) for i in range(3)], [max(c[i] for c in cs) for i in range(3)]
    out = []; inner = [b for b in boxes if b["kind"] != "outer"]; bb = [aabb(b) for b in inner]
    for i in range(len(inner)):
        (l1, h1) = bb[i]
        if max(abs(l1[0]), abs(h1[0]), abs(l1[2]), abs(h1[2])) > half: out.append(f"{inner[i]['feature']} past the walls")
        if l1[0] < clear and h1[0] > -clear and l1[2] < clear and h1[2] > -clear: out.append(f"{inner[i]['feature']} in the spawn area")
        for k in range(i + 1, len(inner)):
            (l2, h2) = bb[k]
            if inner[i]["feature"] != inner[k]["feature"] and all(l1[a] < h2[a] - 1e-6 and l2[a] < h1[a] - 1e-6 for a in (0, 2)):
                out.append(f"{inner[i]['feature']} overlaps {inner[k]['feature']}")
    return out


def stage0(seed=None):
    rng = random.Random(seed); j = (lambda a: rng.uniform(-a, a)) if seed is not None else (lambda a: 0.0)
    R = Room(); half = 6.0
    for k, (o, yaw) in enumerate((([0, 0, half], 0.0), ([0, 0, -half], 0.0), ([half, 0, 0], math.pi / 2), ([-half, 0, 0], math.pi / 2))):
        R.wall(o, 2 * half + 0.2, 2.4, yaw, kind="outer")
    R.stairs([-3.6 + j(.2), 0, 2.4 + j(.2)], steps=6, rise=0.12 + j(.02), run=0.30, width=1.4, landing=1.2, yaw=j(.1), name="stairway")
    R.slope([3.6 + j(.2), 0, 2.4 + j(.2)], angle_deg=18 + j(4), length=2.2, width=1.4, landing=1.0, yaw=j(.1), name="steep slope")
    R.slope([-5.7, 0, -5.15 + j(.05)], angle_deg=8 + j(2), length=2.6, width=1.4, landing=0.8, yaw=math.pi / 2, name="gentle slope")
    R.tunnel([-2.3 + j(.15), 0, -3.3 + j(.15)], length=2.0, inner_w=1.0, clear_h=0.75 + j(.05), yaw=j(.1), name="crawl tunnel")
    R.tunnel([2.5 + j(.15), 0, -3.3 + j(.2)], length=2.2, inner_w=1.0, clear_h=2.05, yaw=j(.1), name="walk tunnel")
    R.wall([0.2 + j(.2), 0, -4.9], 2.4, 1.1, j(.1))
    R.wall([4.9, 0, -0.6 + j(.2)], 2.0, 1.4, math.pi / 2 + j(.1))
    foams = [([2.6, 0, -0.4], [0.9, 0.9, 0.9], .3), ([-2.7, 0, -0.6], [1.0, 0.6, 1.4], -.4), ([1.4, 0, 4.2], [0.7, 0.7, 0.7], .8),
             ([0.0, 0, -2.6], [1.2, 0.35, 0.9], 0.0), ([-1.4, 0, 4.6], [1.6, 0.15, 1.6], 0.0), ([-5.2, 0, 1.2], [0.8, 1.2, 0.8], .5)]
    for k, (o, s, yaw) in enumerate(foams):
        R.foam([o[0] + j(.15), 0, o[2] + j(.15)], [x * (1 + j(.1)) for x in s], yaw + j(.3), FOAM[k % len(FOAM)])
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--out", default="room.json")
    ap.add_argument("--web", default="web/room.json", help="also written here, for the viewer ('' to skip)")
    a = ap.parse_args()
    R = stage0(a.seed); tries = 0
    while problems(R.boxes):   # a variant that does not fit: draw it again (deterministically, from the seed)
        if a.seed is None: raise SystemExit("room: " + "; ".join(problems(R.boxes)))
        tries += 1; R = stage0(a.seed * 1000 + tries)
    doc = {"version": 1, "name": "stage0" if a.seed is None else f"stage0-seed{a.seed}", "floor_y": 0.0, "spawn_clear": 1.8,
           "bounds": [-6.1, 6.1, -6.1, 6.1], "boxes": R.boxes}
    text = ("{\n" + ",\n".join(f'  "{k}": {json.dumps(v)}' for k, v in doc.items() if k != "boxes") + ',\n  "boxes": [\n'
            + ",\n".join("    " + json.dumps(b) for b in R.boxes) + "\n  ]\n}\n")
    for path in [a.out] + ([a.web] if a.web and a.seed is None else []):
        with open(path, "w") as f: f.write(text)
    feats = {}
    for b in R.boxes: feats[b["feature"]] = feats.get(b["feature"], 0) + 1
    print(f"{a.out}: {len(R.boxes)} boxes: " + ", ".join(f"{k} {v}" for k, v in feats.items()))


if __name__ == "__main__":
    main()
