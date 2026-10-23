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
COL = {"wall": "#d9dce8", "outer": "#c9cde0", "tunnel": "#6fb7b0", "stair": "#d8b48a", "slope": "#c9a27a", "platform": "#d2ae86",
       "climb": "#b9a7d6", "hold": "#f2884b", "ledge": "#9fb8c9", "parkour": "#a9c99f"}
# grip labels (room.json v2), for the hands: 0 nothing to hold, 1 a ledge (its top edges can be held), 2 a handhold (all of it)
NONE, LEDGE, HOLD = 0, 1, 2


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

    def box(self, kind, c, h, q=(1, 0, 0, 0), mu=0.9, soft=0.0, color=None, feature=None, grip=NONE):
        self.boxes.append(dict(kind=kind, feature=feature or kind, c=[float(x) for x in c], h=[float(x) for x in h], q=[float(x) for x in q],
                               mu=float(mu), soft=float(soft), grip=int(grip), color=color or COL.get(kind, "#cccccc")))

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


    def block(self, o, size, yaw=0.0, kind="parkour", name=None, grip=LEDGE, mu=0.9):
        """a solid block standing on the floor (size: width, height, depth); by default its top edges can be held"""
        c, q = self.place(o, yaw, [0, size[1] / 2, 0]); self.box(kind, c, [s / 2 for s in size], q, mu=mu, feature=name or kind, grip=grip)

    def climbing_wall(self, o, width, height, yaw=0.0, thick=0.3, rng=None, name="climbing wall", spacing=0.42):
        """a wall facing -w (toward the room) with a grid of handholds on that face, jittered, and a holdable top edge"""
        rng = rng or random.Random(0)
        c, q = self.place(o, yaw, [0, height / 2, 0])
        self.box("climb", c, [width / 2, height / 2, thick / 2], q, mu=0.9, feature=name, grip=LEDGE)
        nu = int((width - 0.3) / spacing); nv = int((height - 0.35) / spacing)
        for i in range(nu + 1):
            for k in range(nv + 1):
                u = -width / 2 + 0.15 + i * (width - 0.3) / max(nu, 1) + rng.uniform(-0.08, 0.08)
                v = 0.3 + k * (height - 0.45) / max(nv, 1) + rng.uniform(-0.08, 0.08)
                hh = [rng.uniform(0.06, 0.10), rng.uniform(0.03, 0.05), 0.04]   # a small jug: wide, shallow, standing 8 cm proud
                hc, hq = self.place(o, yaw, [u, v, -thick / 2 - hh[2]])
                self.box("hold", hc, hh, hq, mu=1.2, feature=name, grip=HOLD)


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


def stage1(seed=None):
    """stage 1 (16 x 16 m): stage 0's features in the middle, and around them the climbing and parkour band:
    a climbing wall with handholds and a mantle block (north), a parkour lane of platforms, gaps, a vault block and a
    drop (east), stepping stones (south). Ledges and holds carry grip labels for the hands."""
    rng = random.Random(seed if seed is not None else 1); j = (lambda a: rng.uniform(-a, a)) if seed is not None else (lambda a: 0.0)
    R = stage0(seed); R.boxes = [b for b in R.boxes if b["kind"] != "outer"]; half = 8.0
    for o, yaw in (([0, 0, half], 0.0), ([0, 0, -half], 0.0), ([half, 0, 0], math.pi / 2), ([-half, 0, 0], math.pi / 2)):
        R.wall(o, 2 * half + 0.2, 4.0, yaw, kind="outer")
    # north: a climbing wall 3.6 m high against the outer wall, facing the room; a mantle block with ledges at hang height
    R.climbing_wall([-3.4 + j(.2), 0, half - 0.35], width=4.4, height=3.6 + j(.2), yaw=math.pi, rng=rng)
    R.block([1.6 + j(.2), 0, half - 0.75], [2.0, 1.55 + j(.15), 1.0], name="mantle block")
    R.block([4.4 + j(.2), 0, half - 0.75], [1.8, 2.1 + j(.1), 1.0], name="high ledge")
    # east: the parkour lane, south to north along x ~ 7: platforms 0.6 m up with widening gaps, a vault block, a drop
    x = half - 0.9 + j(.05); z = -half + 1.0; lane = "parkour lane"
    R.block([x, 0, z + 0.6], [1.2, 0.3, 1.2], name=lane)                     # a step up
    z += 1.2
    for gap, h in ((0.5, 0.6), (0.8, 0.6), (1.1, 0.6), (0.7, 0.9)):         # jumps between platforms
        z += gap + j(.05); R.block([x, 0, z + 0.6], [1.2, h + j(.03), 1.2], name=lane); z += 1.2
    z += 1.4; R.block([x, 0, z + 0.2], [1.4, 1.0 + j(.05), 0.4], name="vault block")   # vault it, or climb over
    z += 1.6; R.block([x, 0, z + 0.7], [1.2, 1.25 + j(.05), 1.4], name="drop platform")   # climb up, drop off the far side
    # south: stepping stones, low blocks with gaps to step or hop across
    for k in range(6):
        R.block([-5.0 + 1.0 * k + j(.1), 0, -half + 1.1 + (0.35 if k % 2 else -0.15) + j(.1)], [0.5, 0.18 + 0.06 * (k % 3), 0.5], name="stepping stones", grip=NONE)
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--out", default="room.json"); ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--web", default="web/room.json", help="also written here, for the viewer ('' to skip)")
    a = ap.parse_args()
    build = stage1 if a.stage == 1 else stage0; half = 8.0 if a.stage == 1 else 6.0
    R = build(a.seed); tries = 0
    while problems(R.boxes, half=half - 0.1):   # a variant that does not fit: draw it again (deterministically, from the seed)
        if a.seed is None: raise SystemExit("room: " + "; ".join(problems(R.boxes, half=half - 0.1)))
        tries += 1; R = build(a.seed * 1000 + tries)
    nm = f"stage{a.stage}"; b = half + 0.1
    doc = {"version": 2, "name": nm if a.seed is None else f"{nm}-seed{a.seed}", "floor_y": 0.0, "spawn_clear": 1.8,
           "bounds": [-b, b, -b, b], "grip": {"0": "none", "1": "ledge: its top edges can be held", "2": "handhold"}, "boxes": R.boxes}
    text = ("{\n" + ",\n".join(f'  "{k}": {json.dumps(v)}' for k, v in doc.items() if k != "boxes") + ',\n  "boxes": [\n'
            + ",\n".join("    " + json.dumps(b) for b in R.boxes) + "\n  ]\n}\n")
    for path in [a.out] + ([a.web] if a.web and a.seed is None else []):
        with open(path, "w") as f: f.write(text)
    feats = {}
    for b in R.boxes: feats[b["feature"]] = feats.get(b["feature"], 0) + 1
    print(f"{a.out}: {len(R.boxes)} boxes: " + ", ".join(f"{k} {v}" for k, v in feats.items()))


if __name__ == "__main__":
    main()
