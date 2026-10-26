#!/usr/bin/env python3
"""feasibility.py: can her body do a clip? A check of the reference motion alone, before any training.
    python3 tools/feasibility.py CLIPS.bin COURSES.json [--strength M] [--json OUT]
Her bodies (body.c BODY_DEFS: 45 kg, box per segment) posed by the clip's world rotations from its root (forward
kinematics as env.c), placed in the clip's room as course mode does (frame 0 over the origin, raised onto what is under
it). Per clip:
  gap      at frames PARC labels a foot (or hand) in contact: how far its lowest corner is above (+) or into (-) the
           terrain under it, median and worst (retargeting onto her limb lengths can leave feet off or inside the boxes);
  sink     the deepest any body goes into the terrain or the floor at any frame;
  flight   in flight (no contact for 3+ frames): the whole-body acceleration's distance from free fall, in g (a clip
           physics cannot follow without outside help shows here);
  strength the whole-body support the motion needs (mass x (acceleration + g)), shared by the limbs in contact, times
           each joint's horizontal lever to the contact, over her muscle there (muscle.h: elbow 37.5, shoulder 52.5,
           knee 101, hip 125 N m): 90th percentile over contact frames, > 1 beyond her (arms: the elbows and shoulders).
Clips without contact labels (the CMU climbs) take contact as a foot within 4 cm of a surface.
A clip passes when |gap| median < 0.06 m, worst sink > -0.10 m, flight < 0.35 g and strength / M <= 1.2, M her muscles'
multiplier (TaskCfg.strength). The 1.2 is from what she has and has not done in the simulator at x1: the standing jumps she
lands unaided score 0.6 to 0.7 and walking up to 1.1; the running jumps no run has learned, 1.4 to 2.1. (A static
estimate: it reads about 1 for plain walking.)"""
import json, sys, numpy as np

# body.c BODY_DEFS: parent, anchor, centre, half extents, mass fraction
BD = [(-1, (0, .928, .004), (0, .915, .005), (.12, .07, .09), .120), (0, (0, .980, .016), (0, 1.036, .017), (.11, .055, .08), .140),
      (1, (0, 1.093, .019), (0, 1.205, 0), (.13, .11, .09), .170), (2, (0, 1.332, -.034), (0, 1.450, -.010), (.08, .115, .09), .070),
      (2, (.105, 1.293, -.025), (.215, 1.293, -.025), (.11, .035, .035), .026), (4, (.324, 1.293, -.025), (.472, 1.293, -.025), (.148, .03, .03), .022),
      (2, (-.105, 1.293, -.025), (-.215, 1.293, -.025), (.11, .035, .035), .026), (6, (-.324, 1.293, -.025), (-.472, 1.293, -.025), (.148, .03, .03), .022),
      (0, (.076, .888, 0), (.076, .709, -.004), (.06, .18, .06), .146), (8, (.076, .529, -.008), (.076, .319, -.020), (.045, .21, .045), .048),
      (9, (.076, .109, -.033), (.076, .035, .035), (.04, .035, .105), .013), (0, (-.076, .888, 0), (-.076, .709, -.004), (.06, .18, .06), .146),
      (11, (-.076, .529, -.008), (-.076, .319, -.020), (.045, .21, .045), .048), (12, (-.076, .109, -.033), (-.076, .035, .035), (.04, .035, .105), .013)]
M = 45.0; G = 9.81; MF = np.array([b[4] for b in BD]); MF = MF / MF.sum()
L_FARM, R_FARM, L_FOOT, R_FOOT = 5, 7, 10, 13
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)

def qrot(q, v):   # q (..., 4) w x y z, v (..., 3)
    w, u = q[..., :1], q[..., 1:]; t = 2 * np.cross(u, v); return v + w * t + np.cross(u, t)

def fk(root, Q):   # body centres and the 8 corners of each box, (T, 14, 3) and (T, 14, 8, 3)
    T = len(root); C = np.zeros((T, 14, 3)); C[:, 0] = root
    for i in range(1, 14):
        p, an, ce, _, _ = BD[i]; pc = np.array(BD[p][2]); an = np.array(an); ce = np.array(ce)
        C[:, i] = C[:, p] + qrot(Q[:, p], np.broadcast_to(an - pc, (T, 3))) + qrot(Q[:, i], np.broadcast_to(ce - an, (T, 3)))
    K = np.zeros((T, 14, 8, 3))
    for i in range(14): K[:, i] = C[:, i, None] + qrot(np.repeat(Q[:, i, None], 8, 1), SIGNS * np.array(BD[i][3]))
    return C, K

def top_under(boxes, x, z, below=9.0):   # the highest surface under (x, z) not above `below`: box tops, or the floor at 0
    t = np.zeros_like(x)
    for b in boxes:
        c, h, q = np.array(b['c']), np.array(b['h']), np.array(b['q'], float)
        d = np.stack([x - c[0], np.zeros_like(x), z - c[2]], -1); l = qrot(np.broadcast_to(q * [1, -1, -1, -1], d.shape[:-1] + (4,)), d)
        inside = (np.abs(l[..., 0]) < h[0]) & (np.abs(l[..., 2]) < h[2]); top = c[1] + h[1]
        t = np.where(inside & (top <= below + 1e-6), np.maximum(t, top), t)
    return t

def smooth(a, k=7):   # moving average along time, edges held
    p = np.pad(a, [(k // 2, k // 2)] + [(0, 0)] * (a.ndim - 1), mode='edge'); w = np.ones(k) / k
    return np.apply_along_axis(lambda s: np.convolve(s, w, 'valid'), 0, p)

MUSCLE = {'elbow': 37.5, 'shoulder': 52.5, 'knee': 101.2, 'hip': 125.15}   # muscle.h, N m
def anchor(C, Q, i):   # world position of body i's joint with its parent
    return C[:, i] - qrot(Q[:, i], np.broadcast_to(np.array(BD[i][2]) - np.array(BD[i][1]), (len(C), 3)))

def check(clip, room, mult=1.0):
    root = clip['root'].astype(float); Q = np.asarray(clip['q'], float).reshape(len(root), 14, 4); fps = clip['fps']
    boxes = room['boxes']; lift = top_under(boxes, np.zeros(1), np.zeros(1))[0]
    root = root - [root[0, 0], 0, root[0, 2]] + [0, lift, 0]
    C, K = fk(root, Q); T = len(root)
    low = K[..., 1].min(-1)                                                   # lowest corner of each body, (T, 14)
    ground = np.stack([top_under(boxes, C[:, i, 0], C[:, i, 2], low[:, i].max() + 0.5) for i in range(14)], 1)
    gapb = low - ground                                                       # each body's lowest corner over the surface under it
    lab = room.get('contacts')
    if lab is not None:
        lab = np.array(lab[:T]); feet = [(lab >> 0) & 1, (lab >> 1) & 1]; hands = [(lab >> 2) & 1, (lab >> 3) & 1]
    else:   # no labels: a foot within 4 cm of the lowest it gets over the surface under it
        feet = [(gapb[:, f] < gapb[:, f].min() + 0.04).astype(int) for f in (L_FOOT, R_FOOT)]; hands = [np.zeros(T, int)] * 2
    gaps = np.concatenate([gapb[feet[0] > 0, L_FOOT], gapb[feet[1] > 0, R_FOOT], gapb[hands[0] > 0, L_FARM], gapb[hands[1] > 0, R_FARM]])
    com = (C * MF[None, :, None]).sum(1); a = np.gradient(np.gradient(smooth(com, 5), 1 / fps, axis=0), 1 / fps, axis=0)
    F = M * (a + [0, G, 0])                                                   # supporting force needed, N
    anyc = (feet[0] | feet[1] | hands[0] | hands[1]) > 0
    fl = np.array([0 < s < T - 1 and not anyc[max(0, s - 2):s + 3].any() for s in range(T)])   # inside 5+ frame flights
    flight = float(np.median(np.linalg.norm(a[fl] + [0, G, 0], axis=1)) / G) if fl.sum() >= 2 else 0.0
    # strength: the vertical support shared by the limbs in contact; each limb's joints hold it over the horizontal lever
    # from joint to contact (a static estimate: limb inertia left out; a straight limb under the load needs little)
    n = sum(x.astype(float) for x in feet + hands); Fy = np.where(n > 0, np.maximum(F[:, 1], 0) / np.maximum(n, 1), 0)
    hand_pt = lambda i, sx: C[:, i] + qrot(Q[:, i], np.broadcast_to([sx * 0.148, 0, 0], (T, 3)))
    limbs = [(hands[0], hand_pt(L_FARM, 1), [(anchor(C, Q, L_FARM), 'elbow'), (anchor(C, Q, 4), 'shoulder')]),
             (hands[1], hand_pt(R_FARM, -1), [(anchor(C, Q, R_FARM), 'elbow'), (anchor(C, Q, 6), 'shoulder')]),
             (feet[0], C[:, L_FOOT], [(anchor(C, Q, 9), 'knee'), (anchor(C, Q, 8), 'hip')]),
             (feet[1], C[:, R_FOOT], [(anchor(C, Q, 12), 'knee'), (anchor(C, Q, 11), 'hip')])]
    need = np.zeros(T); arm = np.zeros(T)
    for on, pt, joints in limbs:
        for jp, name in joints:
            r = Fy * np.hypot(*(pt - jp)[:, [0, 2]].T) / MUSCLE[name] * on
            need = np.maximum(need, r)
            if name in ('elbow', 'shoulder'): arm = np.maximum(arm, r)
    pct = lambda x: float(np.percentile(x[x > 0], 90)) if (x > 0).sum() >= 3 else 0.0
    r = dict(gap_med=float(np.median(np.abs(gaps))) if len(gaps) else 0.0, gap_worst=float(gaps.min()) if len(gaps) else 0.0,
             sink=float(gapb.min()), flight=flight, strength=pct(need), arms=pct(arm),
             speed=float(np.linalg.norm(np.gradient(root[:, [0, 2]], 1 / fps, axis=0), axis=1).max()))
    r['pass'] = bool(r['gap_med'] < 0.06 and r['sink'] > -0.10 and r['flight'] < 0.35 and r['strength'] / mult <= 1.2)
    return r

if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/gpu'); import gpu_env as ge
    clips = ge.load_mref(sys.argv[1]); doc = json.load(open(sys.argv[2])); out = {}
    mult = float(sys.argv[sys.argv.index('--strength') + 1]) if '--strength' in sys.argv else 1.0
    for room in doc['rooms']:
        k = room['clip']; r = check(clips[k], room, mult); out[k] = dict(r, source=room.get('source', k))
        print(f"{k:12s} {'PASS' if r['pass'] else 'fail'}  gap {r['gap_med']:.3f} (worst {r['gap_worst']:+.2f})  sink {r['sink']:+.2f}  flight {r['flight']:.2f} g  "
              f"strength {r['strength']:.2f} (arms {r['arms']:.2f})  speed {r['speed']:.1f}  {room.get('source', '')[:40]}")
    n = sum(v['pass'] for v in out.values()); print(f"{n} of {len(out)} pass at strength x{mult:g}")
    if '--json' in sys.argv: json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
