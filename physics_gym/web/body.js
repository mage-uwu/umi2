// body.js: the browser twin of body.c / env.c / policy.c, so a policy trained in C drives the same physics here.
// Units: metres, kilograms, seconds. The model faces +Z, her left is +X. Body frames are world-aligned in the
// rest T-pose, so a body's world rotation is the bone's rotation relative to rest (as the shader page uses).

import { JLIM_LO, JLIM_HI } from './joint_limits.js';   // joint ranges from the motion capture (tools/joint_limits.py; body.c's joint_limits.h)

// ---------- vectors and quaternions ([x,y,z], [w,x,y,z]) ----------
const v3 = (x, y, z) => [x, y, z];
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vlen = a => Math.sqrt(vdot(a, a));
const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const qconj = a => [a[0], -a[1], -a[2], -a[3]];
const qnorm = a => { const n = Math.hypot(a[0], a[1], a[2], a[3]); return [a[0] / n, a[1] / n, a[2] / n, a[3] / n]; };
function qrot(q, v) { const u = [q[1], q[2], q[3]], t = vscale(vcross(u, v), 2); return vadd(vadd(v, vscale(t, q[0])), vcross(u, t)); }
function qexp(r) {
  const a = vlen(r); if (a < 1e-12) return qnorm([1, r[0] * .5, r[1] * .5, r[2] * .5]);
  const s = Math.sin(a * .5) / a; return [Math.cos(a * .5), r[0] * s, r[1] * s, r[2] * s];
}
function qlog(q) {
  if (q[0] < 0) q = [-q[0], -q[1], -q[2], -q[3]];
  const s = Math.hypot(q[1], q[2], q[3]); if (s < 1e-12) return [2 * q[1], 2 * q[2], 2 * q[3]];
  return vscale([q[1], q[2], q[3]], 2 * Math.atan2(s, q[0]) / s);
}
const QID = [1, 0, 0, 0];
export const math = { v3, vadd, vsub, vscale, vdot, vcross, vlen, qmul, qconj, qnorm, qrot, qexp, qlog, QID };

// ---------- the body (same table as body.c) ----------
export const B = { PELVIS: 0, SPINE: 1, CHEST: 2, HEAD: 3, L_UARM: 4, L_FARM: 5, R_UARM: 6, R_FARM: 7, L_THIGH: 8, L_SHIN: 9, L_FOOT: 10, R_THIGH: 11, R_SHIN: 12, R_FOOT: 13 };
export const BODY_DEFS = [
  ['pelvis', 'hips', -1, [0, 0.928, 0.004], [0, 0.915, 0.005], [0.12, 0.07, 0.09], 0.120, 0, 0, 0],
  ['spine', 'spine', 0, [0, 0.980, 0.016], [0, 1.036, 0.017], [0.11, 0.055, 0.08], 0.140, 1000, 100, 0],
  ['chest', 'chest', 1, [0, 1.093, 0.019], [0, 1.205, 0.000], [0.13, 0.11, 0.09], 0.170, 1000, 100, 0],
  ['head', 'head', 2, [0, 1.332, -0.034], [0, 1.450, -0.010], [0.08, 0.115, 0.09], 0.070, 100, 10, 0],
  ['l_uarm', 'leftUpperArm', 2, [0.105, 1.293, -0.025], [0.215, 1.293, -0.025], [0.11, 0.035, 0.035], 0.026, 400, 40, 0],
  ['l_farm', 'leftLowerArm', 4, [0.324, 1.293, -0.025], [0.472, 1.293, -0.025], [0.148, 0.03, 0.03], 0.022, 300, 30, 2],
  ['r_uarm', 'rightUpperArm', 2, [-0.105, 1.293, -0.025], [-0.215, 1.293, -0.025], [0.11, 0.035, 0.035], 0.026, 400, 40, 0],
  ['r_farm', 'rightLowerArm', 6, [-0.324, 1.293, -0.025], [-0.472, 1.293, -0.025], [0.148, 0.03, 0.03], 0.022, 300, 30, 2],
  ['l_thigh', 'leftUpperLeg', 0, [0.076, 0.888, 0.000], [0.076, 0.709, -0.004], [0.06, 0.18, 0.06], 0.146, 500, 50, 0],
  ['l_shin', 'leftLowerLeg', 8, [0.076, 0.529, -0.008], [0.076, 0.319, -0.020], [0.045, 0.21, 0.045], 0.048, 500, 50, 1],
  ['l_foot', 'leftFoot', 9, [0.076, 0.109, -0.033], [0.076, 0.035, 0.035], [0.04, 0.035, 0.105], 0.013, 400, 40, 0],
  ['r_thigh', 'rightUpperLeg', 0, [-0.076, 0.888, 0.000], [-0.076, 0.709, -0.004], [0.06, 0.18, 0.06], 0.146, 500, 50, 0],
  ['r_shin', 'rightLowerLeg', 11, [-0.076, 0.529, -0.008], [-0.076, 0.319, -0.020], [0.045, 0.21, 0.045], 0.048, 500, 50, 1],
  ['r_foot', 'rightFoot', 12, [-0.076, 0.109, -0.033], [-0.076, 0.035, 0.035], [0.04, 0.035, 0.105], 0.013, 400, 40, 0],
].map(([name, bone, parent, anchor, center, half, mass_frac, kp, kd, hinge]) => ({ name, bone, parent, anchor, center, half, mass_frac, kp, kd, hinge }));
const NB = BODY_DEFS.length, G_ACC = 9.81, ASSIST_K = 2700, ASSIST_KR = 2000;
const CORNERS = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]];
const corner = (i, k) => { const h = BODY_DEFS[i].half; return [CORNERS[k][0] * h[0], CORNERS[k][1] * h[1], CORNERS[k][2] * h[2]]; };
const hingeAxis = h => h === 1 ? [1, 0, 0] : h === 2 ? [0, 1, 0] : [0, 0, 1];

// ---------- the room (physics_gym): static oriented boxes from room.json, the same table body.c collides with ----------
// [{c, h, q, mu, soft}] -> each box also gets rad = |h| (broad phase). One room for every World, as in body.c.
let ROOM = [];
export function setRoom(boxes) { ROOM = (boxes || []).map(b => ({ c: b.c, h: b.h, q: b.q, mu: b.mu, soft: b.soft, rad: Math.sqrt(b.h[0] * b.h[0] + b.h[1] * b.h[1] + b.h[2] * b.h[2]) })); }
export const room = () => ROOM;
// the heights where the vertical line through (x, z) is inside box o, or null (body.c column_span)
function columnSpan(o, x, z) {
  const dx = x - o.c[0], dz = z - o.c[2]; if (dx * dx + dz * dz > o.rad * o.rad) return null;
  const qi = qconj(o.q), A = qrot(qi, [dx, -o.c[1], dz]), Bv = qrot(qi, [0, 1, 0]); let lo = -1e30, hi = 1e30;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(Bv[k]) < 1e-12) { if (Math.abs(A[k]) > o.h[k]) return null; continue; }
    let ta = (-o.h[k] - A[k]) / Bv[k], tb = (o.h[k] - A[k]) / Bv[k]; if (ta > tb) [ta, tb] = [tb, ta];
    if (ta > lo) lo = ta;
    if (tb < hi) hi = tb;
  }
  return lo > hi ? null : [lo, hi];
}
// the surface under (x, z) at or below ytop: the floor (0) or a room box's top there (body.c support_y)
export function supportY(x, z, ytop) { let best = 0; for (const o of ROOM) { const s = columnSpan(o, x, z); if (s && s[1] <= ytop + 1e-9 && s[1] > best) best = s[1]; } return best; }

export class World {
  constructor(totalMass = 45, substeps = 20, dt = 1 / 30) {
    this.totalMass = totalMass; this.substeps = substeps; this.h = dt / substeps; this.mu = 0.9; this.iters = 4; this.limitK = 5000;   // limitK: joint ligaments (body.c World.limit_k); 0 = no limits
    this.kpScale = 1; this.kdScale = 1; this.t = 0;
    this.tone = BODY_DEFS.map(() => 1);   // per-joint muscle tone (body.c tone)
    this.push = { body: -1, force: [0, 0, 0], left: 0 };
    // grab (viewer only, not in the C twin): a damped spring from a point on one body (r, in its rest frame)
    // toward a target, its force capped: { body, r, target: [x, y, z], k, c, max }, or null
    this.grab = null;
    // training aid (body.c): pelvis held to assistX / assistQ by compliant constraints, strength assist 0..1
    this.assist = 0; this.assistX = [0, 0, 0]; this.assistV = [0, 0, 0]; this.assistQ = QID;
    const fsum = BODY_DEFS.reduce((s, d) => s + d.mass_frac, 0);
    this.b = BODY_DEFS.map(d => {
      const m = totalMass * d.mass_frac / fsum, X = 2 * d.half[0], Y = 2 * d.half[1], Z = 2 * d.half[2];
      return { x: [0, 0, 0], v: [0, 0, 0], w: [0, 0, 0], xprev: [0, 0, 0], q: QID, qprev: QID, invMass: 1 / m,
        invI: [12 / (m * (Y * Y + Z * Z)), 12 / (m * (X * X + Z * Z)), 12 / (m * (X * X + Y * Y))] };
    });
    this.target = BODY_DEFS.map(() => QID);
    this.touch = BODY_DEFS.map(() => 0);   // contact force on each body, mean over the last step (N), as body.c World.touch
  }
  pose(rootX, rootQ, rel, rootV = [0, 0, 0]) {
    for (let i = 0; i < NB; i++) {
      const d = BODY_DEFS[i], b = this.b[i];
      if (d.parent < 0) { b.q = rootQ; b.x = rootX; } else {
        const p = this.b[d.parent], pd = BODY_DEFS[d.parent];
        const a = vadd(p.x, qrot(p.q, vsub(d.anchor, pd.center)));
        b.q = qnorm(qmul(p.q, rel ? rel[i] : QID));
        b.x = vadd(a, qrot(b.q, vsub(d.center, d.anchor)));
      }
      b.v = rootV.slice(); b.w = [0, 0, 0]; b.xprev = b.x; b.qprev = b.q;
    }
  }
  invI(b, v) { let l = qrot(qconj(b.q), v); l = [l[0] * b.invI[0], l[1] * b.invI[1], l[2] * b.invI[2]]; return qrot(b.q, l); }
  rotateBy(b, dw) { b.q = qnorm(qmul(qexp(dw), b.q)); }
  genW(b, r, n) { const rn = vcross(r, n); return b.invMass + vdot(rn, this.invI(b, rn)); }
  solvePos(b1, b2, r1, r2, n, C, alpha) {
    const h = this.h, w1 = this.genW(b1, r1, n), w2 = b2 ? this.genW(b2, r2, n) : 0, at = alpha / (h * h);
    if (w1 + w2 + at <= 0) return 0;
    const dl = -C / (w1 + w2 + at), p = vscale(n, dl);
    b1.x = vadd(b1.x, vscale(p, b1.invMass)); this.rotateBy(b1, this.invI(b1, vcross(r1, p)));
    if (b2) { b2.x = vsub(b2.x, vscale(p, b2.invMass)); this.rotateBy(b2, vscale(this.invI(b2, vcross(r2, p)), -1)); }
    return dl;
  }
  solveAng(b1, b2, n, theta, alpha) {
    const h = this.h, w = vdot(n, this.invI(b1, n)) + vdot(n, this.invI(b2, n)), at = alpha / (h * h);
    const p = vscale(n, -theta / (w + at));
    this.rotateBy(b1, this.invI(b1, p)); this.rotateBy(b2, vscale(this.invI(b2, p), -1));
  }
  velImpulse(b, r, n, dv) {
    const w = this.genW(b, r, n); if (w <= 0) return;
    const J = vscale(n, dv / w); b.v = vadd(b.v, vscale(J, b.invMass)); b.w = vadd(b.w, this.invI(b, vcross(r, J)));
  }
  substep() {
    const h = this.h, contacts = [];
    for (let i = 0; i < NB; i++) {
      const b = this.b[i];
      b.v = [b.v[0], b.v[1] - h * G_ACC, b.v[2]];
      if (i === this.push.body && this.push.left > 0) b.v = vadd(b.v, vscale(this.push.force, h * b.invMass));
      if (this.grab && i === this.grab.body) {
        const g = this.grab, rw = qrot(b.q, g.r), p = vadd(b.x, rw), vp = vadd(b.v, vcross(b.w, rw));
        let F = vsub(vscale(vsub(g.target, p), g.k), vscale(vp, g.c)); const f = vlen(F); if (f > g.max) F = vscale(F, g.max / f);
        const J = vscale(F, h); b.v = vadd(b.v, vscale(J, b.invMass)); b.w = vadd(b.w, this.invI(b, vcross(rw, J)));
      }
      b.xprev = b.x; b.x = vadd(b.x, vscale(b.v, h));
      const wl = qrot(qconj(b.q), b.w), Il = [wl[0] / b.invI[0], wl[1] / b.invI[1], wl[2] / b.invI[2]];
      let tq = vcross(wl, Il); tq = [-tq[0] * b.invI[0], -tq[1] * b.invI[1], -tq[2] * b.invI[2]];
      b.w = vadd(b.w, vscale(qrot(b.q, tq), h));
      b.qprev = b.q; this.rotateBy(b, vscale(b.w, h));
    }
    if (this.push.left > 0) this.push.left -= h;
    for (let it = 0; it < this.iters; it++) for (let i = 1; i < NB; i++) {
      const d = BODY_DEFS[i], c = this.b[i], p = this.b[d.parent], pd = BODY_DEFS[d.parent];
      const rc = qrot(c.q, vsub(d.anchor, d.center)), rp = qrot(p.q, vsub(d.anchor, pd.center));
      const D = vsub(vadd(c.x, rc), vadd(p.x, rp)), C = vlen(D);
      if (C > 1e-12) this.solvePos(c, p, rc, rp, vscale(D, 1 / C), C, 0);
      let tw = qmul(p.q, this.target[i]), e = qlog(qmul(c.q, qconj(tw)));
      const kp = d.kp * this.kpScale * this.tone[i];
      if (d.hinge) {
        const aw = qrot(p.q, hingeAxis(d.hinge)), off = vsub(e, vscale(aw, vdot(e, aw))), to = vlen(off);
        if (to > 1e-12) this.solveAng(c, p, vscale(off, 1 / to), to, 0);
        tw = qmul(p.q, this.target[i]); e = qlog(qmul(c.q, qconj(tw)));
        const ta = vdot(e, aw);
        if (Math.abs(ta) > 1e-12) this.solveAng(c, p, ta > 0 ? aw : vscale(aw, -1), Math.abs(ta), 1 / kp);
      } else { const th = vlen(e); if (th > 1e-12) this.solveAng(c, p, vscale(e, 1 / th), th, 1 / kp); }
      if (this.limitK > 0) {   // ligament: back inside the joint's range from the capture, if it left it
        const r = qlog(qmul(qconj(p.q), c.q)), lo = JLIM_LO[i], hi = JLIM_HI[i], rc = r.map((v, k) => v < lo[k] ? lo[k] : v > hi[k] ? hi[k] : v);
        if (rc[0] !== r[0] || rc[1] !== r[1] || rc[2] !== r[2]) {
          const el = qlog(qmul(c.q, qconj(qmul(p.q, qexp(rc))))), tl = vlen(el);
          if (tl > 1e-12) this.solveAng(c, p, vscale(el, 1 / tl), tl, 1 / this.limitK);
        }
      }
    }
    if (this.assist > 0) {
      const p = this.b[0], D = vsub(p.x, this.assistX), C = vlen(D);
      if (C > 1e-12) this.solvePos(p, null, [0, 0, 0], [0, 0, 0], vscale(D, 1 / C), C, 1 / (this.assist * ASSIST_K));
      const e = qlog(qmul(p.q, qconj(this.assistQ))), th = vlen(e);
      if (th > 1e-12) { const n = vscale(e, 1 / th), w = vdot(n, this.invI(p, n)), at = 1 / (this.assist * ASSIST_KR * h * h); this.rotateBy(p, this.invI(p, vscale(n, -th / (w + at)))); }
    }
    for (let i = 0; i < NB; i++) for (let k = 0; k < 8; k++) {
      const b = this.b[i], r = qrot(b.q, corner(i, k)), pw = vadd(b.x, r);
      if (pw[1] >= 0) continue;
      const ln = this.solvePos(b, null, r, [0, 0, 0], [0, -1, 0], -pw[1], 0); this.touch[i] += Math.abs(ln) / (h * h);
      contacts.push([i, k, ln]);
    }
    // room boxes (body.c): a corner inside one is pushed out through its nearest face, at the box's softness
    const roomContacts = [];
    if (ROOM.length) for (let i = 0; i < NB; i++) {
      const b = this.b[i], hb = BODY_DEFS[i].half, br = Math.sqrt(hb[0] * hb[0] + hb[1] * hb[1] + hb[2] * hb[2]);
      for (const o of ROOM) {
        const dc = vsub(b.x, o.c); if (vdot(dc, dc) > (br + o.rad) * (br + o.rad)) continue;
        for (let k = 0; k < 8; k++) {
          const r = qrot(b.q, corner(i, k)), l = qrot(qconj(o.q), vsub(vadd(b.x, r), o.c));
          const dx = o.h[0] - Math.abs(l[0]), dy = o.h[1] - Math.abs(l[1]), dz = o.h[2] - Math.abs(l[2]);
          if (dx <= 0 || dy <= 0 || dz <= 0) continue;
          let nl, d;
          if (dx <= dy && dx <= dz) { d = dx; nl = [l[0] < 0 ? -1 : 1, 0, 0]; } else if (dy <= dz) { d = dy; nl = [0, l[1] < 0 ? -1 : 1, 0]; } else { d = dz; nl = [0, 0, l[2] < 0 ? -1 : 1]; }
          const n = qrot(o.q, nl);
          const ln = this.solvePos(b, null, r, [0, 0, 0], vscale(n, -1), d, o.soft); this.touch[i] += Math.abs(ln) / (h * h);
          roomContacts.push([i, k, ln, n, o.mu]);
        }
      }
    }
    for (let i = 0; i < NB; i++) {
      const b = this.b[i];
      b.v = vscale(vsub(b.x, b.xprev), 1 / h);
      const dq = qmul(b.q, qconj(b.qprev)), w = vscale([dq[1], dq[2], dq[3]], 2 / h);
      b.w = dq[0] >= 0 ? w : vscale(w, -1);
    }
    for (const [i, k, ln] of contacts) {
      const b = this.b[i], r = qrot(b.q, corner(i, k));
      let vp = vadd(b.v, vcross(b.w, r));
      if (vp[1] < 0) this.velImpulse(b, r, [0, 1, 0], -vp[1]);
      vp = vadd(b.v, vcross(b.w, r)); const vt = [vp[0], 0, vp[2]], s = vlen(vt);
      if (s > 1e-9) this.velImpulse(b, r, vscale(vt, 1 / s), -Math.min(s, this.mu * Math.abs(ln) / h));
    }
    for (const [i, k, ln, n, mu] of roomContacts) {
      const b = this.b[i], r = qrot(b.q, corner(i, k));
      let vp = vadd(b.v, vcross(b.w, r)), vn = vdot(vp, n);
      if (vn < 0) this.velImpulse(b, r, n, -vn);
      vp = vadd(b.v, vcross(b.w, r)); vn = vdot(vp, n); const vt = vsub(vp, vscale(n, vn)), s = vlen(vt);
      if (s > 1e-9) this.velImpulse(b, r, vscale(vt, 1 / s), -Math.min(s, mu * Math.abs(ln) / h));
    }
    if (this.assist > 0) { const p = this.b[0], f = Math.min(1, h * this.assist * 20); p.v = vadd(p.v, vscale(vsub(this.assistV, p.v), f)); p.w = vscale(p.w, 1 - f); }
    for (let i = 1; i < NB; i++) {
      const d = BODY_DEFS[i], c = this.b[i], p = this.b[d.parent];
      const dw = vsub(c.w, p.w), m = vlen(dw); if (m < 1e-12) continue;
      const n = vscale(dw, 1 / m), weff = vdot(n, this.invI(c, n)) + vdot(n, this.invI(p, n));
      const f = Math.min(1, h * d.kd * this.kdScale * this.tone[i] * weff), imp = vscale(n, -f * m / weff);
      c.w = vadd(c.w, this.invI(c, imp)); p.w = vsub(p.w, this.invI(p, imp));
    }
  }
  step() {
    this.touch.fill(0);
    for (let s = 0; s < this.substeps; s++) this.substep();
    for (let i = 0; i < NB; i++) this.touch[i] /= this.substeps;
    this.t += this.h * this.substeps;
  }
  com() {
    let c = [0, 0, 0], v = [0, 0, 0], m = 0;
    for (const b of this.b) { const mi = 1 / b.invMass; c = vadd(c, vscale(b.x, mi)); v = vadd(v, vscale(b.v, mi)); m += mi; }
    return { x: vscale(c, 1 / m), v: vscale(v, 1 / m) };
  }
  lowest(i) { let lo = 1e9; for (let k = 0; k < 8; k++) lo = Math.min(lo, vadd(this.b[i].x, qrot(this.b[i].q, corner(i, k)))[1]); return lo; }
  corners(i) { return CORNERS.map((_, k) => vadd(this.b[i].x, qrot(this.b[i].q, corner(i, k)))); }
}

// ---------- the control interface (env.c) ----------
export const ACT_DIM = 31, OBS_DIM = NB * 15 + 6, CTRL_HZ = 30, ACT_FILTER = 0.35;
// action range per joint (rad per unit action), indexed by the child body (env.c ACT_SCALES)
export const ACT_SCALES = [0, 0.5, 0.5, 0.4, 0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 0.6, 1.0, 1.2, 0.6];
export function standPose() {
  const rel = BODY_DEFS.map(() => QID), a = 68 * Math.PI / 180;
  rel[B.L_UARM] = qexp([0, 0, -a]); rel[B.R_UARM] = qexp([0, 0, a]);
  return rel;
}
// On hands and knees (env.c crawl_pose): joint rotations and the root rotation; the torso slopes up toward the
// shoulders, thighs and arms vertical, shins along the floor.
const arc = (a, b) => { const c = vcross(a, b); return qnorm([1 + vdot(a, b), c[0], c[1], c[2]]); };
export function crawlPose() {
  const rel = BODY_DEFS.map(() => QID), tilt = Math.PI / 2 - 0.75, rootQ = qexp([tilt, 0, 0]), down = qrot(qconj(rootQ), [0, -1, 0]);
  rel[B.L_THIGH] = rel[B.R_THIGH] = arc([0, -1, 0], down); rel[B.L_SHIN] = rel[B.R_SHIN] = qexp([Math.PI / 2, 0, 0]);
  rel[B.L_FOOT] = rel[B.R_FOOT] = qexp([0.6, 0, 0]);
  rel[B.L_UARM] = arc([1, 0, 0], down); rel[B.R_UARM] = arc([-1, 0, 0], down); rel[B.HEAD] = qexp([-tilt + 0.25, 0, 0]);
  return { rel, rootQ };
}
// The stand pose with relaxed idle motion at time t, scaled by amp (rad of ankle sway; env.c sway_pose): ankle
// sway counterbalanced at the hips and spine, a slow weight shift with the unloaded knee softening, breathing in
// the chest and shoulders, slow drifts of the gaze, softly bent elbows.
export function swayPose(t, amp) {
  const rel = standPose(); if (!(amp > 0)) return rel;
  const k = amp / 0.012, s = Math.sin, P = 2 * Math.PI;
  const shift = 0.035 * k * s(P * 0.07 * t + 0.5);
  const fb = amp * s(P * 0.23 * t), lr = 0.6 * amp * s(P * 0.17 * t + 1.0) + shift;
  const br = 0.03 * k * (0.5 + 0.5 * s(P * 0.25 * t));
  const yaw = k * (0.15 * s(P * 0.045 * t) + 0.06 * s(P * 0.11 * t + 2.0)), nod = k * (0.04 * s(P * 0.08 * t + 1.0) - 0.02);
  const ankle = qexp([fb, 0, lr]), hip = qexp([-0.4 * fb, 0, -0.4 * lr]), spine = qexp([0.25 * fb, 0, 0.25 * lr]);
  const mul = (i, q) => { rel[i] = qmul(rel[i], q); };
  mul(B.L_FOOT, ankle); mul(B.R_FOOT, ankle); mul(B.L_THIGH, hip); mul(B.R_THIGH, hip);
  mul(B.L_SHIN, qexp([0.12 * k * Math.max(0, -shift / 0.035), 0, 0])); mul(B.R_SHIN, qexp([0.12 * k * Math.max(0, shift / 0.035), 0, 0]));
  mul(B.SPINE, spine); mul(B.CHEST, qexp([-br, 0, 0])); mul(B.HEAD, qexp([nod + 0.5 * br, yaw, 0]));
  mul(B.L_UARM, qexp([0, 0, 0.5 * br])); mul(B.R_UARM, qexp([0, 0, -0.5 * br]));
  mul(B.L_FARM, qexp([0, -0.25 * k, 0])); mul(B.R_FARM, qexp([0, 0.25 * k, 0]));
  return rel;
}
// Joint targets: reference rotation followed by exp(scale * f), f the action low-passed by ACT_FILTER (kept on the world).
// Elbows and knees bend one way, straight to fully bent (env.c hinge_limit; the floor skills): flexion is -y for
// the left elbow, +y for the right, +x for the knees.
function hingeLimit(i, q) {
  const h = BODY_DEFS[i].hinge, ang = 2 * Math.atan2(q[h], q[0]), flex = i === B.L_FARM ? -ang : ang, mx = i === B.L_FARM || i === B.R_FARM ? 2.5 : 2.6;
  const f = Math.min(mx, Math.max(0, flex));
  return f === flex ? q : qexp(vscale(hingeAxis(h), i === B.L_FARM ? -f : f));
}
// mul: the floor skills (crawl, get up) widen every joint's range (env.c TaskCfg.act_mul)
export function applyAction(world, ref, a, mul = 1) {
  world.filt ||= new Array(ACT_DIM).fill(0);
  for (let j = 0; j < ACT_DIM; j++) world.filt[j] += ACT_FILTER * (a[j] - world.filt[j]);
  const f = world.filt; let k = 0;
  for (let i = 1; i < NB; i++) {
    const d = BODY_DEFS[i];
    const r = d.hinge ? vscale(hingeAxis(d.hinge), f[k++]) : [f[k++], f[k++], f[k++]];
    world.target[i] = qmul(ref[i], qexp(vscale(r, ACT_SCALES[i] * mul)));
    if (d.hinge && mul > 1) world.target[i] = hingeLimit(i, world.target[i]);
  }
}
// phase: radians of the reference clip's cycle (0 for balance); floor: the floor skills' heading (the pelvis's
// forward and up axes together, since forward alone is vertical when she lies face down)
export function observe(world, phase = 0, floor = false) {
  const pz = qrot(world.b[0].q, [0, 0, 1]), py = qrot(world.b[0].q, [0, 1, 0]), pf = floor ? vadd(pz, py) : pz, hi = qexp([0, -Math.atan2(pf[0], pf[2]), 0]), root = world.b[0].x, o = [];
  for (let i = 0; i < NB; i++) {
    const b = world.b[i], p = qrot(hi, vsub(b.x, root)); if (i === 0) p[1] = root[1];
    const q = qmul(hi, b.q), ux = qrot(q, [1, 0, 0]), uy = qrot(q, [0, 1, 0]), v = qrot(hi, b.v), w = qrot(hi, b.w);
    o.push(...p, ...ux, ...uy, ...v, w[0] * .2, w[1] * .2, w[2] * .2);
  }
  o.push(world.lowest(B.L_FOOT) < 0.01 ? 1 : 0, world.lowest(B.R_FOOT) < 0.01 ? 1 : 0);
  const d = qrot(hi, vsub(world.com().x, root)); o.push(d[0], d[2]);
  o.push(Math.sin(phase), Math.cos(phase));
  return o;
}
export function fallen(world) {   // heights from the surface under her (env.c fallen; y = 0 on open floor)
  const p = world.b[0].x, sp = supportY(p[0], p[2], p[1]);
  if (p[1] - sp < 0.55 || world.b[B.HEAD].x[1] - sp < 1.0) return true;
  for (let i = 0; i < NB; i++) if (![B.L_FOOT, B.R_FOOT, B.L_SHIN, B.R_SHIN].includes(i)) { const c = world.b[i].x; if (world.lowest(i) - supportY(c[0], c[2], c[1]) < 0.05) return true; }
  return false;
}
// The go-to skill's observation (env.c env_obs for TASK_GOTO, OBS_DIM 234): the bodies as in observe(), but the pelvis
// height from her lowest foot corner and the foot flags from touch; then the goal from her pelvis in the heading
// frame (horizontal at most 4 m) and its distance; then each body's touch as a share of her weight.
export const OBS_GOTO = OBS_DIM + 4 + NB;
export function observeGoto(world, phase, goal) {
  const o = observe(world, phase, false), root = world.b[0].x, wt = world.totalMass * G_ACC;
  o[1] = root[1] - Math.min(world.lowest(B.L_FOOT), world.lowest(B.R_FOOT));
  o[NB * 15] = world.touch[B.L_FOOT] > 0.02 * wt ? 1 : 0; o[NB * 15 + 1] = world.touch[B.R_FOOT] > 0.02 * wt ? 1 : 0;
  const pz = qrot(world.b[0].q, [0, 0, 1]), hi = qexp([0, -Math.atan2(pz[0], pz[2]), 0]);
  const g = qrot(hi, vsub(goal, root)), r = Math.sqrt(g[0] * g[0] + g[2] * g[2]), s = r > 4 ? 4 / r : 1;
  o.push(g[0] * s, g[1], g[2] * s, Math.min(r, 4));
  for (let i = 0; i < NB; i++) o.push(Math.min(3, world.touch[i] / wt));
  return o;
}

// ---------- policies (policy.c), loaded from the JSON written by tools/pol2json.py ----------
const sigm = x => 1 / (1 + Math.exp(-x)), softplus = x => x > 20 ? x : Math.log1p(Math.exp(x));
const erf = x => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
const gelu = x => 0.5 * x * (1 + erf(x * Math.SQRT1_2));
const dot = (a, ao, b, n) => { let s = 0; for (let i = 0; i < n; i++) s += a[ao + i] * b[i]; return s; };
const rmsn = (x, w, wo, d) => { let s = 0; for (let i = 0; i < d; i++) s += x[i] * x[i]; const r = 1 / Math.sqrt(s / d + 1e-6); return x.map((v, i) => v * r * w[wo + i]); };

export class Policy {
  constructor(spec) {   // spec: { type, in, out, hidden, layers, trees, depth, n, p, theta: Float32Array, mean, std }
    Object.assign(this, spec); this.reset();
  }
  reset() {
    const N = (1 << (this.depth + 1)) - 1;
    this.h = new Float32Array(this.type === 'ssm' ? this.layers * this.trees * N * this.n * this.p : this.type === 'ltc' ? this.hidden : 1);
  }
  act(obs) {
    const I = this.in, O = this.out, H = this.hidden, th = this.theta;
    const o = obs.map((v, i) => Math.max(-5, Math.min(5, (v - this.mean[i]) / this.std[i])));
    const a = new Array(O);
    if (this.type === 'mlp') {
      const W1 = 0, b1 = W1 + I * H, W2 = b1 + H, b2 = W2 + H * H, W3 = b2 + H, b3 = W3 + H * O;
      const x = Array.from({ length: H }, (_, i) => Math.tanh(dot(th, W1 + i * I, o, I) + th[b1 + i]));
      const y = Array.from({ length: H }, (_, i) => Math.tanh(dot(th, W2 + i * H, x, H) + th[b2 + i]));
      for (let k = 0; k < O; k++) a[k] = Math.tanh(dot(th, W3 + k * H, y, H) + th[b3 + k]);
      return a;
    }
    if (this.type === 'ltc') {
      const Wu = 0, Wx = Wu + H * I, b = Wx + H * H, lt = b + H, A = lt + H, Wo = A + H, bo = Wo + O * H, UNFOLD = 6, dt = 1 / (30 * UNFOLD);
      const uin = Array.from({ length: H }, (_, i) => dot(th, Wu + i * I, o, I) + th[b + i]);
      let h = Array.from(this.h);
      for (let k = 0; k < UNFOLD; k++) h = h.map((hi, i) => { const f = sigm(dot(th, Wx + i * H, h, H) + uin[i]); return (hi + dt * f * th[A + i]) / (1 + dt * (Math.exp(-th[lt + i]) + f)); });
      this.h.set(h);
      for (let k = 0; k < O; k++) a[k] = Math.tanh(dot(th, Wo + k * H, h, H) + th[bo + k]);
      return a;
    }
    const d = H, T = this.trees, D = this.depth, n = this.n, pv = this.p, N = (1 << (D + 1)) - 1, TN = T * N, Pd = 2 * n * T + 2 * pv + T;
    let q = 0; const Win = q, bin = Win + I * d; q = bin + d;
    let x = Array.from({ length: d }, (_, i) => dot(th, Win + i * I, o, I) + th[bin + i]);
    for (let l = 0; l < this.layers; l++) {
      const norm = q, w_in = norm + d, w_out = w_in + TN * d, bias = w_out + TN * d, A_log = bias + TN, proj = A_log + TN, dt_bias = proj + Pd * d, U = dt_bias + T;
      q = U + T * pv * d;
      const Hs = l * TN * n * pv, u = rmsn(x, th, norm, d);
      const pp = Array.from({ length: Pd }, (_, r) => dot(th, proj + r * d, u, d));
      for (let g = 0; g < 2 * T; g++) { let ss = 0; for (let j = 0; j < n; j++) ss += pp[g * n + j] ** 2; const rr = 1 / Math.sqrt(ss + 1e-6); for (let j = 0; j < n; j++) pp[g * n + j] *= rr; }
      const Vv = 2 * T * n, Z = Vv + pv, dtz = Z + pv, y = new Array(d).fill(0), Racc = new Array(T * pv).fill(0);
      for (let tr = 0; tr < T; tr++) {
        const dt = softplus(pp[dtz + tr] + th[dt_bias + tr]), Bo = tr * n, Co = T * n + tr * n;
        let node = 0;
        for (let k = 0; k <= D; k++) {
          const row = tr * N + node, Hn = Hs + row * n * pv, dec = Math.exp(-Math.exp(th[A_log + row]) * dt), R = new Array(pv).fill(0);
          for (let j = 0; j < n; j++) for (let qq = 0; qq < pv; qq++) { this.h[Hn + j * pv + qq] *= dec; R[qq] += pp[Co + j] * this.h[Hn + j * pv + qq]; }
          for (let j = 0; j < n; j++) for (let qq = 0; qq < pv; qq++) this.h[Hn + j * pv + qq] += dt * pp[Bo + j] * pp[Vv + qq];
          let s = dot(th, w_in + row * d, u, d) + th[bias + row]; for (let qq = 0; qq < pv; qq++) s += R[qq] * pp[Z + qq];
          const g = gelu(s);
          for (let i = 0; i < d; i++) y[i] += g * th[w_out + row * d + i];
          for (let qq = 0; qq < pv; qq++) Racc[tr * pv + qq] += R[qq];
          node = 2 * node + 1 + (s > 0 ? 1 : 0);
        }
      }
      for (let r = 0; r < T * pv; r++) for (let i = 0; i < d; i++) y[i] += Racc[r] * th[U + r * d + i];
      x = x.map((v, i) => v + y[i]);
    }
    const nf = q, Wo = nf + d, bo = Wo + O * d, u = rmsn(x, th, nf, d);
    for (let k = 0; k < O; k++) a[k] = Math.tanh(dot(th, Wo + k * d, u, d) + th[bo + k]);
    return a;
  }
}
export async function loadPolicy(url) {
  const j = await (await fetch(url)).json();
  const f32 = s => new Float32Array(Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer);
  return new Policy({ ...j, theta: f32(j.theta), mean: f32(j.mean), std: f32(j.std) });
}

// A nanotss policy (ppo_tss.c): the language model's own stack, compiled to WebAssembly from nanotss_blt.c
// (tss_wasm.c), so the browser runs exactly the network that was trained. Each policy gets its own instance,
// since the module keeps one model and one recurrent state. The JSON holds the .tss and .head files in base64.
export async function loadTssPolicy(url, wasmUrl = 'tss.wasm') {
  const j = await (await fetch(url)).json();
  const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const tss = bytes(j.tss), head = bytes(j.head);
  const wasi = new Proxy({}, { get: () => () => 0 });   // no files, no console: the policy never needs them
  const { instance } = await WebAssembly.instantiate(await (await fetch(wasmUrl)).arrayBuffer(), { wasi_snapshot_preview1: wasi });
  const X = instance.exports; X._initialize();
  const p = X.tss_buf(tss.length + head.length);
  const mem = new Uint8Array(X.memory.buffer, p, tss.length + head.length); mem.set(tss); mem.set(head, tss.length);
  const d = X.tss_load(p, tss.length, head.length);
  if (d <= 0) throw new Error(`nanotss policy ${url} did not load (${d})`);
  return {
    type: 'nanotss', params: j.params, width: d,
    reset() { X.tss_reset(); },
    act(obs) {
      new Float32Array(X.memory.buffer, X.tss_obs(), obs.length).set(obs);   // 216 inputs for the body/ skills, 234 for go-to
      return Array.from(new Float32Array(X.memory.buffer, X.tss_step(), ACT_DIM));
    },
  };
}

// ---------- reference clips (motions_ref.bin from tools/motions_to_ref.py; mref.c) ----------
const qslerp = (a, b, t) => { if (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0) b = b.map(v => -v); return qnorm(a.map((v, i) => v + t * (b[i] - v))); };
export class Clip {
  constructor(name, frames, fps, root, q) { Object.assign(this, { name, frames, fps, root, q }); this.T = (frames - 1) / fps; this.shift = [0, 0, 0]; }
  // Reference at clip time t: joint rotations (rel[0] = root rotation), body world rotations, root position,
  // the root carried forward across loops and offset by this.shift.
  at(t) {
    const u = ((t % this.T) + this.T) % this.T, x = u * this.fps; let f0 = Math.floor(x); if (f0 >= this.frames - 1) f0 = this.frames - 2;
    const w = x - f0, world = [];
    for (let i = 0; i < NB; i++) world.push(qslerp(this.q[f0 * NB + i], this.q[(f0 + 1) * NB + i], w));
    const r0 = this.root[f0], r1 = this.root[f0 + 1], cyc = Math.floor(t / this.T), last = this.root[this.frames - 1], first = this.root[0];
    const root = [0, 1, 2].map(k => r0[k] + w * (r1[k] - r0[k]) + (k === 1 ? 0 : cyc * (last[k] - first[k])) + this.shift[k]);
    const rel = world.map((q, i) => i ? qmul(qconj(world[BODY_DEFS[i].parent]), q) : q);
    return { rel, world, root };
  }
}
export async function loadClips(url) {
  // url: motions_ref.json ({ b64 }) as published with the page, or the raw motions_ref.bin
  const res = await fetch(url);
  const buf = url.endsWith('.json') ? Uint8Array.from(atob((await res.json()).b64), c => c.charCodeAt(0)).buffer : await res.arrayBuffer();
  const dv = new DataView(buf); let o = 4;
  const n = dv.getInt32(o, true); o += 4; const clips = {};
  for (let k = 0; k < n; k++) {
    const name = new TextDecoder().decode(new Uint8Array(buf, o, 16)).replace(/\0+$/, ''); o += 16;
    const frames = dv.getInt32(o, true), fps = dv.getFloat32(o + 4, true); o += 8;
    const root = [], q = [];
    for (let f = 0; f < frames; f++) {
      root.push([dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)]); o += 12;
      for (let i = 0; i < NB; i++) { q.push(qnorm([0, 1, 2, 3].map(j => dv.getFloat32(o + 4 * j, true)))); o += 16; }
    }
    clips[name] = new Clip(name, frames, fps, root, q);
  }
  return clips;
}
// Place the world in a clip's pose at time t with the clip's velocities (env.c env_reset, TASK_IMITATE).
export function poseFromClip(world, clip, t) {
  clip.shift = [0, 0, 0];
  const a = clip.at(t); clip.shift = [-a.root[0], 0, -a.root[2]];
  const A = clip.at(t), Bn = clip.at(t + 1 / CTRL_HZ), dt = 1 / CTRL_HZ;
  const W2 = new World(45, 20, dt); W2.pose(Bn.root, Bn.rel[0], Bn.rel);
  world.pose(A.root, A.rel[0], A.rel);
  world.b.forEach((b, i) => { b.v = vscale(vsub(W2.b[i].x, b.x), 1 / dt); b.w = vscale(qlog(qmul(W2.b[i].q, qconj(b.q))), 1 / dt); });
  let lo = 1e9; for (let i = 0; i < NB; i++) lo = Math.min(lo, world.lowest(i));
  if (lo < 0) for (const b of world.b) b.x = [b.x[0], b.x[1] - lo, b.x[2]];
  for (const b of world.b) b.xprev = b.x;
  world.target = A.rel.slice();
  return A;
}

// Pelvis support toward a clip's root path and rotation, strength s in 0..1 (env.c, TASK_IMITATE assist):
// clip time t is the start of the coming control step; the target is the reference at its end.
export function setAssist(world, clip, t, s) {
  world.assist = s || 0; if (!s) return;
  const A = clip.at(t), Bn = clip.at(t + 1 / CTRL_HZ);
  world.assistX = Bn.root; world.assistQ = Bn.world[0]; world.assistV = vscale(vsub(Bn.root, A.root), CTRL_HZ);
}
