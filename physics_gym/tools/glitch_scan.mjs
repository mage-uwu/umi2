// glitch_scan.mjs: the go-to policy walking to goals all over a room bank in body.js (the viewer's physics and control
// loop), watching the physics for glitches: a body corner deep inside a room box after a step (penetration), a body's
// velocity jumping by more than 4 m/s in one control step while she is up (an impulse), and a loaded foot (over 100 N)
// sliding along the surface it stands on (friction). Reports where (the feature under her) and how often.
//   node tools/glitch_scan.mjs ROOMS.json [policy.json] [episodes] [seconds] [assist]
import { World, setRoom, supportY, setHandUp, observeGoto, applyAction, loadTssPolicy, loadClips, poseFromClip, fallen, BODY_DEFS, B, math } from '../web/body.js';
import { readFileSync } from 'fs';
globalThis.fetch = async u => { const p = new URL(u, import.meta.url.replace('tools/glitch_scan.mjs', 'web/')); const buf = readFileSync(p); return { ok: true, json: async () => JSON.parse(buf.toString()), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) }; };
const [roomsFile, polFile = 'goto_tss.json', nEp = '24', secs = '20', assistS = '0.057'] = process.argv.slice(2);
const doc = JSON.parse(readFileSync(roomsFile)); const rooms = doc.rooms || [doc];
const pol = await loadTssPolicy(polFile), clips = await loadClips('cmu_gait.json'), clip = clips.walk;
const { qrot, qconj, vsub, vadd, qexp, qmul } = math;
let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
function featureAt(boxes, x, z, y) { let best = null, bh = -1; for (const o of boxes) { const l = qrot(qconj(o.q), vsub([x, y - 0.05, z], o.c)); if (Math.abs(l[0]) < o.h[0] && Math.abs(l[2]) < o.h[2] && Math.abs(l[1]) < o.h[1] + 0.3 && o.c[1] + o.h[1] > bh) { best = o.feature; bh = o.c[1] + o.h[1]; } } return best || 'floor'; }
function depthIn(o, p) { const l = qrot(qconj(o.q), vsub(p, o.c)); const d = Math.min(o.h[0] - Math.abs(l[0]), o.h[1] - Math.abs(l[1]), o.h[2] - Math.abs(l[2])); return d; }
const corners = i => { const h = BODY_DEFS[i].half, out = []; for (const a of [-1, 1]) for (const b of [-1, 1]) for (const c of [-1, 1]) out.push([a * h[0], b * h[1], c * h[2]]); return out; };
const stats = { steps: 0, falls: 0, goals: 0, pen: {}, imp: {}, slip: {}, slipT: {} };
const add = (m, k, v = 1) => { m[k] = (m[k] || 0) + v; };
for (let ep = 0; ep < +nEp; ep++) {
  const R = rooms[ep % rooms.length]; setRoom(R.boxes); const boxes = R.boxes;
  const tops = boxes.filter(o => o.kind !== 'outer' && o.kind !== 'pillar' && o.kind !== 'wall' && qrot(o.q, [0, 1, 0])[1] > 0.9);
  const free = () => { for (let t = 0; t < 200; t++) { const x = (rnd() * 2 - 1) * 7, z = (rnd() * 2 - 1) * 7; let ok = true; for (const a of [-0.45, 0, 0.45]) for (const b of [-0.45, 0, 0.45]) if (supportY(x + a, z + b, 3) > 0.005) ok = false; if (ok) return [x, z]; } return [0, 0]; };
  const ahead = g => { const f = qrot(W.b[0].q, [0, 0, 1]), dx = g[0] - W.b[0].x[0], dz = g[2] - W.b[0].x[2], r = Math.hypot(dx, dz); return r > 1 && r < 6 && (f[0] * dx + f[2] * dz) / (Math.hypot(f[0], f[2]) * r) > Math.cos(0.5); };
  const goalPick = () => { for (let t = 0; t < 300; t++) { const g = goalAny(); if (ahead(g)) return g; } return goalAny(); };
  const goalAny = () => { if (rnd() < 0.7 && tops.length) { const o = tops[Math.floor(rnd() * tops.length)]; const p = vadd(o.c, qrot(o.q, [(rnd() * 2 - 1) * 0.6 * o.h[0], o.h[1], (rnd() * 2 - 1) * 0.6 * o.h[2]])); const y = supportY(p[0], p[2], p[1] + 0.01); if (y > 0.02 && y < 0.8) return [p[0], y, p[2]]; } const [x, z] = free(); return [x, 0, z]; };
  const W = new World(45, 20, 1 / 30); let ref = poseFromClip(W, clip, 0).rel;
  const [sx, sz] = free(), yaw = rnd() * 2 * Math.PI, q = qexp([0, yaw, 0]), p0 = W.b[0].x.slice();
  for (const b of W.b) { const d = qrot(q, vsub(b.x, p0)); b.x = [sx + d[0], b.x[1], sz + d[2]]; b.q = qmul(q, b.q); b.v = qrot(q, b.v); b.w = qrot(q, b.w); b.xprev = b.x; b.qprev = b.q; }
  pol.reset(); const pend = []; let goal = goalPick(), vprev = W.b.map(b => b.v.slice()), fprev = [W.b[B.L_FOOT].x.slice(), W.b[B.R_FOOT].x.slice()];
  for (let s = 0; s < +secs * 30; s++) {
    setHandUp(W, +assistS); const phase = 2 * Math.PI * W.t / clip.T;
    applyAction(W, ref, pol.act(observeGoto(W, phase, goal)), 1); W.step(); ref = clip.at(W.t).rel; stats.steps++;
    const px = W.b[0].x, here = featureAt(boxes, px[0], px[2], px[1]);
    if (fallen(W)) { stats.falls++; add(stats.imp, 'fall@' + here); pend.length = 0; break; }
    while (pend.length && pend[0][0] < s - 30) add(stats.imp, pend.shift()[1]);   // spikes a second or more before any fall: not the fall itself
    if (Math.hypot(px[0] - goal[0], px[2] - goal[2]) < 0.5) { stats.goals++; goal = goalPick(); }
    for (let i = 0; i < 14; i++) {
      const b = W.b[i], dv = Math.hypot(b.v[0] - vprev[i][0], b.v[1] - vprev[i][1], b.v[2] - vprev[i][2]); if (dv > 4) pend.push([s, here + " " + BODY_DEFS[i].name]);
      let dmax = 0; for (const c of corners(i)) { const p = vadd(b.x, qrot(b.q, c)); for (const o of boxes) { const d = depthIn(o, p); if (d > dmax) dmax = d; } }
      if (dmax > 0.02) add(stats.pen, here + (dmax > 0.05 ? ' >5cm' : ' 2-5cm'));
    }
    [B.L_FOOT, B.R_FOOT].forEach((fi, j) => { const f = W.b[fi].x; if (W.touch[fi] > 100) { const fh = featureAt(boxes, f[0], f[2], f[1]); add(stats.slip, fh, Math.hypot(f[0] - fprev[j][0], f[2] - fprev[j][2])); add(stats.slipT, fh, 1 / 30); } fprev[j] = f.slice(); });
    vprev = W.b.map(b => b.v.slice());
  }
  for (const [, h] of pend) add(stats.imp, h);
}
const slip = Object.fromEntries(Object.keys(stats.slip).map(k => [k, `${(100 * stats.slip[k] / stats.slipT[k]).toFixed(1)} cm/s over ${stats.slipT[k].toFixed(0)} s`]));
console.log(JSON.stringify({ episodes: +nEp, seconds: (stats.steps / 30).toFixed(0), falls: stats.falls, goals: stats.goals, impulses_over_4ms: stats.imp, penetration: stats.pen, loaded_foot_slip: slip }, null, 1));
