// climb_course.mjs: a room bank of climbing courses, each built round one clip of data/cmu_climb.bin (tools/cmu_fetch.py):
// the boxes its feet stand on, and the obstacle it jumps over. Room r is for clip r % (number of clips), in the clip's own
// frame (its root at frame 0 over the origin), as env.c's course mode plays it (task getup_imitate in a room).
//   Steps, a stool, a ledge, a platform: wherever a foot rests (slower than 0.15 m/s for 6 frames) over 0.1 m above the clip's
//   ground, or a hand rests by such a surface, it stands on something. Those contacts, grouped by height (within 0.08 m), give one box each,
//   solid from the floor: its top at the level, its near edge (along the clip's travel) 6 cm short of the first contact
//   on it (or clear of any body below its top), and every box running on to 0.6 m past the last contact on the top level, so lower steps lie under the higher ones.
//   A jump over: where both feet are off the ground longest, a stool 0.4 m high and 0.3 m deep across the flight's middle,
//   lowered if any body of the clip passes lower over it.
//   node tools/climb_course.mjs data/cmu_climb.bin out.json
import { World, loadClips, B, BODY_DEFS, math } from '../web/body.js';
import { readFileSync, writeFileSync } from 'fs';
globalThis.fetch = async u => { const buf = readFileSync(u); return { ok: true, arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) }; };
const [file, out] = process.argv.slice(2);
const clips = await loadClips(file), names = Object.keys(clips);
const FARMS = [B.L_FARM, B.R_FARM], FEET = [B.L_FOOT, B.R_FOOT];
const rooms = names.map(name => {
  const c = clips[name]; c.shift = [0, 0, 0]; const r0 = c.at(0).root; c.shift = [-r0[0], 0, -r0[2]];
  const W = new World(45, 20, 1 / 30), fr = [];
  for (let f = 0; f < c.frames; f++) {
    const a = c.at(Math.min(f / c.fps, (c.frames - 1) / c.fps * (1 - 1e-9))); W.pose(a.root, a.rel[0], a.rel);   // (the clip's end itself wraps to its start)
    fr.push(W.b.map((b, i) => {   // each body's centre, lowest point and box corners
      const h = BODY_DEFS[i].half, cs = [];
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) cs.push(math.vadd(b.x, math.qrot(b.q, [sx * h[0], sy * h[1], sz * h[2]])));
      return { x: b.x[0], y: b.x[1], z: b.x[2], lo: W.lowest(i), cs };
    }));
  }
  const g = Math.min(...fr.flat().filter((_, k) => FEET.includes(k % BODY_DEFS.length)).map(p => p.lo));
  const end = c.at((c.frames - 1) / c.fps * (1 - 1e-9)).root, start = c.at(0).root;
  let ux = end[0] + c.shift[0] - (start[0] + c.shift[0]), uz = end[2] - start[2]; const L = Math.hypot(ux, uz) || 1; ux /= L; uz /= L;
  const along = p => p.x * ux + p.z * uz, side = p => -p.x * uz + p.z * ux;
  const speed = (f, i) => f === 0 ? 1e9 : Math.hypot(fr[f][i].x - fr[f - 1][i].x, fr[f][i].lo - fr[f - 1][i].lo, fr[f][i].z - fr[f - 1][i].z) * 30;
  const still = (f, i, v) => { for (let k = Math.max(1, f - 3); k <= Math.min(fr.length - 1, f + 2); k++) if (speed(k, i) > v) return false; return true; };   // for 6 frames
  const feet = [], hands = [];
  for (let f = 1; f < fr.length; f++) {
    for (const i of FEET) if (fr[f][i].lo - g > 0.1 && still(f, i, 0.15)) feet.push({ ...fr[f][i], h: fr[f][i].lo - g });
    for (const i of FARMS) if (still(f, i, 0.2)) hands.push({ ...fr[f][i], h: fr[f][i].lo - g });
  }
  feet.sort((a, b) => a.h - b.h);
  const levels = [];
  for (const p of feet) { const l = levels.find(l => p.h - l.ref < 0.08); if (l) l.pts.push(p); else levels.push({ ref: p.h, pts: [p] }); }
  const lv = levels.filter(l => l.pts.length >= 8).map(l => {
    const hs = l.pts.map(p => p.h).sort((a, b) => a - b); let top = hs[0], most = 0;   // the top: the densest 3 cm of its contacts
    for (const h of hs) { const k = hs.filter(x => x >= h && x < h + 0.03).length; if (k > most) { most = k; top = h; } }
    const pts = [...l.pts, ...hands.filter(p => p.h > top - 0.02 && p.h < top + 0.12)];   // a hand resting on its edge is on it too
    return { top, near: Math.min(...pts.map(along)), far: Math.max(...pts.map(along)), s0: Math.min(...pts.map(side)), s1: Math.max(...pts.map(side)) };
  }).sort((a, b) => a.top - b.top);
  const box = (a0, a1, s0, s1, h, feature) => {
    const am = (a0 + a1) / 2, sm = (s0 + s1) / 2, yaw = Math.atan2(ux, uz);
    return { kind: 'platform', feature, c: [am * ux - sm * uz, h / 2, am * uz + sm * ux], h: [(s1 - s0) / 2, h / 2, (a1 - a0) / 2],
      q: [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0], mu: 0.9, soft: 0, grip: 1, color: feature === 'obstacle' ? '#b98d63' : '#c8a27a' };
  };
  const boxes = [];
  if (lv.length) {
    const far = Math.max(...lv.map(l => l.far)) + 0.6, s0 = Math.min(...lv.map(l => l.s0)) - 0.35, s1 = Math.max(...lv.map(l => l.s1)) + 0.35;
    for (const l of lv) {   // the near edge 6 cm short of the first contact, and past every body that is lower than the top (a takeoff foot)
      let near = l.near - 0.06;
      for (const p of fr) for (const b of p) for (const q of b.cs) { const cq = { x: q[0], z: q[2] };
        if (q[1] - g < l.top - 0.01 && side(cq) > s0 && side(cq) < s1 && along(cq) < l.far && along(cq) + 0.01 > near) { near = along(cq) + 0.01; if (process.env.DEBUG) l.why = `body ${p.indexOf(b)} frame ${fr.indexOf(p)} corner y ${(q[1] - g).toFixed(2)}`; } }
      if (process.env.DEBUG) console.log(`  level ${l.top.toFixed(2)}: contacts along ${l.near.toFixed(2)}..${l.far.toFixed(2)}, near edge ${near.toFixed(2)} (${l.why || 'contacts'})`);
      boxes.push(box(near, far, s0, s1, l.top, `step ${l.top.toFixed(2)}`));
    }
  } else {   // a jump over: the longest flight
    const air = fr.map(p => FEET.every(i => p[i].lo - g > 0.06)); let best = null;
    for (let f = 1; f < fr.length; f++) if (air[f] && !air[f - 1]) { let e = f; while (e < fr.length && air[e]) e++; if (e < fr.length && (!best || e - f > best.e - best.f)) best = { f, e }; }
    if (best) {
      const mid = (along(fr[best.f - 1][B.PELVIS]) + along(fr[best.e][B.PELVIS])) / 2, s = side(fr[best.f][B.PELVIS]), a0 = mid - 0.15, a1 = mid + 0.15;
      let h = 0.4;
      for (const p of fr) for (const b of p) if (along(b) > a0 - 0.05 && along(b) < a1 + 0.05 && Math.abs(side(b) - s) < 0.35) h = Math.min(h, b.lo - g - 0.04);
      boxes.push(box(a0, a1, s - 0.3, s + 0.3, Math.max(0.15, h), 'obstacle'));
    }
  }
  console.log(`${name}: ${boxes.map(b => `${b.feature} (top ${(2 * b.h[1]).toFixed(2)} m, ${(2 * b.h[2]).toFixed(2)} m deep)`).join(', ') || 'no boxes'}`);
  return { name: `climb_${name}`, clip: name, bounds: [-8, 8, -8, 8], boxes };
});
writeFileSync(out, JSON.stringify({ version: 1, clips: names, rooms }));
console.log(`${out}: ${rooms.length} rooms`);
