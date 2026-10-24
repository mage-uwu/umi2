// jump_course.mjs: a room bank of jump courses, each built around one jump clip so its flight crosses a gap: platform A
// under the run-up's footfalls, platform B under the landing's and on 1.5 m past the clip's end, the gap between them
// where both feet are off the ground, the tops at the same height (0.3 to 1.0 m over the bank). Room r is for clip
// r % (number of clips), in the clip's own frame (its root at frame 0 over the origin), as env.c's course mode plays it.
//   node tools/jump_course.mjs data/cmu_jump.bin run_leap1,run_leap2,... out.json [rooms]
import { World, loadClips, B } from '../web/body.js';
import { readFileSync, writeFileSync } from 'fs';
globalThis.fetch = async u => { const buf = readFileSync(u); return { ok: true, arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) }; };
const [file, names, out, nRooms = '32'] = process.argv.slice(2);
const clips = await loadClips(file), list = names.split(',');
const UP = 0.06;   // a foot is off the ground when its lowest corner is this high above the clip's ground
const courses = list.map(name => {
  const c = clips[name]; if (!c) throw new Error('no clip ' + name);
  c.shift = [0, 0, 0]; const r0 = c.at(0).root; c.shift = [-r0[0], 0, -r0[2]];
  const W = new World(45, 20, 1 / 30), fr = [];
  for (let f = 0; f < c.frames; f++) { const a = c.at(f / c.fps); W.pose(a.root, a.rel[0], a.rel); fr.push([B.L_FOOT, B.R_FOOT].map(i => ({ y: W.lowest(i), x: W.b[i].x[0], z: W.b[i].x[2] }))); }
  const g = Math.min(...fr.flat().map(p => p.y));   // the clip's ground (its lowest foot corner)
  const air = fr.map(p => p.every(q => q.y - g > UP));
  // the longest flight: both feet up, by horizontal distance between the last foot down before and the first after
  let best = null;
  for (let f = 1; f < fr.length; f++) if (air[f] && !air[f - 1]) {
    let e = f; while (e < fr.length && air[e]) e++; if (e >= fr.length) break;
    const off = fr[f - 1].reduce((a, b) => (a.y < b.y ? a : b)), on = fr[e].reduce((a, b) => (a.y < b.y ? a : b));
    const d = Math.hypot(on.x - off.x, on.z - off.z); if (!best || d > best.d) best = { f, e, off, on, d };
  }
  const end = fr[fr.length - 1], tail = { x: (end[0].x + end[1].x) / 2, z: (end[0].z + end[1].z) / 2 };
  return { name, g, best, start: { x: (fr[0][0].x + fr[0][1].x) / 2, z: (fr[0][0].z + fr[0][1].z) / 2 }, tail, T: (c.frames - 1) / c.fps };
});
const box = (a, b, h, w, ext0, ext1, feature) => {   // a box from a to b (horizontal), extended ext0 before a and ext1 past b
  const dx = b.x - a.x, dz = b.z - a.z, L = Math.hypot(dx, dz) || 1e-6, ux = dx / L, uz = dz / L, yaw = Math.atan2(ux, uz);
  const len = L + ext0 + ext1, mid = { x: a.x - ux * ext0 + ux * len / 2, z: a.z - uz * ext0 + uz * len / 2 };
  return { kind: 'platform', feature, c: [mid.x, h / 2, mid.z], h: [w / 2, h / 2, len / 2], q: [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0], mu: 0.9, soft: 0, grip: 1, color: feature === 'takeoff' ? '#d2ae86' : '#c8a27a' };
};
const rooms = [];
for (let r = 0; r < +nRooms; r++) {
  const k = r % courses.length, C = courses[k], h = 0.3 + 0.7 * Math.floor(r / courses.length) / Math.max(1, Math.floor((+nRooms - 1) / courses.length));
  const { off, on } = C.best, M = 0.12;   // the gap's edges M clear of the takeoff and landing feet
  const dir = { x: on.x - off.x, z: on.z - off.z }, L = Math.hypot(dir.x, dir.z) || 1e-6, u = { x: dir.x / L, z: dir.z / L };
  const A = box(C.start, { x: off.x + u.x * M, z: off.z + u.z * M }, h, 1.0, 1.0, 0, 'takeoff');
  const Bx = box({ x: on.x - u.x * M, z: on.z - u.z * M }, C.tail, h, 1.0, 0, 1.5, 'landing');
  rooms.push({ name: `jump_${C.name}_${h.toFixed(2)}`, clip: C.name, height: h, gap: Math.max(0, L - 2 * M), bounds: [-8, 8, -8, 8], boxes: [A, Bx] });
}
writeFileSync(out, JSON.stringify({ version: 2, clips: list, rooms }, null, 0));
for (const C of courses) console.log(`${C.name}: ${C.T.toFixed(1)} s, ground ${C.g.toFixed(3)}, flight frames ${C.best?.f}-${C.best?.e}, takeoff to landing ${C.best?.d.toFixed(2)} m (gap ${Math.max(0, C.best?.d - 0.24).toFixed(2)} m)`);
