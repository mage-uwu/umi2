// ac_course.mjs: the Rooftops course (web/course.js) as a bank of layouts, checked.
//   node tools/ac_course.mjs [OUT.json] [N]    (default data/ac_course_bank.json, 64)
// Writes seeds 1..N (training) and 1001..1000+N/4 (held out), each a room doc (env.c's room format, plus its course: the
// places, the route tree, the spawn). Checks each: under ROOM_MAX boxes, inside the path helper's 16 m square, solid boxes
// not overlapping (a box resting on or touching another is fine), each incline's ends level with what they meet; and
// prints what the routes ask of her (MOVES' ratings) over the bank.
import { writeFileSync } from 'node:fs';
import { makeCourse, MOVES } from '../web/course.js';

const out = process.argv[2] || new URL('../data/ac_course_bank.json', import.meta.url).pathname, N = +(process.argv[3] || 64);
const qrot = (q, v) => {   // q (w x y z) applied to v
  const [w, x, y, z] = q, [a, b, c] = v, tx = 2 * (y * c - z * b), ty = 2 * (z * a - x * c), tz = 2 * (x * b - y * a);
  return [a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx];
};
function check(doc) {
  const bad = [], B = doc.boxes, flat = B.filter(b => b.q[0] === 1 && b.kind !== 'outer');
  if (B.length > 256) bad.push(`${B.length} boxes`);
  const ext = b => [0, 2].map(k => Math.max(...[-1, 1].flatMap(i => [-1, 1].flatMap(j => [-1, 1].map(l => Math.abs(qrot(b.q, [i * b.h[0], j * b.h[1], l * b.h[2]])[k]))))));   // its world half-extents in x and z
  for (const b of B) if (b.kind !== 'outer') { const [ex, ez] = ext(b); if (Math.abs(b.c[0]) + ex > 7.95 || Math.abs(b.c[2]) + ez > 7.95) bad.push(`${b.feature} outside`); }
  for (let i = 0; i < flat.length; i++) for (let j = i + 1; j < flat.length; j++) {
    const a = flat[i], b = flat[j], o = [0, 1, 2].map(k => a.h[k] + b.h[k] - Math.abs(a.c[k] - b.c[k]));
    if (o.every(v => v > 0.01)) bad.push(`${a.feature} overlaps ${b.feature} by ${o.map(v => v.toFixed(2)).join(' ')}`);
  }
  const top = (x, z) => Math.max(0, ...flat.filter(b => Math.abs(x - b.c[0]) < b.h[0] - 1e-6 && Math.abs(z - b.c[2]) < b.h[2] - 1e-6).map(b => b.c[1] + b.h[1]));
  for (const s of B.filter(b => b.kind === 'slope')) for (const e of [-1, 1]) {   // each end of its top: level with the ground or roof beyond it
    const p = qrot(s.q, [0, s.h[1], e * s.h[2]]).map((v, k) => v + s.c[k]), d = qrot(s.q, [0, 0, e * 0.05]);
    const ground = top(p[0] + d[0] * 2, p[2] + d[2] * 2);
    if (Math.abs(ground - p[1]) > 0.03) bad.push(`${s.feature}: end at ${p[1].toFixed(2)} m meets ${ground.toFixed(2)} m`);
  }
  return bad;
}
const seeds = [0, ...Array.from({ length: N }, (_, i) => i + 1), ...Array.from({ length: Math.ceil(N / 4) }, (_, i) => 1001 + i)];
const rooms = [], tally = {};
let failed = 0;
for (const s of seeds) {
  const doc = makeCourse(s), bad = check(doc);
  if (bad.length) { failed++; console.log(`seed ${s}:`, bad.join('; ')); }
  rooms.push({ ...doc, set: s === 0 ? 'nominal' : s > 1000 ? 'heldout' : 'train' });
  for (const e of doc.course.edges) { const t = (tally[e.move] ||= { known: 0, stretch: 0, beyond: 0, lo: 1e9, hi: -1e9 }); t[e.rating]++; t.lo = Math.min(t.lo, e.size); t.hi = Math.max(t.hi, e.size); }
}
writeFileSync(out, JSON.stringify({ version: 1, source: 'web/course.js', rooms }));
console.log(`${out}: ${rooms.length} layouts (${N} training, ${Math.ceil(N / 4)} held out, the nominal), ${failed} with problems`);
const n0 = makeCourse(0);
console.log(`nominal: ${n0.boxes.length} boxes; route tree:`);
for (const e of n0.course.edges) console.log(`  ${e.route.padEnd(9)} ${e.from.padEnd(7)} -> ${e.to.padEnd(7)} ${e.move.padEnd(7)} ${String(e.size).padStart(5)}  ${e.rating}`);
console.log('over the bank (edges by rating, size range):');
for (const [m, t] of Object.entries(tally)) console.log(`  ${m.padEnd(8)} known ${String(t.known).padStart(3)}  stretch ${String(t.stretch).padStart(3)}  beyond ${String(t.beyond).padStart(3)}   ${t.lo.toFixed(2)}..${t.hi.toFixed(2)}  (${MOVES[m].what})`);
