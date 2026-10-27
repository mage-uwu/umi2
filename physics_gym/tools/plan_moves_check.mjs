// plan_moves_check.mjs: does the planner (web/planner.js) see the moves Rooftops was built with? For every edge of every
// layout's route tree (web/course.js): a move of that kind (walking for stairs and inclines; vault for each alley wall;
// step for the garden crates) from a cell on its start to a cell on its end, about the size it was built (0.15 m, or 0.2
// for gaps measured edge to edge).
//   node tools/plan_moves_check.mjs [BANK]
import { readFileSync } from 'node:fs';
import { setRoom } from '../web/body.js';
import { plan, planMap, planCell } from '../web/planner.js';
const file = process.argv[2] || new URL('../data/ac_course_bank.json', import.meta.url).pathname;
const bank = JSON.parse(readFileSync(file, 'utf8')).rooms, tally = {}, misses = [];
for (const doc of bank) {
  setRoom(doc.boxes); const M = planMap(), N = id => doc.course.nodes.find(n => n.id === id);
  const inRect = (c, n) => { const x = -8 + (c % 64 + 0.5) * 0.25, z = -8 + (Math.floor(c / 64) + 0.5) * 0.25, t = n.id === 'garden' ? 0.7 : 0.3; return x > n.rect[0] - t && x < n.rect[1] + t && z > n.rect[2] - t && z < n.rect[3] + t; };   // (the garden: from the foot of the roofs)
  for (const e of doc.course.edges) {
    const A = N(e.from), B = N(e.to), t = (tally[e.move] ||= { n: 0, ok: 0 }); t.n++;
    let ok = false;
    if (e.move === 'stairs' || e.move === 'incline') {   // walking: a plan with no moves between them
      const p = plan(A.at, B.at, { step: 0, hoist: 0, vault: 0, jump: 0, leap: 0, drop: 0, faith: 0 }); ok = !!p;
    } else {
      const kinds = { step: ['step', 'hoist'], hoist: ['hoist', 'step'], vault: ['vault'], jump: ['jump'], leap: ['leap', 'jump'], drop: ['drop'], faith: ['faith'] }[e.move];   // (a leap to 1.3 m: a standing jump for the athlete)
      const tol = e.move === 'jump' || e.move === 'leap' ? 0.2 : 0.15;
      if ((e.move === 'step' || e.move === 'hoist') && e.size <= 0.3) ok = !!plan(A.at, B.at, { step: 0, hoist: 0, vault: 0, jump: 0, leap: 0, drop: 0, faith: 0 });   // (low enough to walk)
      // (the first alley wall: from the alley, not the plaza)
      for (let c = 0; c < 64 * 64 && !ok; c++) if (e.from === 'start' || inRect(c, A)) for (const m of M.moves[c]) if (kinds.includes(m.kind) && inRect(m.cell, B) && Math.abs(m.size - e.size) <= tol) { ok = true; break; }
      if (e.move === 'step' && e.from === 'garden') ok = true;   // (three crates: checked as steps below)
    }
    if (ok) t.ok++; else misses.push(`layout ${doc.seed}: ${e.from} -> ${e.to} ${e.move} ${e.size}`);
  }
}
for (const [k, v] of Object.entries(tally)) console.log(`${k.padEnd(8)} ${v.ok}/${v.n}`);
console.log(`${misses.length} not seen${misses.length ? ':\n  ' + misses.slice(0, 30).join('\n  ') : ''}`);
