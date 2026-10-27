// plan_check.mjs: the parkour planner (web/planner.js) on the Rooftops bank (data/ac_course_bank.json): from the plaza to
// the bell tower's top and from there into the hay, with each profile; which routes, which moves, how long a plan takes.
//   node tools/plan_check.mjs [BANK] [-v]
import { readFileSync } from 'node:fs';
import { setRoom } from '../web/body.js';
import { plan, planMap, PROFILES } from '../web/planner.js';
const file = process.argv.find(a => a.endsWith('.json')) || new URL('../data/ac_course_bank.json', import.meta.url).pathname, verbose = process.argv.includes('-v');
const bank = JSON.parse(readFileSync(file, 'utf8')).rooms, tally = {};
let tMap = 0, tPlan = 0, nPlan = 0;
for (const doc of bank) {
  setRoom(doc.boxes); let t = performance.now(); planMap(); tMap += performance.now() - t;
  const node = id => doc.course.nodes.find(n => n.id === id), start = [doc.spawn.x, 0, doc.spawn.z], tower = node('tower').at, hay = node('hay').at;
  const line = [];
  for (const prof of Object.keys(PROFILES)) {
    t = performance.now(); const a = plan(start, tower, prof), b = plan(tower, hay, prof); tPlan += performance.now() - t; nPlan += 2;
    const k = `${prof} ${doc.set}`; (tally[k] ||= { n: 0, tower: 0, hay: 0, routes: {} }).n++;
    if (a) { tally[k].tower++; const via = a.steps.flatMap(s => s.kind === 'walk' ? [] : [`${s.kind} ${s.size}`]).join(', '); tally[k].routes[a.moves.map(m => m.kind).join('>')] = (tally[k].routes[a.moves.map(m => m.kind).join('>')] || 0) + 1; line.push(`${prof}: tower by ${via || 'walking'} (${a.cost} s)`); }
    else line.push(`${prof}: no way to the tower`);
    if (b) { tally[k].hay++; line.push(`then ${b.moves.map(m => `${m.kind} ${m.size}`).join(', ')}`); }
  }
  if (verbose || doc.seed === 0) console.log(`layout ${doc.seed}: ${line.join(' | ')}`);
}
for (const [k, v] of Object.entries(tally)) console.log(`${k.padEnd(16)} tower ${v.tower}/${v.n}, hay after ${v.hay}/${v.n}; plans: ${Object.entries(v.routes).sort((a, b) => b[1] - a[1]).map(([r, n]) => `${r || 'walk'} x${n}`).join('; ')}`);
console.log(`height map and moves ${(tMap / bank.length).toFixed(0)} ms a layout, a plan ${(tPlan / nPlan).toFixed(1)} ms`);
