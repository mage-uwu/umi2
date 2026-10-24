// slope_test.mjs: the lying (friction-only) half of tools/slope_test.c in body.js (node tools/slope_test.mjs)
import { World, setRoom, supportY, math, B } from '../web/body.js';
import { readFileSync } from 'fs';
setRoom(JSON.parse(readFileSync(new URL('../room.json', import.meta.url))).boxes);
const run = (name, x, z, yaw) => {
  const W = new World(45, 20, 1 / 30), rel = Array.from({ length: 14 }, () => [1, 0, 0, 0]), y0 = supportY(x, z, 3.0);
  W.pose([x, y0 + 0.25, z], math.qmul(math.qexp([0, yaw, 0]), math.qexp([-Math.PI / 2, 0, 0])), rel, [0, 0, 0]); W.kpScale = 0.02;
  for (let i = 0; i < 14; i++) W.target[i] = rel[i];
  const p0 = W.b[0].x.slice(); let line = name.padEnd(20) + ` ground ${y0.toFixed(3)}`;
  for (let s = 1; s <= 90; s++) { W.step(); if (s % 30 === 0) { const d = math.vsub(W.b[0].x, p0); line += `  t=${s / 30}s drift ${Math.hypot(d[0], d[2]).toFixed(3)} (y ${W.b[0].x[1].toFixed(3)})`; } }
  console.log(line);
};
console.log('lying limp (friction alone):');
run('steep slope (18 deg)', 3.6, 3.4, Math.PI / 2); run('gentle slope (8 deg)', -4.4, -5.15, 0); run('flat platform', 3.6, 5.0, 0); run('open floor', 0, 0, 0);
