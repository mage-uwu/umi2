// stand_slope_test.mjs: standing (stand pose held by the muscles, no policy) on the ramps in body.js: how far the
// feet and pelvis slide downhill (node tools/stand_slope_test.mjs)
import { World, setRoom, supportY, standPose, math, B } from '../web/body.js';
import { readFileSync } from 'fs';
const MU = +(process.env.MU || 0); setRoom(JSON.parse(readFileSync(new URL('../room.json', import.meta.url))).boxes.map(b => MU ? { ...b, mu: MU } : b));
const run = (name, x, z, yaw, stick = true) => {
  const W = new World(45, 20, 1 / 30); W.stick = stick;
  const rel = standPose(); for (let i = 0; i < 14; i++) W.target[i] = rel[i];
  const y0 = supportY(x, z, 3.0);
  W.pose([x, y0 + 1.0, z], math.qexp([0, yaw, 0]), rel, [0, 0, 0]);
  let lo = 1e9; for (let i = 0; i < 14; i++) lo = Math.min(lo, W.lowest(i));
  for (const b of W.b) { b.x[1] += supportY(x, z, 3.0) - lo + 0.002; b.xprev = b.x.slice(); }
  for (let s = 0; s < 6; s++) W.step();   // settle
  const f0 = W.b[B.L_FOOT].x.slice(), p0 = W.b[0].x.slice(); let line = `${name.padEnd(22)} stick ${stick ? 'on ' : 'off'}`;
  for (let s = 1; s <= 90; s++) { W.step(); if (s % 30 === 0) { const d = math.vsub(W.b[B.L_FOOT].x, f0), e = math.vsub(W.b[0].x, p0); line += `  t=${s / 30}s foot ${Math.hypot(d[0], d[2]).toFixed(3)} pelvis ${Math.hypot(e[0], e[2]).toFixed(3)} y ${W.b[0].x[1].toFixed(2)}`; } }
  console.log(line);
};
for (const st of (process.env.OFF ? [true, false] : [true])) { run('steep slope (18 deg)', 3.6, 3.4, 0, st); run('gentle slope (8 deg)', -4.4, -5.15, Math.PI / 2, st); run('open floor', 0, 0, 0, st); }
