// grip_test.mjs: tools/grip_test.c in body.js; the two print the same lines (node tools/grip_test.mjs, diffed against ./grip_test)
import { World, B, setRoom, GRIP_MAX, GRIP_HOLD } from '../web/body.js';
const NB = 14;
const qexp = v => { const a = Math.hypot(...v); if (a < 1e-12) return [1, 0, 0, 0]; const s = Math.sin(a / 2) / a; return [Math.cos(a / 2), v[0] * s, v[1] * s, v[2] * s]; };
const f = (x, n) => x.toFixed(n);
function run(name, hands, gripMax, letGo, limits) {
  const W = new World(45, 20, 1 / 30), rel = Array.from({ length: NB }, () => [1, 0, 0, 0]);
  rel[B.L_UARM] = qexp([0, 0, Math.PI / 2]); rel[B.R_UARM] = qexp([0, 0, -Math.PI / 2]);
  W.pose([0, 0.915 + 0.6, 0.005], [1, 0, 0, 0], rel, [0, 0, 0]); W.kpScale = 0.05;
  for (let i = 0; i < NB; i++) W.target[i] = rel[i];
  const hl = W.handPoint(0);
  setRoom([{ c: [0, hl[1] + 0.021, hl[2]], h: [0.6, 0.02, 0.02], q: [1, 0, 0, 0], mu: 0.9, soft: 0, grip: GRIP_HOLD }]); W.gripMax = gripMax;
  if (!limits) W.limitK = 0;
  let line = name.padEnd(22);
  for (let s = 0; s <= 90; s++) {
    W.gripCmd[0] = 1; W.gripCmd[1] = hands > 1 ? 1 : 0;
    if (letGo > 0 && s >= letGo * 30) W.gripCmd[0] = W.gripCmd[1] = 0;
    if (s % 15 === 0) line += `  t=${f(s / 30, 1)} y=${f(W.b[0].x[1], 2)} grip=${+(W.gripBox[0] >= 0)}${+(W.gripBox[1] >= 0)} f=${f(W.gripForce[0] + W.gripForce[1], 0)}`;
    W.step();
  }
  console.log(line);
}
run('both hands', 2, GRIP_MAX, 0, 0); run('one hand', 1, GRIP_MAX, 0, 0); run('weak grip (150 N)', 2, 150, 0, 0);
run('let go at 1 s', 2, GRIP_MAX, 1.0, 0); run('both, joint limits on', 2, GRIP_MAX, 0, 1);
