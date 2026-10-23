// leg_test.mjs: tools/leg_test.c in body.js (node tools/leg_test.mjs; the same lines as ./leg_test)
import { World, B, math } from '../web/body.js';
const NB = 14, W = new World(45, 20, 1 / 30), rel = Array.from({ length: NB }, () => [1, 0, 0, 0]);
W.pose([0, 1.0, 0.005], [1, 0, 0, 0], rel, [0, 0, 0]); W.limitK = 0;
for (let i = 0; i < NB; i++) W.target[i] = rel[i];
W.target[B.L_THIGH] = math.qexp([0, 0, -0.35]); W.target[B.R_THIGH] = math.qexp([0, 0, 0.35]);
W.assist = 1; W.assistX = W.b[0].x.slice(); W.assistQ = [1, 0, 0, 0]; W.assistV = [0, 0, 0];
for (let s = 0; s <= 30; s++) {
  if (s % 10 === 0) console.log(`t=${(s / 30).toFixed(2)} knees ${(W.b[B.L_SHIN].x[0] - W.b[B.R_SHIN].x[0]).toFixed(3)} feet ${(W.b[B.L_FOOT].x[0] - W.b[B.R_FOOT].x[0]).toFixed(3)} thighs ${(W.b[B.L_THIGH].x[0] - W.b[B.R_THIGH].x[0]).toFixed(3)}`);
  W.step();
}
