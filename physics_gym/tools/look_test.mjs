// look_test.mjs: tools/look_test.c in body.js (node tools/look_test.mjs; diff against ./look_test)
import { World, setRoom, supportY, observeGoto, math } from '../web/body.js';
import { readFileSync } from 'fs';
setRoom(JSON.parse(readFileSync(new URL('../web/room_slope.json', import.meta.url))).boxes);
const goal = [[4.5, 4.5], [-5.0, 2.0], [1.0, -5.5]], yaw = [0.3, -2.2];
for (let s = 0; s < 16; s++) for (let y = 0; y < 2; y++) {
  const x = -6 + 0.8 * s, z = -5 + 0.65 * s * (s % 3 === 0 ? -1 : 1) + 5 * (s % 2);
  const W = new World(45, 20, 1 / 30), rel = Array.from({ length: 14 }, () => [1, 0, 0, 0]);
  W.pose([x, 0.915 + supportY(x, z, 3), z], math.qexp([0, yaw[y], 0]), rel, [0, 0, 0]);
  const g = goal[(s + y) % 3], o = observeGoto(W, 0, [g[0], supportY(g[0], g[1], 3), g[1]]);
  console.log(`${s} ${y} ` + o.slice(o.length - 26).map(v => v.toFixed(4)).join(' '));
}
