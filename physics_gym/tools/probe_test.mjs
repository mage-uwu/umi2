// probe_test.mjs: tools/probe_test.c in body.js (node tools/probe_test.mjs; diff against ./probe_test)
import { World, setRoom, supportY, observeGoto, math } from '../web/body.js';
import { readFileSync } from 'fs';
setRoom(JSON.parse(readFileSync(new URL('../room.json', import.meta.url))).boxes);
const spot = [[-3.6, 1.6], [3.6, 1.6], [0, -1.8], [-1.4, 3.6], [2.0, -0.4], [-2.3, -1.8], [5.5, -0.6], [0, 6.5]], yaw = [0, 0.6, -2.5];
for (let s = 0; s < 8; s++) for (let y = 0; y < 3; y++) {
  const W = new World(45, 20, 1 / 30), rel = Array.from({ length: 14 }, () => [1, 0, 0, 0]);
  W.pose([spot[s][0], 0.915 + supportY(spot[s][0], spot[s][1], 3), spot[s][1]], math.qexp([0, yaw[y], 0]), rel, [0, 0, 0]);
  const o = observeGoto(W, 0, null);
  console.log(`${s} ${y} ` + o.slice(o.length - 16).map(v => v.toFixed(4)).join(' '));
}
