// node rollout_check.mjs policy.json steps  -> same columns as tools/rollout.c
import { World, standPose, applyAction, observe, fallen, loadPolicy, B, BODY_DEFS, ACT_DIM } from '../web/body.js';
import { readFileSync } from 'fs';
globalThis.fetch = async u => ({ json: async () => JSON.parse(readFileSync(u, 'utf8')) });
const pol = await loadPolicy(process.argv[2]), steps = +process.argv[3];
const W = new World(45, 20, 1 / 30), ref = standPose();
W.pose(BODY_DEFS[0].center, ref[0], ref);
let lo = 1e9; for (let i = 0; i < 14; i++) lo = Math.min(lo, W.lowest(i));
for (const b of W.b) { b.x = [b.x[0], b.x[1] - lo, b.x[2]]; b.xprev = b.x; }
W.target = ref.slice();
for (let s = 1; s <= steps; s++) {
  const ob = observe(W), a = pol.act(ob);
  if (process.env.DUMP && s <= +process.env.DUMP) { const z = ob.map((v, i) => Math.max(-5, Math.min(5, (v - pol.mean[i]) / pol.std[i]))); z.forEach((v, i) => console.error('o' + i, v.toFixed(5))); a.forEach((v, k) => console.error('a' + k, v.toFixed(5))); }
  applyAction(W, ref, a); W.step();
  console.log(s, W.b[0].x[1].toFixed(6), W.b[B.HEAD].x[1].toFixed(6), a[0].toFixed(6));
  if (fallen(W)) break;
}
