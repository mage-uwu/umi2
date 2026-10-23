// node walk_check.mjs policy.json [assist] [seconds]: the viewer's walk loop (index.html control()) headless,
// with the viewer's own code (body.js), for an MLP/SSM policy JSON or a nanotss one (type "nanotss").
import { World, applyAction, observe, fallen, loadPolicy, loadTssPolicy, loadClips, poseFromClip, setAssist, CTRL_HZ } from '../web/body.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
const here = p => fileURLToPath(new URL(p, import.meta.url));
globalThis.fetch = async u => ({ json: async () => JSON.parse(readFileSync(u, 'utf8')), arrayBuffer: async () => readFileSync(u) });
const file = process.argv[2], assist = +(process.argv[3] ?? 0.07), secs = +(process.argv[4] ?? 20);
const pol = JSON.parse(readFileSync(file, 'utf8')).type === 'nanotss' ? await loadTssPolicy(file, process.env.WASM || here('../web/tss.wasm')) : await loadPolicy(file);
const clip = (await loadClips(here('../web/motions_ref.json'))).walk;
const world = new World(45, 20, 1 / CTRL_HZ); let ref = poseFromClip(world, clip, 0).rel; pol.reset();
const x0 = world.b[0].x.slice();
while (world.t < secs) {
  setAssist(world, clip, world.t, assist);
  applyAction(world, ref, pol.act(observe(world, 2 * Math.PI * world.t / clip.T)));
  world.step(); ref = clip.at(world.t).rel;
  if (fallen(world)) break;
}
const x = world.b[0].x;
console.log(`${file}: upright ${world.t.toFixed(1)} s of ${secs}, travelled ${Math.hypot(x[0] - x0[0], x[2] - x0[2]).toFixed(2)} m`);
