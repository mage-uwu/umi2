// node floor_check.mjs crawl.json getup.json: the floor skills in the viewer's own code (body.js), headless.
// Crawl from all fours for 10 s; getting up from all fours and from lying face down, until she stands calmly
// (the viewer's hand-back test) or 12 s pass.
import { World, BODY_DEFS, B, standPose, crawlPose, applyAction, observe, loadTssPolicy, CTRL_HZ, math } from '../web/body.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
globalThis.fetch = async u => ({ json: async () => JSON.parse(readFileSync(u, 'utf8')), arrayBuffer: async () => readFileSync(u) });
const wasm = process.env.WASM || fileURLToPath(new URL('../web/tss.wasm', import.meta.url)), MUL = 2.5;
const ground = w => { let lo = 1e9; for (let i = 0; i < BODY_DEFS.length; i++) lo = Math.min(lo, w.lowest(i)); for (const b of w.b) { b.x = [b.x[0], b.x[1] - lo, b.x[2]]; b.xprev = b.x; } };
function spawn(kind) {
  const w = new World(45, 20, 1 / CTRL_HZ);
  if (kind === 'fours') { const c = crawlPose(); w.pose([0, 1, 0], c.rootQ, c.rel); ground(w); w.target = c.rel.slice(); }
  else { const r = standPose(); w.pose([0, 1, 0], math.qexp([Math.PI / 2, 0, 0]), r); ground(w); w.target = r.slice(); }   // face down, flat
  for (let k = 0; k < 15; k++) w.step();
  return w;
}
const [crawlFile, getupFile] = process.argv.slice(2);
if (crawlFile && crawlFile !== '-') {
  const pol = await loadTssPolicy(crawlFile, wasm), ref = crawlPose().rel, w = spawn('fours'), x0 = w.b[0].x.slice(); pol.reset(); let hs = 0, n = 0, rolled = -1;
  while (w.t < 10) { applyAction(w, ref, pol.act(observe(w, 0, true)), MUL); w.step(); hs += w.b[0].x[1]; n++; if (rolled < 0 && math.qrot(w.b[B.CHEST].q, [0, 0, 1])[1] > 0.3) rolled = w.t; }
  console.log(`crawl: ${Math.hypot(w.b[0].x[0] - x0[0], w.b[0].x[2] - x0[2]).toFixed(2)} m in 10 s, mean pelvis height ${(hs / n).toFixed(2)} m, ${rolled < 0 ? 'stayed face down' : `rolled over at ${rolled.toFixed(1)} s`}`);
}
if (getupFile) for (const kind of ['fours', 'prone']) {
  const pol = await loadTssPolicy(getupFile, wasm), ref = standPose(), w = spawn(kind); pol.reset(); let calm = -1, stood = -1;
  while (w.t < 12 && stood < 0) {
    applyAction(w, ref, pol.act(observe(w, 0, true)), MUL); w.step();
    const v = w.com().v, still = w.b[B.HEAD].x[1] > 1.3 && w.b[0].x[1] > 0.8 && Math.hypot(v[0], v[2]) < 0.3;
    if (!still) calm = -1; else if (calm < 0) calm = w.t; if (calm >= 0 && w.t - calm > 1) stood = w.t;
  }
  console.log(`get up from ${kind}: ${stood > 0 ? `standing calmly at ${stood.toFixed(1)} s` : `not up after 12 s (head ${w.b[B.HEAD].x[1].toFixed(2)} m, pelvis ${w.b[0].x[1].toFixed(2)} m)`}`);
}
