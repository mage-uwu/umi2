// planner.js: the parkour planner. The path helper (body.js nav*) finds ways she can walk (steps of 0.3 m at most); this
// finds ways over the room with its moves as well, and says where each begins, which way it goes and how big it is, so a
// controller can be told what is coming and a skill started at the right place.
//
// The room as a height map (PLAN_FINE m cells: the surface she would stand on), and over it a graph on PLAN_CELL m
// cells: walking between neighbours (a rise or fall of PLAN_WALK at most, as on stairs and inclines), and moves found by
// looking along each of 8 directions from every cell she can stand on, at the edge in front of her:
//   step   up onto a top she can stand on, PLAN_WALK to STEP_MAX high
//   hoist  up a ledge, STEP_MAX to profile.hoist high (a top at least 0.5 m deep)
//   vault  over a wall (less than 0.5 m thick) to profile.vault high, landing about level with the take-off
//   jump   a standing jump across a gap to profile.jump wide, landing within -0.7 to +0.3 m of the take-off's height
//   leap   a running leap across a gap to profile.leap wide (a straight run-up of PLAN_RUNUP m behind the take-off)
//   drop   down off an edge, to profile.drop deep
//   faith  the leap of faith: down into something soft (the hay), to profile.faith deep
// A profile is what the body can be asked to do (each move's largest size; 0: not at all). The cheapest way (Dijkstra:
// seconds of walking at 1 m/s, a time for each move, more the nearer it is to the profile's limit) is the plan: walks
// (way points) and moves ({ kind, from: the take-off spot, to: where she lands, u: its direction, size, edge: where the
// edge is, runup, rating: 'known' within PROFILES.now, else 'stretch' }).
import { supportY, columnHit, room } from './body.js';

export const PLAN_HALF = 8.0, PLAN_CELL = 0.25, PLAN_N = 64, PLAN_FINE = 0.0625, PLAN_WALK = 0.3, PLAN_RUNUP = 1.5;
export const PROFILES = {
   // her trackers today: the step and hoist by the climb tracker (c81: 0.62 m), the vault by c40_06 (0.3 to 0.4 m), the
   // standing jumps fjump1/2 (0.9 m)
  now: { step: 0.47, hoist: 0.77, vault: 0.45, jump: 1.0, leap: 0, drop: 0, faith: 0 },
  // her motion data at twice her strength (tools/feasibility.py): PARC's climbs and vaults to 1 m, running leaps
  athlete: { step: 0.47, hoist: 1.05, vault: 1.1, jump: 1.3, leap: 1.9, drop: 1.6, faith: 2.2 },
};
const STEP_MAX = 0.47;   // up to this a step (the knee), above it a hoist (the arms)
const MOVE_TIME = { step: 2.0, hoist: 4.0, vault: 3.0, jump: 3.5, leap: 2.5, drop: 1.5, faith: 1.5 };   // s, before the risk
const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]].map(([x, z]) => [x / Math.hypot(x, z), z / Math.hypot(x, z)]);
const NF = Math.round(2 * PLAN_HALF / PLAN_FINE);

let MAP = null;   // { boxes, H: fine heights, soft: fine soft flags, stand: per cell, moves: per cell [{...}] }
const fineAt = (x, z) => { const ix = Math.floor((x + PLAN_HALF) / PLAN_FINE), iz = Math.floor((z + PLAN_HALF) / PLAN_FINE); return ix < 0 || iz < 0 || ix >= NF || iz >= NF ? -1 : iz * NF + ix; };
const hAt = (x, z) => { const i = fineAt(x, z); return i < 0 ? 0 : MAP.H[i]; };
const cellXZ = c => [-PLAN_HALF + (c % PLAN_N + 0.5) * PLAN_CELL, -PLAN_HALF + (Math.floor(c / PLAN_N) + 0.5) * PLAN_CELL];
export const planCell = (x, z) => { const ix = Math.floor((x + PLAN_HALF) / PLAN_CELL), iz = Math.floor((z + PLAN_HALF) / PLAN_CELL); return ix < 0 || iz < 0 || ix >= PLAN_N || iz >= PLAN_N ? -1 : iz * PLAN_N + ix; };

// The height map and each cell's moves, for the room as it is (rebuilt when the room changes).
export function planMap() {
  const boxes = room();
  if (MAP && MAP.boxes === boxes) return MAP;
  const H = new Float32Array(NF * NF), soft = new Uint8Array(NF * NF);
  const softBoxes = boxes.filter(b => b.soft > 0);
  for (let i = 0; i < NF * NF; i++) {
    const x = -PLAN_HALF + (i % NF + 0.5) * PLAN_FINE, z = -PLAN_HALF + (Math.floor(i / NF) + 0.5) * PLAN_FINE, h = supportY(x, z, 3.0);
    H[i] = h;
    for (const b of softBoxes) if (Math.abs(x - b.c[0]) < b.h[0] && Math.abs(z - b.c[2]) < b.h[2] && Math.abs(b.c[1] + b.h[1] - h) < 0.02) soft[i] = 1;
  }
  MAP = { boxes, H, soft, stand: new Uint8Array(PLAN_N * PLAN_N), h: new Float32Array(PLAN_N * PLAN_N), moves: [] };
  // she can stand on a cell: the surface within PLAN_WALK of level over 0.2 m around (a stair's tread is; a wall's top,
  // or a roof's edge, is not) (not a wall's top, not an edge) and room
  // above it (nothing in the way up to 1.6 m)
  for (let c = 0; c < PLAN_N * PLAN_N; c++) {
    const [x, z] = cellXZ(c), h = supportY(x, z, 3.0); MAP.h[c] = h;   // (at its centre, where the headroom is looked at)
    let ok = !columnHit(x, z, h + 0.05, h + 1.6);
    for (const [dx, dz] of [[0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2]]) if (Math.abs(hAt(x + dx, z + dz) - h) > PLAN_WALK + 0.01) ok = false;
    MAP.stand[c] = ok ? 1 : 0;
  }
  for (let c = 0; c < PLAN_N * PLAN_N; c++) MAP.moves[c] = MAP.stand[c] ? findMoves(c) : [];
  return MAP;
}

// The moves from cell c: along each direction, the first edge in front of her and what is beyond it.
function findMoves(c) {
  const [x, z] = cellXZ(c), h0 = MAP.h[c], out = [], P = PROFILES.athlete;
  for (const u of DIRS) {
    const at = s => [x + u[0] * s, z + u[1] * s], v = s => { const [a, b] = at(s); return hAt(a, b) - h0; };
    const ds = PLAN_FINE / 2, S = 3.2;
    let a = -1; for (let s = ds; s < 0.7; s += ds) if (Math.abs(v(s)) > 0.12) { a = s; break; }
    if (a < 0.1) continue;   // no edge near in front of her (or she is on it): another cell sees it
    // square to her way: the same edge 0.2 m to either side
    const side = o => { const ox = -u[1] * o, oz = u[0] * o; for (let s = ds; s < 0.9; s += ds) if (Math.abs(hAt(x + ox + u[0] * s, z + oz + u[1] * s) - h0) > 0.12) return s; return -1; };
    if ([-0.2, 0.2].some(o => Math.abs(side(o) - a) > 0.15)) continue;
    const r = v(a + 0.1), edge = at(a), cand = [];
    const land = (s, kind, size, extra = {}) => {   // where she lands, if she can stand there
      const [lx, lz] = at(s), lc = planCell(lx, lz); if (lc < 0 || !MAP.stand[lc]) return;
      cand.push({ kind, cell: lc, size: +size.toFixed(2), from: [x, h0, z], to: [lx, MAP.h[lc], lz], u: u.slice(), edge: [edge[0], h0, edge[1]], ...extra });
    };
    if (r > PLAN_WALK) {   // up: a top she can stand on (step, hoist), or a wall to vault
      let t = -1; for (let s = a + 0.1; s < a + 1.2; s += ds) if (Math.abs(v(s) - r) > 0.12) { t = s - a; break; }
      if (t < 0 || t >= 0.5) { if (r <= P.hoist) land(a + 0.4, r <= STEP_MAX ? 'step' : 'hoist', r); }
      else if (r <= P.vault && Math.abs(v(a + t + 0.1)) <= 0.3) land(a + t + 0.45, 'vault', r);
    } else if (r < -PLAN_WALK) {   // down: a gap to jump or leap across (a far side about as high, a gap 0.3 m deeper than it
      // in between), a drop (landing just below the edge), or the leap of faith (something soft within 1.6 m out)
      let b = -1, low = 0; for (let s = a + 0.05; s < a + 2.4; s += ds) { const vs = v(s); if (vs >= -0.7 && vs <= 0.3 && low < vs - 0.3) { b = s; break; } low = Math.min(low, vs); }
      if (b > 0 && b - a >= 0.2) {
        const w = b - a, flat = (() => { for (let s = 0; s < PLAN_RUNUP; s += 0.125) if (Math.abs(v(-s)) > 0.12) return false; return true; })();
        if (w <= P.jump) land(b + 0.4, 'jump', w);
        else if (w <= P.leap && flat) land(b + 0.4, 'leap', w, { runup: PLAN_RUNUP });
      }
      let f = -1; for (let s = a + 0.1; s < a + 1.6; s += ds) { const [fx, fz] = at(s); if (MAP.soft[fineAt(fx, fz)] === 1) { f = s; break; } }
      if (f > 0 && -v(f) <= P.faith) land(f + 0.5, 'faith', -v(f));
      const d = -v(a + 0.2); if (d <= P.drop) land(a + 0.45, 'drop', d);
    }
    out.push(...cand);
  }
  return out;
}

const rating = (kind, size) => size <= (PROFILES.now[kind] ?? 0) ? 'known' : 'stretch';
// The plan from p to goal ([x, y, z]) for a profile (a name in PROFILES or the limits themselves): null if there is no way.
export function plan(p, goal, profile = 'athlete') {
  const M = planMap(), P = typeof profile === 'string' ? PROFILES[profile] : profile;
  const on = q => [q[0], supportY(q[0], q[2], (q[1] ?? 3) + 0.05), q[2]];   // the surface under a point (a pelvis, a goal on a top)
  let s = nearestStand(on(p)), g = nearestStand(on(goal)); if (s < 0 || g < 0) return null;
  const N = PLAN_N * PLAN_N, dist = new Float64Array(N).fill(Infinity), via = new Int32Array(N).fill(-1), how = new Array(N);
  dist[s] = 0; const heap = [[0, s]], push = (d, c) => { heap.push([d, c]); let i = heap.length - 1; while (i > 0) { const j = (i - 1) >> 1; if (heap[j][0] <= heap[i][0]) break; [heap[i], heap[j]] = [heap[j], heap[i]]; i = j; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[i], heap[m]] = [heap[m], heap[i]]; i = m; } } return top; };
  while (heap.length) {
    const [d, c] = pop(); if (d > dist[c]) continue; if (c === g) break;
    const cx = c % PLAN_N, cz = Math.floor(c / PLAN_N);
    for (let k = 0; k < 8; k++) {   // walking
      const [dx, dz] = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]][k], nx = cx + dx, nz = cz + dz; if (nx < 0 || nz < 0 || nx >= PLAN_N || nz >= PLAN_N) continue;
      const n = nz * PLAN_N + nx, ok = m => M.stand[m] && Math.abs(M.h[m] - M.h[c]) <= PLAN_WALK;
      if (!ok(n) || (k >= 4 && (!ok(cz * PLAN_N + nx) || !ok(nz * PLAN_N + cx)))) continue;
      const nd = d + PLAN_CELL * Math.hypot(dx, dz) * (1 + 2 * Math.abs(M.h[n] - M.h[c]));   // (a little slower up and down steps)
      if (nd < dist[n]) { dist[n] = nd; via[n] = c; how[n] = null; push(nd, n); }
    }
    for (const m of M.moves[c]) {   // the moves within the profile
      const lim = P[m.kind] || 0; if (m.size > lim) continue;
      const nd = d + MOVE_TIME[m.kind] * (1 + 3 * (m.size / lim) ** 4) + (m.runup || 0);
      if (nd < dist[m.cell]) { dist[m.cell] = nd; via[m.cell] = c; how[m.cell] = m; push(nd, m.cell); }
    }
  }
  if (!isFinite(dist[g])) return null;
  const steps = []; let walk = [];
  const at = c => { const [x, z] = cellXZ(c); return [+x.toFixed(3), +M.h[c].toFixed(3), +z.toFixed(3)]; };
  const chain = []; for (let c = g; c >= 0; c = via[c]) chain.push(c); chain.reverse();
  for (let i = 0; i < chain.length; i++) {
    const c = chain[i], m = how[c];
    if (m && i > 0) { if (walk.length) steps.push({ kind: 'walk', pts: walk }); steps.push({ ...m, rating: rating(m.kind, m.size) }); walk = [at(c)]; }
    else walk.push(at(c));
  }
  if (walk.length > 1 || !steps.length) steps.push({ kind: 'walk', pts: walk });
  return { cost: +dist[g].toFixed(2), steps, moves: steps.filter(s => s.kind !== 'walk') };
}
function nearestStand(p) {   // the cell she stands on, or the nearest she could, on the same level (within 0.75 m)
  const M = MAP, c0 = planCell(p[0], p[2]); if (c0 < 0) return -1;
  const y = p[1] != null ? p[1] : M.h[c0]; let best = -1, bd = 1e9;
  for (let dz = -3; dz <= 3; dz++) for (let dx = -3; dx <= 3; dx++) {
    const ix = c0 % PLAN_N + dx, iz = Math.floor(c0 / PLAN_N) + dz; if (ix < 0 || iz < 0 || ix >= PLAN_N || iz >= PLAN_N) continue;
    const c = iz * PLAN_N + ix; if (!M.stand[c] || Math.abs(M.h[c] - y) > 0.35) continue;
    const d = dx * dx + dz * dz; if (d < bd) { bd = d; best = c; }
  }
  return best;
}
// What to tell a controller about the plan from where she is: the next move's kind, the way to its take-off spot (her
// frame is the caller's), its size, the distance there along the plan.
export function nextMove(pl, p) {
  if (!pl || !pl.moves.length) return null;
  const m = pl.moves[0]; let d = 0, q = [p[0], p[2]];
  for (const s of pl.steps) { if (s === m) break; if (s.kind === 'walk') for (const w of s.pts) { d += Math.hypot(w[0] - q[0], w[2] - q[1]); q = [w[0], w[2]]; } }
  d += Math.hypot(m.from[0] - q[0], m.from[2] - q[1]);
  return { ...m, along: +d.toFixed(2) };
}
