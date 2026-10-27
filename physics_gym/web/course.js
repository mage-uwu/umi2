// course.js: Rooftops, a parkour course after Assassin's Creed: a start plaza, routes that branch and join again over
// roofs, an alley, a courtyard and a garden, a bell tower to reach and a leap of faith from it into a hay cart. Each seed
// is another layout of the same plan: its gaps, heights, walls, stairs and inclines drawn from ranges, the whole turned by
// a quarter turn and mirrored, so a controller that clears it has to generalise rather than remember one course
// (tools/ac_course.mjs writes a bank: seeds 1 to 64 for training, 1001 on held out; seed 0 is the nominal course).
//
// The routes, a tree from the plaza (each edge one way of moving; the routes join on the central roof):
//   rooftops:  stairs up to the west roof -> a standing jump across a gap -> a running leap to the next roof
//              (or a drop down to the garden) -> an incline up to the central roof
//   alley:     three walls of rising height to vault -> an incline up to the market roof -> a crate -> a hoist up
//              to the central roof
//   courtyard: an incline -> a step -> a step up onto the central roof
//   garden:    (dropped into from the roofs) crates of rising height -> a hoist up the bell tower's side
//   finish:    a hoist from the central roof up the bell tower; the leap of faith from its top into the hay
// Complete when she has stood on top of the bell tower (by any route) and then landed in the hay; the sync points on the
// way are optional, as is which routes she takes.
//
// Each edge is rated against what her motions cover (MOVES): 'known' (a skill of hers: the walk on stairs and inclines,
// the standing jumps fjump1/2 across 0.9 m, the vaults c40_06 over 0.3 to 0.4 m, the hoists c81 up 0.6 m), 'stretch' (her
// motion data has it but no tracker of hers does it alone yet: running leaps to 1.5 m, PARC's vaults and climbs to 1 m) or
// 'beyond' (nothing she has does it).

export function rng(seed) {   // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const COLOR = { roof: '#c47a4c', roof2: '#b86a3f', tower: '#9c5638', stair: '#d8b48a', slope: '#c9a27a', wall: '#d9cdb4', crate: '#a8805a',
  hay: '#e2c35a', warehouse: '#a58d74', outer: '#c9cde0' };
// The moves, with the sizes her motions cover: known up to, stretch up to (beyond that: beyond).
export const MOVES = {
  stairs: { what: 'stairs (rise of a step, m)', known: 0.2, stretch: 0.3 },
  incline: { what: 'incline (degrees)', known: 15, stretch: 22 },
  step: { what: 'step up (m)', known: 0.45, stretch: 0.6 },
  jump: { what: 'standing jump across a gap (m)', known: 1.0, stretch: 1.3 },
  leap: { what: 'running leap across a gap (m)', known: 0, stretch: 1.6 },
  vault: { what: 'vault a wall (height, m)', known: 0.45, stretch: 1.0 },
  hoist: { what: 'hoist up a ledge (height, m)', known: 0.7, stretch: 1.0 },
  drop: { what: 'drop down (m)', known: 0.6, stretch: 1.3 },
  faith: { what: 'leap of faith into hay (drop, m)', known: 0, stretch: 2.0 },
};
export const rate = (move, size) => size <= MOVES[move].known ? 'known' : size <= MOVES[move].stretch ? 'stretch' : 'beyond';

// One layout. Plan coordinates: x east, z north, the plaza in the south-west; T() turns and mirrors them for the seed.
export function makeCourse(seed = 0) {
  const R = rng(seed * 7919 + 17), base = seed === 0;
  const U = (lo, hi, nominal) => base ? nominal : lo + (hi - lo) * R();
  const turn = base ? 0 : Math.floor(R() * 4), mirror = !base && R() < 0.5;
  const T = (x, z) => { if (mirror) x = -x; for (let k = 0; k < turn; k++) [x, z] = [z, -x]; return [x, z]; };
  const boxes = [], nodes = [], edges = [];
  const r4 = v => +v.toFixed(4);
  const box = (kind, feature, c, h, q, color, extra = {}) => boxes.push({ kind, feature, c: c.map(r4), h: h.map(r4), q: q.map(v => +v.toFixed(6)), mu: 0.9, soft: 0, grip: 0, color, ...extra });
  const rect = (x0, x1, z0, z1) => { const [a, b] = T(x0, z0), [c, d] = T(x1, z1); return [Math.min(a, c), Math.max(a, c), Math.min(b, d), Math.max(b, d)]; };
  const block = (kind, feature, x0, x1, z0, z1, top, color, extra) => {   // from the floor up to `top` over a plan rectangle
    const [a, b, c, d] = rect(x0, x1, z0, z1); box(kind, feature, [(a + b) / 2, top / 2, (c + d) / 2], [(b - a) / 2, top / 2, (d - c) / 2], [1, 0, 0, 0], color, extra);
  };
  // An incline: a plate 0.1 m thick whose top runs from plan (x0, z0) at height y0 to (x1, z1) at y1, `width` across. Turned
  // by its yaw (local z along the incline), then tipped up by its pitch about its own x; returns its angle in degrees.
  const ramp = (feature, x0, z0, x1, z1, y0, y1, width) => {
    const [ax, az] = T(x0, z0), [bx, bz] = T(x1, z1), L = Math.hypot(bx - ax, bz - az), yaw = Math.atan2(bx - ax, bz - az), p = Math.atan2(y1 - y0, L), t = 0.1;
    const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2), ca = Math.cos(-p / 2), sa = Math.sin(-p / 2);
    const q = [cy * ca, cy * sa, sy * ca, -sy * sa];
    // its centre: the middle of its top, down half its thickness along its normal (-sin p sin yaw, cos p, -sin p cos yaw)
    const c = [(ax + bx) / 2 + (t / 2) * Math.sin(p) * Math.sin(yaw), (y0 + y1) / 2 - (t / 2) * Math.cos(p), (az + bz) / 2 + (t / 2) * Math.sin(p) * Math.cos(yaw)];
    box('slope', feature, c, [width / 2, t / 2, Math.hypot(L, y1 - y0) / 2], q, COLOR.slope);
    return p * 180 / Math.PI;
  };
  // Stairs: n steps of `rise` and 0.3 m treads from plan (x0, z0) north, `width` across.
  const slab = (kind, feature, x0, x1, z0, z1, y0, y1, color, extra) => {   // from y0 up to y1 (resting on what is under it)
    const [a, b, c, d] = rect(x0, x1, z0, z1); box(kind, feature, [(a + b) / 2, (y0 + y1) / 2, (c + d) / 2], [(b - a) / 2, (y1 - y0) / 2, (d - c) / 2], [1, 0, 0, 0], color, extra);
  };
  const stairs = (feature, x0, z0, n, rise, width) => { for (let i = 0; i < n; i++) block('stair', feature, x0 - width / 2, x0 + width / 2, z0 + 0.3 * i, z0 + 0.3 * (i + 1), (i + 1) * rise, COLOR.stair); };
  // The places on the way (to track her by): a plan rectangle and the height of its top; at: the point to draw an edge to.
  const node = (id, name, kind, x0, x1, z0, z1, top) => { const r = rect(x0, x1, z0, z1); nodes.push({ id, name, kind, rect: r.map(v => +v.toFixed(3)), top: +top.toFixed(3), at: [+((r[0] + r[1]) / 2).toFixed(3), +top.toFixed(3), +((r[2] + r[3]) / 2).toFixed(3)] }); };
  const edge = (from, to, move, size, route) => edges.push({ from, to, move, size: +size.toFixed(2), rating: rate(move, size), route });

  // ---- this seed's sizes ----
  const rise1 = U(0.12, 0.2, 0.15), n1 = Math.round(U(1.0, 1.35, 1.2) / rise1), roof1 = n1 * rise1;   // the west roof, as high as its stairs
  const gap1 = U(0.7, 1.1, 0.9), roof2 = roof1 + U(-0.1, 0.1, 0);
  const gap2 = U(1.2, 1.8, 1.4), roof3 = roof2 - U(0.1, 0.3, 0.2);
  const roof4 = U(1.45, 1.75, 1.6), tower = roof4 + U(0.5, 0.7, 0.6), hay = 0.5;
  const walls = [U(0.35, 0.6, 0.5), U(0.6, 0.85, 0.8), U(0.85, 1.1, 1.0)];
  const market = U(0.5, 0.7, 0.6), crate = market + U(0.4, 0.6, 0.5);
  const yard = U(0.7, 0.9, 0.8), stepTop = roof4 - U(0.3, 0.45, 0.4);
  const crates = [0.5, 1.0, 1.5].map(v => v + U(-0.08, 0.08, 0));

  // ---- the plaza, and the town wall round it all ----
  node('start', 'Plaza', 'start', -7.6, -4.0, -7.9, -5.0, 0);
  for (const [cx, cz, hx, hz] of [[0, 8, 8.1, 0.1], [0, -8, 8.1, 0.1], [8, 0, 0.1, 8.1], [-8, 0, 0.1, 8.1]]) {
    const [a, b, c, d] = rect(cx - hx, cx + hx, cz - hz, cz + hz); box('outer', 'town wall', [(a + b) / 2, 1.2, (c + d) / 2], [(b - a) / 2, 1.2, (d - c) / 2], [1, 0, 0, 0], COLOR.outer, { mu: 0.6 });
  }
  // ---- rooftops: the west roofs (going north), a leap east, an incline up ----
  const z1 = -5.0 + n1 * 0.3;   // where the stairs end and the west roof starts
  stairs('stairs to the west roof', -6.5, -5.0, n1, rise1, 2.2);
  block('platform', 'west roof', -7.6, -5.4, z1, z1 + 2.0, roof1, COLOR.roof, { grip: 1 });
  node('roof1', 'West roof', 'roof', -7.6, -5.4, z1, z1 + 2.0, roof1);
  const z2a = z1 + 2.0 + gap1, z2b = Math.min(Math.max(z2a + 1.8, 2.4), 2.8);
  block('platform', 'west roof across the gap', -7.6, -5.4, z2a, z2b, roof2, COLOR.roof2, { grip: 1 });
  node('roof2', 'West roof, sync point', 'sync', -7.6, -5.4, z2a, z2b, roof2);
  const x3a = -5.4 + gap2, x3b = -2.0;
  block('platform', 'leap roof', x3a, x3b, z2a, z2b, roof3, COLOR.roof, { grip: 1 });
  node('roof3', 'Leap roof', 'roof', x3a, x3b, z2a, z2b, roof3);
  block('platform', 'under the incline', x3b, 0.5, z2a, z2b, roof3, COLOR.roof2);   // solid: no way in under it
  const angR = ramp('incline to the central roof', x3b, (z2a + z2b) / 2, 0.5, (z2a + z2b) / 2, roof3, roof4, 1.4);
  // ---- the central roof, the bell tower, the hay cart ----
  block('platform', 'central roof', 0.5, 4.5, -1.5, 2.4, roof4, COLOR.roof2, { grip: 1 });
  node('roof4', 'Central roof', 'roof', 0.5, 4.5, -1.5, 2.4, roof4);
  block('platform', 'bell tower', 1.5, 3.5, 2.4, 4.4, tower, COLOR.tower, { grip: 1 });
  node('tower', 'Bell tower', 'tower', 1.5, 3.5, 2.4, 4.4, tower);
  block('foam', 'hay cart', 1.7, 3.3, 5.2, 6.6, hay, COLOR.hay, { mu: 1.0, soft: 0.0003 });
  node('hay', 'Hay cart', 'hay', 1.7, 3.3, 5.2, 6.6, hay);
  // ---- the alley (south, going east): three walls, an incline up to the market roof, a crate ----
  block('platform', 'warehouse', -4.0, 4.4, -5.6, -3.9, 2.6, COLOR.warehouse);
  walls.forEach((h, i) => {
    block('wall', `alley wall, ${h.toFixed(2)} m`, -2.6 + 2.5 * i, -2.35 + 2.5 * i, -7.9, -5.6, h, COLOR.wall);
    node(`alley${i + 1}`, `Alley past wall ${i + 1}`, 'street', -2.35 + 2.5 * i, i < 2 ? -0.1 + 2.5 * i : 5.6, -7.9, -5.6, 0);
  });
  const angM = ramp('incline to the market roof', 6.5, -7.2, 6.5, -3.8, 0, market, 1.4);
  block('platform', 'market roof', 3.0, 7.6, -3.8, -1.5, market, COLOR.roof, { grip: 1 });
  node('market', 'Market roof, sync point', 'sync', 4.4, 7.6, -3.8, -2.3, market);
  slab('platform', 'market crate', 3.2, 4.4, -2.3, -1.5, market, crate, COLOR.crate, { grip: 1 });
  node('crate', 'Market crate', 'roof', 3.2, 4.4, -2.3, -1.5, crate);
  // ---- the courtyard (middle): an incline, a landing, a step ----
  const angY = ramp('courtyard incline', -4.4, -2.9, -1.0, -2.9, 0, yard, 1.4);
  block('platform', 'courtyard landing', -1.0, 0.5, -3.6, -1.5, yard, COLOR.roof2, { grip: 1 });
  node('landing', 'Courtyard landing', 'roof', -1.0, 0.5, -3.6, -1.5, yard);
  block('platform', 'courtyard step', -1.0, 0.5, -1.5, z2a, stepTop, COLOR.roof, { grip: 1 });
  node('yard', 'Courtyard step, sync point', 'sync', -1.0, 0.5, -1.5, z2a, stepTop);
  // ---- the garden (north): crates up the bell tower's west side ----
  const cx = [-2.3, -1.03, 0.24, 1.5];   // side by side, the last against the tower
  crates.forEach((h, i) => block('platform', `garden crate, ${h.toFixed(2)} m`, cx[i], cx[i + 1], 3.1, 4.3, h, COLOR.crate, { grip: 1 }));
  node('garden', 'Garden', 'street', -7.6, 1.2, Math.max(z2b, 3.0) + 0.3, 7.6, 0);
  node('crate3', 'Top garden crate, sync point', 'sync', 0.24, 1.5, 3.1, 4.3, crates[2]);

  // ---- the route tree ----
  edge('start', 'roof1', 'stairs', rise1, 'rooftops');
  edge('roof1', 'roof2', 'jump', gap1, 'rooftops');
  edge('roof2', 'roof3', 'leap', gap2, 'rooftops');
  edge('roof3', 'roof4', 'incline', angR, 'rooftops');
  edge('roof2', 'garden', 'drop', roof2, 'garden');
  edge('roof3', 'garden', 'drop', roof3, 'garden');
  edge('garden', 'crate3', 'step', Math.max(crates[0], crates[1] - crates[0], crates[2] - crates[1]), 'garden');
  edge('crate3', 'tower', 'hoist', tower - crates[2], 'garden');
  walls.forEach((h, i) => edge(i ? `alley${i}` : 'start', `alley${i + 1}`, 'vault', h, 'alley'));
  edge('alley3', 'market', 'incline', angM, 'alley');
  edge('market', 'crate', 'step', crate - market, 'alley');
  edge('crate', 'roof4', 'hoist', roof4 - crate, 'alley');
  edge('start', 'landing', 'incline', angY, 'courtyard');
  edge('landing', 'yard', 'step', stepTop - yard, 'courtyard');
  edge('yard', 'roof4', 'step', roof4 - stepTop, 'courtyard');
  edge('roof4', 'tower', 'hoist', tower - roof4, 'finish');
  edge('tower', 'hay', 'faith', tower - hay, 'finish');

  const [sx, sz] = T(-6.2, -6.6), [nx, nz] = T(0, 1);
  return {
    version: 2, name: `Rooftops ${seed}`, seed, floor_y: 0, bounds: [-8.1, 8.1, -8.1, 8.1],
    grip: { 0: 'none', 1: 'ledge: its top edges can be held', 2: 'handhold' },
    spawn: { x: +sx.toFixed(3), z: +sz.toFixed(3), yaw: +Math.atan2(nx, nz).toFixed(4) },   // in the plaza, facing (plan) north
    boxes, course: { nodes, edges, finish: ['tower', 'hay'], turn, mirror },
  };
}
