// ac_course_map.mjs: plan maps of Rooftops layouts (web/course.js) as one SVG: the boxes from above, shaded by height
// (labelled in m), the course's places, and its route tree with each move's size, coloured by its rating.
//   node tools/ac_course_map.mjs OUT.svg [seed ...]    (default seeds: 0 1 2 1001)
import { writeFileSync } from 'node:fs';
import { makeCourse } from '../web/course.js';

const [out, ...rest] = process.argv.slice(2), seeds = rest.length ? rest.map(Number) : [0, 1, 2, 1001];
const S = 34, M = 24, W = 16.2 * S + 2 * M, HH = W + 60;   // px per m, margin; one panel
const RATE = { known: '#2f8f46', stretch: '#d08a10', beyond: '#c03030' }, UNIT = { incline: '°', stairs: ' m/step' };
const qrot = (q, v) => {
  const [w, x, y, z] = q, [a, b, c] = v, tx = 2 * (y * c - z * b), ty = 2 * (z * a - x * c), tz = 2 * (x * b - y * a);
  return [a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx];
};
const shade = h => { const t = Math.min(1, h / 2.4), c = (a, b) => Math.round(a + (b - a) * t); return `rgb(${c(236, 150)},${c(222, 80)},${c(200, 50)})`; };
function panel(seed, ox, oy) {
  const doc = makeCourse(seed), C = doc.course, px = x => ox + M + (x + 8.1) * S, pz = z => oy + 40 + M + (8.1 - z) * S;   // north up
  const g = [`<rect x="${ox}" y="${oy}" width="${W}" height="${HH}" fill="#fbfaf7" stroke="#ddd"/>`,
    `<text x="${ox + M}" y="${oy + 26}" font-size="17" font-weight="700">Rooftops, layout ${seed}${seed === 0 ? ' (nominal)' : seed > 1000 ? ' (held out)' : ''}</text>`,
    `<text x="${ox + W - M}" y="${oy + 26}" font-size="12" text-anchor="end" fill="#666">turned ${C.turn * 90}°${C.mirror ? ', mirrored' : ''}</text>`,
    `<rect x="${px(-8.1)}" y="${pz(8.1)}" width="${16.2 * S}" height="${16.2 * S}" fill="#e9eef8"/>`];
  const sorted = doc.boxes.filter(b => b.kind !== 'outer').map(b => ({ b, top: Math.max(...[-1, 1].flatMap(i => [-1, 1].map(l => qrot(b.q, [i * b.h[0], b.h[1], l * b.h[2]])[1]))) + b.c[1] })).sort((a, b) => a.top - b.top);
  for (const { b, top } of sorted) {
    const pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([i, l]) => { const v = qrot(b.q, [i * b.h[0], b.h[1], l * b.h[2]]); return [px(b.c[0] + v[0]), pz(b.c[2] + v[2])]; });
    const fill = b.kind === 'foam' ? '#f1d76a' : b.kind === 'slope' ? '#d9c3a0' : b.kind === 'stair' ? '#e3cfae' : b.kind === 'wall' ? '#8e8a80' : shade(top);
    g.push(`<polygon points="${pts.map(p => p.map(v => v.toFixed(1)).join(',')).join(' ')}" fill="${fill}" stroke="#5a4a3a" stroke-width="${b.kind === 'stair' ? 0.4 : 0.9}"/>`);
    if (b.kind !== 'stair' && b.kind !== 'slope' && Math.min(b.h[0], b.h[2]) * 2 * S > 22) g.push(`<text x="${px(b.c[0]).toFixed(1)}" y="${(pz(b.c[2]) + 4).toFixed(1)}" font-size="10" text-anchor="middle" fill="#333">${top.toFixed(2)}</text>`);
  }
  const at = id => C.nodes.find(n => n.id === id).at;
  for (const e of C.edges) {   // a curve from place to place, bowed to one side so the parallel ones part
    const [ax, , az] = at(e.from), [bx, , bz] = at(e.to), x1 = px(ax), y1 = pz(az), x2 = px(bx), y2 = pz(bz), mx = (x1 + x2) / 2 + (y2 - y1) * 0.15, my = (y1 + y2) / 2 - (x2 - x1) * 0.15;
    g.push(`<path d="M${x1.toFixed(1)},${y1.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}" fill="none" stroke="${RATE[e.rating]}" stroke-width="2.4" stroke-dasharray="7 4" marker-end="url(#arr-${e.rating})"/>`);
    const tx = (x1 + 2 * mx + x2) / 4, ty = (y1 + 2 * my + y2) / 4;
    g.push(`<text x="${tx.toFixed(1)}" y="${(ty - 3).toFixed(1)}" font-size="10.5" text-anchor="middle" fill="${RATE[e.rating]}" stroke="#fbfaf7" stroke-width="3" paint-order="stroke" font-weight="700">${e.move} ${e.size}${UNIT[e.move] || (e.move === 'incline' ? '' : ' m')}</text>`);
  }
  for (const n of C.nodes) {
    const [x, , z] = n.at, r = n.kind === 'sync' || n.kind === 'tower' ? 7 : 4.5, col = n.kind === 'tower' ? '#c0392b' : n.kind === 'sync' ? '#d4a52a' : n.kind === 'start' ? '#2f5fb3' : '#555';
    g.push(`<circle cx="${px(x).toFixed(1)}" cy="${pz(z).toFixed(1)}" r="${r}" fill="${col}" stroke="#fff" stroke-width="1.5"/>`);
    g.push(`<text x="${(px(x) + 9).toFixed(1)}" y="${(pz(z) + 13).toFixed(1)}" font-size="10" fill="#222" stroke="#fbfaf7" stroke-width="3" paint-order="stroke">${n.name.replace(', sync point', ' ◆')}</text>`);
  }
  return g.join('\n');
}
const cols = Math.min(2, seeds.length), rows = Math.ceil(seeds.length / cols), legendH = 46;
const parts = seeds.map((s, i) => panel(s, (i % cols) * (W + 10), Math.floor(i / cols) * (HH + 10)));
const LW = cols * W + (cols - 1) * 10, LY = rows * (HH + 10);
const legend = [`<text x="10" y="${LY + 18}" font-size="12" fill="#333">Route tree: green, a skill of hers does it; amber, her motion data has it but no tracker does it alone yet; red, beyond her motions.  ◆ sync point.  Box labels: height of the top (m).  North up.</text>`,
  `<text x="10" y="${LY + 36}" font-size="12" fill="#333">Complete: stand on the bell tower (any route), then the leap of faith into the hay cart (yellow).</text>`];
const defs = Object.entries(RATE).map(([k, c]) => `<marker id="arr-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`).join('');
writeFileSync(out, `<svg xmlns="http://www.w3.org/2000/svg" width="${LW}" height="${LY + legendH}" font-family="Helvetica, Arial, sans-serif"><defs>${defs}</defs><rect width="100%" height="100%" fill="#fff"/>\n${parts.join('\n')}\n${legend.join('\n')}</svg>`);
console.log(`${out}: layouts ${seeds.join(', ')}`);
