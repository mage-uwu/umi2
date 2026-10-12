/* umi-1M-chat worker: loads the WebAssembly build of umi_chat.c and the trained
 * checkpoint, then answers one message at a time. Everything runs in the browser. */
importScripts('umi.js');

// The checkpoint ships as base64 text in three parts (each a multiple of 4 characters).
const MODEL_PARTS = ['model/umi-1M-chat.b64.0.txt', 'model/umi-1M-chat.b64.1.txt', 'model/umi-1M-chat.b64.2.txt'];
const DATA = ['data/answers.txt', 'data/train.txt'];
let M = null, say = null, reset = null;

async function fetchAll(urls, onBytes) {
  const out = [];
  for (const u of urls) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(`Could not download ${u} (HTTP ${r.status}).`);
    const reader = r.body.getReader(), chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); onBytes(value.length);
    }
    const n = chunks.reduce((s, c) => s + c.length, 0), buf = new Uint8Array(n);
    let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
    out.push(buf);
  }
  return out;
}

function b64bytes(text) {
  const s = atob(new TextDecoder().decode(text)), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

async function load(beta, gamma) {
  let got = 0;
  const tick = (n) => { got += n; postMessage({ type: 'progress', bytes: got }); };
  M = await createUmi({ locateFile: (p) => p });
  const parts = (await fetchAll(MODEL_PARTS, tick)).map(b64bytes);
  const [answers, train] = await fetchAll(DATA, tick);
  const total = parts.reduce((s, p) => s + p.length, 0), model = new Uint8Array(total);
  let o = 0; for (const p of parts) { model.set(p, o); o += p.length; }
  M.FS.mkdir('/m'); M.FS.mkdir('/d');
  M.FS.writeFile('/m/umi.bin', model);
  M.FS.writeFile('/d/answers.tsv', answers);
  M.FS.writeFile('/d/train.tsv', train);
  postMessage({ type: 'stage', text: 'Building the n-gram cache' });
  const t0 = performance.now();
  const n = M.ccall('umi_load', 'number', ['string', 'string', 'number', 'number'], ['/m/umi.bin', '/d', beta, gamma]);
  M.FS.unlink('/d/train.tsv');
  say = M.cwrap('umi_say', 'string', ['string', 'number']);
  reset = M.cwrap('umi_reset', null, []);
  postMessage({ type: 'ready', answers: n, ms: Math.round(performance.now() - t0) });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load(m.beta, m.gamma);
    else if (m.type === 'say') {
      const t0 = performance.now();
      const r = JSON.parse(say(m.text, m.threshold));
      postMessage({ type: 'reply', id: m.id, ...r, ms: Math.round(performance.now() - t0) });
    } else if (m.type === 'reset') { reset(); postMessage({ type: 'reset' }); }
  } catch (err) {
    postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
