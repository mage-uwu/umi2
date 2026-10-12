/* umi-1M-chat worker: loads the WebAssembly build of umi_chat.c and the trained
 * checkpoint, then answers one message at a time. Everything runs in the browser.
 *
 * Downloads are listed in manifest.json (written by pack_web.py). Each file is gzip,
 * base64-encoded into .txt parts because artifacts serve text, not binary. The model's
 * hashed n-gram table is fp16 and is widened back to float32 here, so the C code loads
 * an ordinary checkpoint. */
importScripts('umi.js');
let M = null, say = null, reset = null;

async function fetchText(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not download ${url} (HTTP ${r.status}).`);
  const reader = r.body.getReader(), chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); onBytes(value.length);
  }
  return new TextDecoder().decode(await new Blob(chunks).arrayBuffer());
}
function b64bytes(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
async function loadFile(parts, onBytes) {
  let b64 = '';
  for (const p of parts) b64 += await fetchText(p, onBytes); // each part is a multiple of 4 characters
  return gunzip(b64bytes(b64));
}
// fp16 -> fp32 through a 65,536-entry table
function halfTable() {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
}
function widenModel(packed, m) {
  const out = new Uint8Array(m.bytes);
  out.set(packed.subarray(0, m.prefix), 0);
  const half = new Uint16Array(packed.buffer.slice(packed.byteOffset + m.prefix, packed.byteOffset + m.prefix + 2 * m.count));
  const full = new Float32Array(out.buffer, m.prefix, m.count), lut = halfTable(); // prefix is a multiple of 4
  for (let i = 0; i < m.count; i++) full[i] = lut[half[i]];
  out.set(packed.subarray(m.prefix + 2 * m.count), m.prefix + 4 * m.count);
  return out;
}

async function load(beta, gamma) {
  let got = 0;
  const man = await (await fetch('manifest.json')).json();
  postMessage({ type: 'size', bytes: man.download_bytes });
  const tick = (n) => { got += n; postMessage({ type: 'progress', bytes: got }); };
  M = await createUmi({ locateFile: (p) => p });
  const model = widenModel(await loadFile(man.model.parts, tick), man.model);
  const answers = await loadFile(man.data.answers, tick), train = await loadFile(man.data.train, tick);
  M.FS.mkdir('/m'); M.FS.mkdir('/d');
  M.FS.writeFile('/m/umi.bin', model);
  M.FS.writeFile('/d/answers.tsv', answers);
  M.FS.writeFile('/d/train.tsv', train);
  postMessage({ type: 'stage', text: 'Building the n-gram cache' });
  const t0 = performance.now();
  const n = M.ccall('umi_load', 'number', ['string', 'string', 'number', 'number'], ['/m/umi.bin', '/d', beta, gamma]);
  M.FS.unlink('/d/train.tsv'); M.FS.unlink('/m/umi.bin');
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
