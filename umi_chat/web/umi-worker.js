/* umi-1M-chat worker: loads the WebAssembly build of umi_chat.c and the trained
 * checkpoint, then answers one message at a time. Everything runs in the browser.
 *
 * Two models: umi-1M-chat picks the reply, umi-smart-internet picks an optional widget.
 * Downloads are listed in NAME.manifest.json (written by pack_web.py). Each file is gzip,
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

// Fetch one packaged model (NAME.manifest.json) into the virtual filesystem under /NAME.
async function fetchModel(name, tick) {
  const man = await (await fetch(name + '.manifest.json')).json();
  const model = widenModel(await loadFile(man.model.parts, tick), man.model);
  const answers = await loadFile(man.data.answers, tick), train = await loadFile(man.data.train, tick);
  M.FS.mkdir('/' + name);
  M.FS.writeFile(`/${name}/model.bin`, model);
  M.FS.writeFile(`/${name}/answers.tsv`, answers);
  M.FS.writeFile(`/${name}/train.tsv`, train);
  return man;
}
function dropFiles(name) { for (const f of ['model.bin', 'train.tsv']) M.FS.unlink(`/${name}/${f}`); }

async function load(o) {
  let got = 0;
  const sizes = await Promise.all(['umi-1M-chat', 'umi-smart-internet'].map(async (n) => (await (await fetch(n + '.manifest.json')).json()).download_bytes));
  postMessage({ type: 'size', bytes: sizes[0] + sizes[1], chat: sizes[0] });
  const tick = (n) => { got += n; postMessage({ type: 'progress', bytes: got }); };
  M = await createUmi({ locateFile: (p) => p });
  await fetchModel('umi-1M-chat', tick);
  postMessage({ type: 'stage', text: 'Building the chat cache' });
  let t0 = performance.now();
  const n = M.ccall('umi_load', 'number', ['string', 'string', 'number', 'number'], ['/umi-1M-chat/model.bin', '/umi-1M-chat', o.beta, o.gamma]);
  dropFiles('umi-1M-chat');
  say = M.cwrap('umi_say', 'string', ['string', 'number']);
  reset = M.cwrap('umi_reset', null, []);
  postMessage({ type: 'ready', answers: n, ms: Math.round(performance.now() - t0) });
  // The widget model loads second, so chatting can start while it downloads.
  await fetchModel('umi-smart-internet', tick);
  postMessage({ type: 'stage', text: 'Building the widget cache' });
  t0 = performance.now();
  const w = M.ccall('umi_load_web', 'number', ['string', 'string', 'number', 'number', 'number'], ['/umi-smart-internet/model.bin', '/umi-smart-internet', o.gate, o.wrapGate, o.webGamma]);
  dropFiles('umi-smart-internet');
  postMessage({ type: 'widgets-ready', widgets: w, ms: Math.round(performance.now() - t0) });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load(m);
    else if (m.type === 'say') {
      const t0 = performance.now();
      const r = JSON.parse(say(m.text, m.threshold));
      postMessage({ type: 'reply', id: m.id, ...r, ms: Math.round(performance.now() - t0) });
    } else if (m.type === 'reset') { reset(); postMessage({ type: 'reset' }); }
  } catch (err) {
    postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
