// shader-view.js: the ANIME NEURAL SHADER page's learned shader (its index.html and gpu.js, coverage/live),
// packaged so this page can draw the physics-driven character with it.
//
// Every frame three.js renders the model's G-buffer (colour as MToon draws it; normal and depth; surface UV
// with a part id) from a camera framed exactly as the shader was trained (portrait 360 x 640, field of view
// 30 degrees, a fixed distance, the coverage lights), then the decoder runs on it: through gpu.js (WebGPU)
// when the browser has it, else through the WebGL engine below. The camera orbits a target that follows her.
//
//   const art = await createShaderView({ THREE, vrm, canvas, detail: 'crisp' });
//   art.frame({ target, yaw, pitch });   // target: THREE.Vector3 (world), yaw/pitch in degrees
//   art.engine                           // e.g. 'WebGPU f16 · crisp' or 'WebGL · fast'
//
// Detail: 'smooth' (sharp, fine-tuned against flicker; WebGPU), 'sharp' (WebGPU), 'fast' (half resolution;
// the WebGL engine runs this one whatever is asked).

const NW = 360, NH = 640, FOV = 30, GREEN = 0x00b140;
const CROP_PAD = 64;   // > the decoder's reach in pixels, so the crop edge never shows
const DIRS = { fast: 'fast/', sharp: 'sharp/', smooth: 'equi/', crisp: 'crisp/' };

export async function createShaderView({ THREE, vrm, canvas, detail = 'crisp', webgl = false }) {
  let gpuAdapter = null;
  if (navigator.gpu && !webgl) {
    try { gpuAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); } catch { gpuAdapter = null; }
  }
  let DETAIL = gpuAdapter ? detail : 'fast';
  const dir = () => DIRS[DETAIL];
  let GBW = NW, GBH = NH;
  const glCanvas = gpuAdapter ? document.createElement('canvas') : canvas;
  const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: !!gpuAdapter, alpha: false, preserveDrawingBuffer: !!gpuAdapter, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const gl = renderer.getContext();

  // ---------- scene: the model alone, lit as in the coverage renders ----------
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(1, 3, 2);
  scene.add(sun);
  const camera = new THREE.PerspectiveCamera(FOV, NW / NH, 0.01, 100);
  vrm.scene.updateMatrixWorld(true);
  const bones = [];
  vrm.scene.traverse(o => { if (o.isBone) bones.push(o); });
  // Framing: the training distance, from the model's height (the shader page's radius).
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const radius = box.getSize(new THREE.Vector3()).y * 0.58 / Math.tan(THREE.MathUtils.degToRad(FOV) / 2);
  const targetY = box.getCenter(new THREE.Vector3()).y;

  // Navy safety shorts under the dress, as in the coverage renders: the body drawn again in the dress's navy
  // over the hips and upper thighs (colour pass only). Built from the pose the model is in now.
  const shorts = [];
  {
    const bone = n => vrm.humanoid.getNormalizedBoneNode(n);
    const hip = new THREE.Vector3(), knee = new THREE.Vector3(), legTop = new THREE.Vector3();
    bone('hips').getWorldPosition(hip); bone('leftLowerLeg').getWorldPosition(knee); bone('leftUpperLeg').getWorldPosition(legTop);
    const bottom = legTop.y - (legTop.y - knee.y) * 0.3, top = hip.y + 0.12;
    const meshes = [];
    vrm.scene.traverse(o => { if (o.isSkinnedMesh && /Body_00_SKIN/.test([].concat(o.material)[0].name)) meshes.push(o); });
    for (const o of meshes) {
      const ids = ['hips', 'spine', 'leftUpperLeg', 'rightUpperLeg'].map(n => o.skeleton.bones.indexOf(vrm.humanoid.getRawBoneNode(n)));
      const mat = new THREE.MeshBasicMaterial({ color: 0x1c2056, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
      // Skin weights decide where the shorts are; heights are measured from the hips, so they follow her.
      mat.onBeforeCompile = sh => {
        const own = ids.map(i => `float(skinIndex.x == ${i}.0) * skinWeight.x + float(skinIndex.y == ${i}.0) * skinWeight.y + float(skinIndex.z == ${i}.0) * skinWeight.z + float(skinIndex.w == ${i}.0) * skinWeight.w`).join(' + ');
        sh.uniforms.hipY = { value: 0 }; mat.userData.shader = sh;
        sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vY;\nvarying float vW;')
          .replace('#include <project_vertex>', `#include <project_vertex>\nvY = (modelMatrix * vec4(position, 1.0)).y;\nvW = ${own};`);
        sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float hipY;\nvarying float vY;\nvarying float vW;')
          .replace('void main() {', `void main() {\n  if (vW < 0.5 || vY - hipY < ${(bottom - hip.y).toFixed(4)} || vY - hipY > ${(top - hip.y).toFixed(4)}) discard;`);
      };
      const s = new THREE.SkinnedMesh(o.geometry, mat);
      s.bind(o.skeleton, o.bindMatrix);
      s.frustumCulled = false; s.visible = false;
      o.parent.add(s);
      shorts.push(s);
    }
  }
  const hipNow = new THREE.Vector3();
  const showShorts = on => {
    vrm.humanoid.getNormalizedBoneNode('hips').getWorldPosition(hipNow);
    for (const m of shorts) { m.visible = on; const sh = m.material.userData.shader; if (sh) sh.uniforms.hipY.value = hipNow.y; }
  };

  // ---------- G-buffer materials ----------
  const gbufVert = `#include <common>
  #include <uv_pars_vertex>
  #include <skinning_pars_vertex>
  varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
  void main() {
    #include <uv_vertex>
    #include <beginnormal_vertex>
    #include <skinbase_vertex>
    #include <skinnormal_vertex>
    #include <defaultnormal_vertex>
    #include <begin_vertex>
    #include <skinning_vertex>
    #include <project_vertex>
    vNormal = normalize(transformedNormal); vUv0 = uv; vDepth = -mvPosition.z;
  }`;
  const ndMat = new THREE.ShaderMaterial({
    vertexShader: gbufVert, side: THREE.DoubleSide, uniforms: { near: { value: 0 }, far: { value: 1 } },
    fragmentShader: `uniform float near, far; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() {
        vec3 n = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0);
        gl_FragColor = vec4(n * 0.5 + 0.5, 1.0 - clamp((vDepth - near) / (far - near), 0.0, 1.0));
      }`,
  });
  const uvMat = new THREE.ShaderMaterial({
    vertexShader: gbufVert, side: THREE.DoubleSide, uniforms: { part: { value: 0 } },
    fragmentShader: `uniform float part; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() { gl_FragColor = vec4(fract(vUv0), part, 1.0); }`,
  });
  const rt = (type, samples) => new THREE.WebGLRenderTarget(NW, NH, { type, samples, depthBuffer: true, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  const gColor = gpuAdapter ? null : rt(THREE.HalfFloatType, 4), gND = gpuAdapter ? null : rt(THREE.HalfFloatType, 4), gUv = gpuAdapter ? null : rt(THREE.FloatType, 0);
  const normalMat = new THREE.ShaderMaterial({
    vertexShader: gbufVert, side: THREE.DoubleSide,
    fragmentShader: `varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() { vec3 n = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0); gl_FragColor = vec4(n * 0.5 + 0.5, 1.0); }`,
  });
  const depthMat = new THREE.ShaderMaterial({
    vertexShader: gbufVert, side: THREE.DoubleSide, uniforms: { near: { value: 0 }, far: { value: 1 } },
    fragmentShader: `uniform float near, far; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() { float d = 1.0 - clamp((vDepth - near) / (far - near), 0.0, 1.0); gl_FragColor = vec4(vec3(d), 1.0); }`,
  });
  const uvSplit = which => new THREE.ShaderMaterial({
    vertexShader: gbufVert, side: THREE.DoubleSide, uniforms: { part: { value: 0 } },
    fragmentShader: `uniform float part; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      vec2 split(float x) { x = fract(x) * 255.0; return vec2(floor(x) / 255.0, fract(x)); }
      void main() {
        vec2 u = split(vUv0.x), v = split(vUv0.y);
        gl_FragColor = ${which === 'a' ? 'vec4(u.x, u.y, v.x, 1.0)' : 'vec4(v.y, part / 255.0, 1.0, 1.0)'};
      }`,
  });
  const uvMatA = uvSplit('a'), uvMatB = uvSplit('b');
  const BLACK = new THREE.Color(0x000000), GREEN_C = new THREE.Color(GREEN);

  function assignParts(names) {
    vrm.scene.traverse(o => {
      if (!o.isMesh) return;
      const name = [].concat(o.material)[0].name.replace(' (Instance)', '');
      o.userData.part = Math.max(0, names.indexOf(name));
      const prev = o.onBeforeRender;
      o.onBeforeRender = (r, s, c, g, mat, ...rest) => {
        if (mat.uniforms?.part) { mat.uniforms.part.value = o.userData.part; mat.uniformsNeedUpdate = true; }
        prev?.call(o, r, s, c, g, mat, ...rest);
      };
    });
  }

  const target = new THREE.Vector3();
  function placeCamera(t, yaw, pitch) {
    target.set(t.x, targetY, t.z);
    const y = THREE.MathUtils.degToRad(yaw), p = THREE.MathUtils.degToRad(pitch);
    camera.position.set(target.x + radius * Math.cos(p) * Math.sin(y), target.y + radius * Math.sin(p), target.z + radius * Math.cos(p) * Math.cos(y));
    camera.lookAt(target);
  }
  function renderGBuffer() {
    scene.background = new THREE.Color(GREEN);
    showShorts(true);
    renderer.setRenderTarget(gColor); renderer.render(scene, camera);
    showShorts(false);
    scene.background = null;
    renderer.setClearColor(0x000000, 0);
    ndMat.uniforms.near.value = radius - 0.8; ndMat.uniforms.far.value = radius + 0.8;
    scene.overrideMaterial = ndMat; renderer.setRenderTarget(gND); renderer.render(scene, camera);
    scene.overrideMaterial = uvMat; renderer.setRenderTarget(gUv); renderer.render(scene, camera);
    scene.overrideMaterial = null;
    renderer.setRenderTarget(null);
  }
  function renderGBufferTiles() {
    depthMat.uniforms.near.value = radius - 0.8; depthMat.uniforms.far.value = radius + 0.8;
    renderer.setScissorTest(true);
    [[GREEN_C, null], [BLACK, normalMat], [BLACK, depthMat], [BLACK, uvMatA], [BLACK, uvMatB]].forEach(([bg, mat], k) => {
      showShorts(!mat);
      renderer.setViewport(k * GBW, 0, GBW, GBH); renderer.setScissor(k * GBW, 0, GBW, GBH);
      scene.background = bg; scene.overrideMaterial = mat;
      renderer.render(scene, camera);
    });
    scene.overrideMaterial = null;
    showShorts(false);
    renderer.setScissorTest(false);
  }

  // The character's box on screen, from its bones, in network pixels.
  const v3 = new THREE.Vector3();
  let lastCrop = null;
  function cropBox(topDown = false) {
    let x0 = 1, y0 = 1, x1 = -1, y1 = -1;
    for (const b of bones) {
      b.getWorldPosition(v3).project(camera);
      x0 = Math.min(x0, v3.x); x1 = Math.max(x1, v3.x); y0 = Math.min(y0, v3.y); y1 = Math.max(y1, v3.y);
    }
    const px = x => (x * 0.5 + 0.5) * NW, py = y => (topDown ? 0.5 - y * 0.5 : y * 0.5 + 0.5) * NH;
    if (topDown) [y0, y1] = [y1, y0];
    const m = 0.12 * NH + CROP_PAD;
    let b = [px(x0) - m, py(y0) - m, px(x1) + m, py(y1) + m];
    const now = b;
    if (lastCrop) b = [Math.min(b[0], lastCrop[0]), Math.min(b[1], lastCrop[1]), Math.max(b[2], lastCrop[2]), Math.max(b[3], lastCrop[3])];
    lastCrop = now;
    const snap = 16;
    return [Math.max(0, Math.floor(b[0] / snap) * snap), Math.max(0, Math.floor(b[1] / snap) * snap),
      Math.min(NW, Math.ceil(b[2] / snap) * snap), Math.min(NH, Math.ceil(b[3] / snap) * snap)];
  }
  const css = n => new THREE.Color().setStyle(getComputedStyle(document.documentElement).getPropertyValue(n).trim(), THREE.LinearSRGBColorSpace);

  // ---------- WebGPU path (gpu.js) ----------
  let gpu = null, parts = [];
  if (gpuAdapter) {
    const net = await (await fetch(`${dir()}shader.json`)).json();
    if (net.sharp) [GBW, GBH] = net.full;
    parts = net.parts;
    const feats = await Promise.all([0, 1, 2].map(async k => createImageBitmap(await (await fetch(`${dir()}feat${k}.png`)).blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })));
    const { createGpuShader } = await import('./gpu.js');
    gpu = await createGpuShader({ net, feats, canvas, NW, NH });
    if (!gpu) throw new Error('WebGPU device unavailable');
    renderer.setPixelRatio(1);
    renderer.setSize(GBW * 5, GBH, false);
    assignParts(parts);
    return {
      engine: `WebGPU ${gpu.f16 ? 'f16' : 'f32'} · ${DETAIL}`,
      scene,
      // While the GPU still has frames in flight (gpu.js caps them at two), skip the draw: the physics keeps
      // its own clock, and the next free frame shows her where she is by then.
      get busy() { return gpu.busy; },
      frame({ target: t, yaw, pitch }) {
        if (gpu.busy) return;
        placeCamera(t, yaw, pitch);
        renderGBufferTiles();
        const crop = cropBox(true);
        const W = Math.round(canvas.clientWidth * Math.min(devicePixelRatio || 1, 2)), H = Math.round(canvas.clientHeight * Math.min(devicePixelRatio || 1, 2));
        if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
        const top = css('--stage-top'), bottom = css('--stage-bottom');
        gpu.frame({ source: glCanvas, crop, pitch: THREE.MathUtils.degToRad(pitch), mode: 0, top: [top.r, top.g, top.b], bottom: [bottom.r, bottom.g, bottom.b] });
      },
    };
  }

  // ---------- WebGL engine ----------
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('This browser cannot render to float textures (EXT_color_buffer_float), which the shader needs.');
  const MAXM = Math.min(8, gl.getParameter(gl.MAX_DRAW_BUFFERS), gl.getParameter(gl.MAX_COLOR_ATTACHMENTS));
  const vao = gl.createVertexArray();
  const VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
  function program(fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
      const name = gl.getActiveUniform(p, i).name.replace('[0]', '');
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u };
  }
  const outs = m => Array.from({ length: m }, (_, k) => `layout(location = ${k}) out vec4 o${k};`).join('\n');
  function arrayTexture(w, h, layers) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA16F, w, h, layers);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, k, v);
    return { tex: t, w, h, layers, fbos: new Map() };
  }
  function fbo(t, first, m) {
    const key = `${first}:${m}`;
    if (!t.fbos.has(key)) {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      for (let k = 0; k < m; k++) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, t.tex, 0, first + k);
      gl.drawBuffers(Array.from({ length: m }, (_, k) => gl.COLOR_ATTACHMENT0 + k));
      t.fbos.set(key, f);
    }
    return t.fbos.get(key);
  }
  const bindTex = (unit, tgt, tex) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(tgt, tex); };
  const glTex = t => renderer.properties.get(t).__webglTexture;
  async function loadImage(url) {
    const img = new Image(); img.src = url; await img.decode();
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, img);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  }

  const net = await (await fetch(`${dir()}shader.json`)).json();
  parts = net.parts;
  const featTex = await Promise.all([0, 1, 2].map(k => loadImage(`${dir()}feat${k}.png`)));
  const div = { x: 1, d0: 1, d1: 2, d2: 4, d3: 8, u0: 4, u1: 2, u2: 1, rgba: 1 };
  const chans = { ...net.channels, rgba: 4 };
  for (const l of net.layers) chans[l.out] = l.cout;
  const chansOf = n => chans[n.replace(/_pool$/, '')] ?? chans[n];
  const tensors = { x: arrayTexture(NW, NH, 5) };
  for (const l of net.layers) { const base = l.out.replace(/_mid$/, ''); tensors[l.out] = arrayTexture(NW / div[base], NH / div[base], Math.ceil(l.cout / 4)); }
  for (const [pool, src] of Object.entries(net.pools)) { const s = tensors[src]; tensors[pool] = arrayTexture(s.w / 2, s.h / 2, s.layers); }
  let rows = 0, width = 0;
  for (const l of net.layers) {
    const ins = l.inputs.map(n => ({ n, c: chansOf(n) }));
    l.L = ins.reduce((s, i) => s + Math.ceil(i.c / 4), 0);
    l.T = l.k * l.k; l.groups = Math.ceil(l.cout / 4); l.row = rows; l.ins = ins;
    l.la = Math.ceil(ins[0].c / 4);
    rows += l.groups; width = Math.max(width, l.L * l.T * 4);
  }
  const data = new Uint16Array(width * rows * 4);
  for (const l of net.layers) {
    const raw = new Uint16Array(Uint8Array.from(atob(l.w), c => c.charCodeAt(0)).buffer);
    let cOff = 0, lOff = 0;
    for (const inp of l.ins) {
      for (let c = 0; c < inp.c; c++) {
        const layer = lOff + (c >> 2), comp = c & 3, tc = cOff + c;
        for (let o = 0; o < l.cout; o++) for (let t = 0; t < l.T; t++) {
          const x = (layer * l.T + t) * 4 + (o & 3), y = l.row + (o >> 2);
          data[(y * width + x) * 4 + comp] = raw[(o * l.cin + tc) * l.T + t];
        }
      }
      cOff += inp.c; lOff += Math.ceil(inp.c / 4);
    }
    l.bias = Float32Array.from({ length: l.groups * 4 }, (_, i) => l.b[i] ?? 0);
    l.scale = Float32Array.from({ length: l.groups * 4 }, (_, i) => l.affine ? (l.affine[0][i] ?? 1) : 1);
    l.shift = Float32Array.from({ length: l.groups * 4 }, (_, i) => l.affine ? (l.affine[1][i] ?? 0) : 0);
    delete l.w;
  }
  const weightTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, weightTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, rows, 0, gl.RGBA, gl.HALF_FLOAT, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const layers = net.layers, progs = {};

  const convProg = m => progs[`conv${m}`] ??= program(`#version 300 es
precision highp float; precision highp sampler2DArray; precision highp sampler2D;
uniform sampler2DArray A, B; uniform sampler2D W;
uniform int la, L, T, K, row0, act; uniform vec2 outSize;
uniform vec4 bias[${m}], scale[${m}], shift[${m}];
${outs(m)}
float gelu(float x) { return 0.5 * x * (1.0 + tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0))); }
void main() {
  vec4 acc[${m}];
  for (int k = 0; k < ${m}; k++) acc[k] = bias[k];
  int r = K / 2;
  for (int t = 0; t < T; t++) {
    vec2 q = gl_FragCoord.xy + vec2(float(t % K - r), -float(t / K - r));
    if (q.x < 0.0 || q.y < 0.0 || q.x > outSize.x || q.y > outSize.y) continue;
    vec2 uv = q / outSize;
    for (int l = 0; l < L; l++) {
      vec4 v = l < la ? texture(A, vec3(uv, float(l))) : texture(B, vec3(uv, float(l - la)));
      int x = (l * T + t) * 4;
      for (int k = 0; k < ${m}; k++) {
        int y = row0 + k;
        acc[k] += vec4(dot(texelFetch(W, ivec2(x, y), 0), v), dot(texelFetch(W, ivec2(x + 1, y), 0), v),
                       dot(texelFetch(W, ivec2(x + 2, y), 0), v), dot(texelFetch(W, ivec2(x + 3, y), 0), v));
      }
    }
  }
  for (int k = 0; k < ${m}; k++) {
    vec4 a = acc[k];
    if (act == 1) a = vec4(gelu(a.x), gelu(a.y), gelu(a.z), gelu(a.w));
    if (act == 2) a = 1.0 / (1.0 + exp(-a));
    acc[k] = a * scale[k] + shift[k];
  }
  ${Array.from({ length: m }, (_, k) => `o${k} = acc[${k}];`).join(' ')}
}`);
  const poolProg = m => progs[`pool${m}`] ??= program(`#version 300 es
precision highp float; precision highp sampler2DArray;
uniform sampler2DArray S; uniform int first;
${outs(m)}
vec4 avg(int l) {
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  return 0.25 * (texelFetch(S, ivec3(p, l), 0) + texelFetch(S, ivec3(p + ivec2(1, 0), l), 0)
    + texelFetch(S, ivec3(p + ivec2(0, 1), l), 0) + texelFetch(S, ivec3(p + ivec2(1, 1), l), 0));
}
void main() { ${Array.from({ length: m }, (_, k) => `o${k} = avg(first + ${k});`).join(' ')} }`);
  const packProg = m => progs[`pack${m}`] ??= program(`#version 300 es
precision highp float; precision highp sampler2D;
uniform sampler2D color, nd, uvT, f0, f1, f2;
uniform int first; uniform float sinP, cosP; uniform vec4 lo0, lo1, hi0, hi1; uniform vec2 tiles; uniform float tile;
${outs(m)}
vec3 toSrgb(vec3 c) { c = max(c, 0.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec4 uvp = texelFetch(uvT, px, 0);
  float mask = uvp.a > 0.5 ? 1.0 : 0.0;
  vec3 c = toSrgb(texelFetch(color, px, 0).rgb);
  vec4 ndv = texelFetch(nd, px, 0);
  vec3 n = ndv.rgb * 2.0 - 1.0;
  float part = floor(uvp.b + 0.5);
  vec2 cell = vec2(mod(part, tiles.x), floor(part / tiles.x));
  vec2 a = (cell + clamp(uvp.rg, 0.5 / tile, 1.0 - 0.5 / tile)) / tiles;
  vec3 s0 = texture(f0, a).rgb, s1 = texture(f1, a).rgb, s2 = texture(f2, a).rgb;
  vec4 L[5];
  L[0] = vec4(c, n.x);
  L[1] = vec4(n.y, n.z, ndv.a, mask);
  L[2] = (lo0 + vec4(s0, s1.r) * (hi0 - lo0)) * mask;
  L[3] = (lo1 + vec4(s1.gb, s2.rg) * (hi1 - lo1)) * mask;
  L[4] = vec4(sinP, cosP, 0.0, 0.0);
  ${Array.from({ length: m }, (_, k) => `o${k} = L[first + ${k}];`).join(' ')}
}`);
  const showProg = () => progs.show ??= program(`#version 300 es
precision highp float; precision highp sampler2DArray; precision highp sampler2D;
uniform sampler2DArray R; uniform vec3 top, bottom; uniform vec4 box, crop; uniform vec2 view;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / view;
  vec3 bg = mix(bottom, top, uv.y);
  vec2 p = (uv - box.xy) / box.zw;
  vec2 q = p * vec2(${NW}.0, ${NH}.0);
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { o = vec4(bg, 1.0); return; }
  if (q.x < crop.x || q.y < crop.y || q.x > crop.z || q.y > crop.w) { o = vec4(bg, 1.0); return; }
  vec4 d = texture(R, vec3(p, 0.0));
  o = vec4(mix(bg, d.rgb, d.a), 1.0);
}`);
  function draw(out, first, m, crop) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo(out, first, m));
    gl.viewport(0, 0, out.w, out.h);
    const s = out.w / NW;
    gl.scissor(Math.floor(crop[0] * s), Math.floor(crop[1] * s), Math.ceil((crop[2] - crop[0]) * s) + 1, Math.ceil((crop[3] - crop[1]) * s) + 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  function runNet(crop, pitch) {
    gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE); gl.disable(gl.STENCIL_TEST);
    gl.colorMask(true, true, true, true);
    gl.enable(gl.SCISSOR_TEST);
    const pr = THREE.MathUtils.degToRad(pitch);
    for (let first = 0; first < 5; first += MAXM) {
      const m = Math.min(MAXM, 5 - first), P = packProg(m);
      gl.useProgram(P.p);
      [[gColor.texture, 'color'], [gND.texture, 'nd'], [gUv.texture, 'uvT']].forEach(([t, n], i) => { bindTex(i, gl.TEXTURE_2D, glTex(t)); gl.uniform1i(P.u[n], i); });
      featTex.forEach((t, i) => { bindTex(3 + i, gl.TEXTURE_2D, t); gl.uniform1i(P.u[`f${i}`], 3 + i); });
      gl.uniform1i(P.u.first, first);
      gl.uniform1f(P.u.sinP, Math.sin(pr)); gl.uniform1f(P.u.cosP, Math.cos(pr));
      gl.uniform4fv(P.u.lo0, net.feat_lo.slice(0, 4)); gl.uniform4fv(P.u.lo1, net.feat_lo.slice(4, 8));
      gl.uniform4fv(P.u.hi0, net.feat_hi.slice(0, 4)); gl.uniform4fv(P.u.hi1, net.feat_hi.slice(4, 8));
      gl.uniform2f(P.u.tiles, net.cols, net.rows); gl.uniform1f(P.u.tile, net.tile);
      draw(tensors.x, first, m, crop);
    }
    for (const l of layers) {
      const out = tensors[l.out], A = tensors[l.inputs[0]], B = tensors[l.inputs[1] ?? l.inputs[0]];
      for (let g = 0; g < l.groups; g += MAXM) {
        const m = Math.min(MAXM, l.groups - g), P = convProg(m);
        gl.useProgram(P.p);
        bindTex(0, gl.TEXTURE_2D_ARRAY, A.tex); gl.uniform1i(P.u.A, 0);
        bindTex(1, gl.TEXTURE_2D_ARRAY, B.tex); gl.uniform1i(P.u.B, 1);
        bindTex(2, gl.TEXTURE_2D, weightTex); gl.uniform1i(P.u.W, 2);
        gl.uniform1i(P.u.la, l.la); gl.uniform1i(P.u.L, l.L); gl.uniform1i(P.u.T, l.T); gl.uniform1i(P.u.K, l.k);
        gl.uniform1i(P.u.row0, l.row + g); gl.uniform1i(P.u.act, l.act === 'gelu' ? 1 : 2);
        gl.uniform2f(P.u.outSize, out.w, out.h);
        gl.uniform4fv(P.u.bias, l.bias.subarray(g * 4, (g + m) * 4));
        gl.uniform4fv(P.u.scale, l.scale.subarray(g * 4, (g + m) * 4));
        gl.uniform4fv(P.u.shift, l.shift.subarray(g * 4, (g + m) * 4));
        draw(out, g, m, crop);
      }
      const pooled = Object.entries(net.pools).find(([, src]) => src === l.out);
      if (pooled) {
        const dst = tensors[pooled[0]];
        for (let first = 0; first < dst.layers; first += MAXM) {
          const m = Math.min(MAXM, dst.layers - first), P = poolProg(m);
          gl.useProgram(P.p);
          bindTex(0, gl.TEXTURE_2D_ARRAY, out.tex); gl.uniform1i(P.u.S, 0);
          gl.uniform1i(P.u.first, first);
          draw(dst, first, m, crop);
        }
      }
    }
    gl.disable(gl.SCISSOR_TEST);
  }
  function show(crop) {
    const w = canvas.clientWidth, h = canvas.clientHeight, size = renderer.getSize(new THREE.Vector2());
    if (size.x !== w || size.y !== h) renderer.setSize(w, h, false);
    const W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
    const s = Math.min(W / NW, H / NH), bw = NW * s / W, bh = NH * s / H, P = showProg();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.SCISSOR_TEST); gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.bindVertexArray(vao);
    gl.useProgram(P.p);
    bindTex(0, gl.TEXTURE_2D_ARRAY, tensors.rgba.tex); gl.uniform1i(P.u.R, 0);
    const top = css('--stage-top'), bottom = css('--stage-bottom');
    gl.uniform3f(P.u.top, top.r, top.g, top.b); gl.uniform3f(P.u.bottom, bottom.r, bottom.g, bottom.b);
    gl.uniform4f(P.u.box, (1 - bw) / 2, (1 - bh) / 2, bw, bh);
    gl.uniform4f(P.u.crop, ...crop);
    gl.uniform2f(P.u.view, W, H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  assignParts(parts);
  return {
    engine: `WebGL · ${DETAIL}`,
    scene,
    frame({ target: t, yaw, pitch }) {
      placeCamera(t, yaw, pitch);
      renderGBuffer();
      const crop = cropBox();
      renderer.resetState();
      runNet(crop, pitch);
      show(crop);
      renderer.resetState();
    },
  };
}
