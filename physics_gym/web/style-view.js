// style-view.js: the Anime Style Shader page's style shader (artifact RjX2v2aTtGYigd3V9Q5c1B: the ternary tree
// network with node memory, its lines and cel shading learned from the film), drawing the physics-driven
// character here. Its own engine (style/nts-gl.js, WebGL2 fragment shaders) and edge-aware upscale (style/guide.js).
//
// Every frame three.js draws her G-buffer tiles (colour as MToon draws it; normal, depth, surface UV and part) into
// the engine's render targets from a camera framed as the network trained (field of view 30 degrees at a fixed
// distance, the coverage lights) on a 720 x 800 frame at the 360 x 640 frames' scale; the network runs inside the
// crop box around her and its display pass draws her premultiplied on a transparent canvas, the room beneath it on
// the page's own canvas (syncCamera lines its camera up).
//
//   const art = await createStyleView({ THREE, vrm, canvas, base: 'style/girl/', shorts: 0x1c2056 });
//   art.frame({ target, yaw, pitch, zoom, about });   // as shader-view.js: target a THREE.Vector3, degrees; false while busy
//   art.syncCamera(camera, w, h)                     // the page's camera to match the last drawing
import { createGlShader } from './style/nts-gl.js';
import { createGuide } from './style/guide.js';

const TRAIN_H = 640, NW = 720, NH = 800, FOV = 30, GREEN = 0x00b140;
const CROP_PAD = 32;   // past her silhouette: the drawing inside matches computing the whole frame

export async function createStyleView({ THREE, vrm, canvas, base = 'style/girl/', shorts: shortsColor = 0x1c2056 }) {
  const VIEW_FOV = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * NH / TRAIN_H));
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, premultipliedAlpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const gl = renderer.getContext();
  const gls = await createGlShader({ THREE, renderer, NW, NH, base, precision: 'half' });

  // ---------- scene: the model alone, lit as in the coverage renders ----------
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(1, 3, 2);
  scene.add(sun);
  const camera = new THREE.PerspectiveCamera(VIEW_FOV, NW / NH, 0.01, 100);
  vrm.scene.updateMatrixWorld(true);
  const bones = [];
  vrm.scene.traverse(o => { if (o.isBone) bones.push(o); });
  // Framing: the training distance, from the model's height (the shader page's radius).
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const radius = box.getSize(new THREE.Vector3()).y * 0.58 / Math.tan(THREE.MathUtils.degToRad(FOV) / 2);
  const targetY = box.getCenter(new THREE.Vector3()).y;

  // Safety shorts under the skirt in the colour its training renders had (render-orbit.mjs --shorts-color): the body
  // drawn again over the hips and upper thighs, colour pass only. Heights from the hips, so they follow her.
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
      const mat = new THREE.MeshBasicMaterial({ color: shortsColor, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
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

  // ---------- G-buffer: five tiles in the coverage renders' 8-bit encodings (the shader page's) ----------
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
  const normalMat = new THREE.ShaderMaterial({ vertexShader: gbufVert, side: THREE.DoubleSide,
    fragmentShader: `varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() { vec3 n = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0); gl_FragColor = vec4(n * 0.5 + 0.5, 1.0); }` });
  const depthMat = new THREE.ShaderMaterial({ vertexShader: gbufVert, side: THREE.DoubleSide, uniforms: { near: { value: radius - 0.8 }, far: { value: radius + 0.8 } },
    fragmentShader: `uniform float near, far; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() { float d = 1.0 - clamp((vDepth - near) / (far - near), 0.0, 1.0); gl_FragColor = vec4(vec3(d), 1.0); }` });
  const uvSplit = which => new THREE.ShaderMaterial({ vertexShader: gbufVert, side: THREE.DoubleSide, uniforms: { part: { value: 0 } },
    fragmentShader: `uniform float part; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      vec2 split(float x) { x = fract(x) * 255.0; return vec2(floor(x) / 255.0, fract(x)); }
      void main() {
        vec2 u = split(vUv0.x), v = split(vUv0.y);
        gl_FragColor = ${which === 'a' ? 'vec4(u.x, u.y, v.x, 1.0)' : 'vec4(v.y, part / 255.0, 1.0, 1.0)'};
      }` });
  const PASSES = [[new THREE.Color(GREEN), null], [new THREE.Color(0), normalMat], [new THREE.Color(0), depthMat], [new THREE.Color(0), uvSplit('a')], [new THREE.Color(0), uvSplit('b')]]
    .map(([bg, mat]) => [bg, mat, () => showShorts(!mat)]);
  const guide = createGuide({ THREE, renderer, vertexShader: gbufVert, depthUniforms: depthMat.uniforms });

  // Parts: the network's part ids by the model's material names.
  vrm.scene.traverse(o => {
    if (!o.isMesh) return;
    o.userData.part = Math.max(0, gls.parts.indexOf([].concat(o.material)[0].name.replace(' (Instance)', '')));
    const prev = o.onBeforeRender;
    o.onBeforeRender = (r, s, c, g, mat, ...rest) => {
      if (mat.uniforms?.part) { mat.uniforms.part.value = o.userData.part; mat.uniformsNeedUpdate = true; }
      prev?.call(o, r, s, c, g, mat, ...rest);
    };
  });

  const target = new THREE.Vector3();
  function placeCamera(t, yaw, pitch) {
    target.set(t.x, targetY + (t.y - 0.9), t.z);   // her height follows the pelvis (0.9 m standing): jumps, ledges, the floor
    const y = THREE.MathUtils.degToRad(yaw), p = THREE.MathUtils.degToRad(pitch);
    camera.position.set(target.x + radius * Math.cos(p) * Math.sin(y), target.y + radius * Math.sin(p), target.z + radius * Math.cos(p) * Math.cos(y));
    camera.lookAt(target);
    camera.updateMatrixWorld();
  }
  // Her box on screen from the bones (network pixels, top-down), padded past the silhouette (on the floor hair and
  // skirt spread: shader-view.js's margins), held over two frames, snapped to 16.
  const v3 = new THREE.Vector3();
  let lastCrop = null;
  function cropBox() {
    let x0 = 1, y0 = 1, x1 = -1, y1 = -1;
    for (const b of bones) {
      b.getWorldPosition(v3).project(camera);
      x0 = Math.min(x0, v3.x); x1 = Math.max(x1, v3.x); y0 = Math.min(y0, v3.y); y1 = Math.max(y1, v3.y);
    }
    const px = x => (x * 0.5 + 0.5) * NW, py = y => (0.5 - y * 0.5) * NH;
    let b = [px(x0) - 60 - CROP_PAD, py(y1) - 80 - CROP_PAD, px(x1) + 60 + CROP_PAD, py(y0) + 48 + CROP_PAD];
    if (!b.every(Number.isFinite)) { lastCrop = null; return [0, 0, NW, NH]; }
    const now = b;
    if (lastCrop) b = [Math.min(b[0], lastCrop[0]), Math.min(b[1], lastCrop[1]), Math.max(b[2], lastCrop[2]), Math.max(b[3], lastCrop[3])];
    lastCrop = now;
    return [Math.max(0, Math.floor(b[0] / 16) * 16), Math.max(0, Math.floor(b[1] / 16) * 16),
      Math.min(NW, Math.ceil(b[2] / 16) * 16), Math.min(NH, Math.ceil(b[3] / 16) * 16)];
  }
  // Zoom about a world point: the view [zoom, shift x, shift y] (fractions of the canvas) that keeps it put.
  let lastView = [1, 0, 0];
  function viewAbout(zoom, about) {
    let fx = 0.5, fy = 0.5;
    if (about) {
      const W = canvas.clientWidth || 1, H = canvas.clientHeight || 1, sc = Math.min(W / NW, H / NH), bw = NW * sc / W, bh = NH * sc / H;
      const p = about.clone().project(camera), k = Math.min(1, Math.max(0, (zoom - 1) / 1.5));
      fx = 0.5 + k * ((1 - bw) / 2 + (p.x * 0.5 + 0.5) * bw - 0.5); fy = 0.5 + k * ((1 - bh) / 2 + (0.5 - p.y * 0.5) * bh - 0.5);
    }
    return lastView = [zoom, fx * (1 - zoom), fy * (1 - zoom)];
  }
  // The page's camera as the drawing sees it on a canvas of css size w x h: same position and direction, the field
  // of view of the whole canvas (the frame is fitted inside it), the zoomed part picked out with a view offset.
  function syncCamera(cam, w, h) {
    const s = Math.min(w / NW, h / NH), [z, vx, vy] = lastView;
    cam.position.copy(camera.position); cam.quaternion.copy(camera.quaternion);
    cam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(VIEW_FOV / 2)) * h / (NH * s)));
    cam.aspect = w / h; cam.setViewOffset(w, h, -vx / z * w, -vy / z * h, w / z, h / z); cam.updateMatrixWorld();
  }
  // One frame in flight: a fence per frame keeps the GPU from queueing frames behind it.
  let fence = null;
  function busy() {
    if (!fence) return false;
    if (gl.clientWaitSync(fence, 0, 0) === gl.TIMEOUT_EXPIRED) return true;
    gl.deleteSync(fence); fence = null;
    return false;
  }
  const art = {
    engine: 'WebGL2 · style shader',
    scene,
    zoomInPass: true,   // zoom goes to frame(): the frame box grows, the canvas is not scaled
    syncCamera,
    ms: null,
    get busy() { return busy(); },
    frame({ target: t, yaw, pitch, zoom = 1, about = null }) {
      if (busy()) return false;
      const t0 = performance.now();
      placeCamera(t, yaw, pitch);
      const [z, vx, vy] = viewAbout(zoom, about);
      const dpr = Math.min(devicePixelRatio || 1, 2), W = Math.max(1, Math.round(canvas.clientWidth * dpr)), H = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== W || canvas.height !== H) renderer.setSize(W, H, false);
      const s = Math.min(W / NW, H / NH), bw = NW * s / W, bh = NH * s / H;
      const fb = { W, H, box: [(vx + z * (1 - bw) / 2) * W, (vy + z * (1 - bh) / 2) * H, z * bw * W, z * bh * H] };
      gls.renderTiles(scene, camera, PASSES, cropBox());
      showShorts(false);
      const g = guide.render(scene, camera, fb, shorts);
      gls.frame({ pitch: THREE.MathUtils.degToRad(pitch), view: { ...fb, top: [0, 0, 0], bottom: [0, 0, 0], ink: false, guide: g, clear: true } });
      fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0); gl.flush();
      art.ms = art.ms == null ? performance.now() - t0 : art.ms * 0.9 + (performance.now() - t0) * 0.1;
      return true;
    },
  };
  return art;
}
