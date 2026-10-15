// three.js site scene: 3DEP terrain draped with imagery (its edge dissolves into a halftone dot screen), buildings, USGS
// LiDAR vegetation, CFD streamlines with glowing time-of-flight pulses, the wind-map sheet, and a storm atmosphere driven
// by the selected hour: overcast darkness and fog from rain / thunder / chance of rain, rain slanted by the real wind,
// lightning only when thunder is 30% or more. House loads are halftone pressure panels (push warm, suction cool).
// Camera: intro() flies in from a high orbit while the airflow draws on; focus() makes eased flights that any user
// input interrupts. Imperative: per-frame work stays out of React. ?perf=1 logs timings and frame stats to the console.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { N8AOPass } from "n8ao";
import { clamp, css, gustRatio, rampLin, type Model, type Tree, bldName } from "./model";
import { boltPath, buildGrass, buildHouses, buildRain, buildSky, buildTrees, groundDetail, halftoneEdge, inRect, loadSat, overcastEnv, type Rect } from "./realism";
import { createEarth, type EarthStatus, type HouseMode } from "./earth";
import { createMap3d } from "./map3d";
import { createStreamlines } from "./streamlines";
import "./scene.css";

export type View = "site" | "house" | "storm" | "plan";
export type Layers = { flow: boolean; slice: boolean; risk: boolean; rain: boolean; lidar: boolean };
export type SceneState = { idx: number; view: View; layer: Layers; sliceH: number; houseMode: HouseMode };
export type FocusTarget = { kind: "overview" } | { kind: "house" } | { kind: "tree"; i: number /* index into M.D.trees */ };
type Opts = { canvas: HTMLCanvasElement; overlay: HTMLElement; needle: SVGGElement; onHover: (t: Tree | null, x: number, y: number) => void;
  googleKey?: string; googleMode?: "tiles" | "map3d" | ""; onEarth?: (s: EarthStatus) => void; credits?: HTMLElement; gmapHost?: HTMLElement };
type Pose = [THREE.Vector3, THREE.Vector3];   // camera position, target

const ease = (t: number) => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

export function createSiteScene(M: Model, o: Opts, init: SceneState) {
  const T0 = performance.now(), PERF = new URLSearchParams(location.search).has("perf");
  const log = (...a: unknown[]) => { if (PERF) console.log("[scene]", ...a); }, since = (t: number) => `${Math.round(performance.now() - t)} ms`;
  const { D, HRS, HALF, W, TER, terrH, secOf, wind } = M;
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let st = init, dirty = true;
  const cv = o.canvas;
  const PRMAX = Math.min(2, devicePixelRatio || 1); let pr = PRMAX;
  const R = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true, powerPreference: "high-performance" });
  R.setPixelRatio(pr); R.outputColorSpace = THREE.SRGBColorSpace; R.toneMapping = THREE.ACESFilmicToneMapping; R.toneMappingExposure = .8;
  // nothing that casts a shadow moves (sway is vertex-only), so the 2048 map is re-rendered only when the world changes
  R.shadowMap.enabled = true; R.shadowMap.type = THREE.PCFShadowMap; R.shadowMap.autoUpdate = false; R.shadowMap.needsUpdate = true;
  R.info.autoReset = false;
  const sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, 1, 3000);
  const ctl = new OrbitControls(cam, cv); ctl.enableDamping = true; ctl.maxPolarAngle = Math.PI * .47; ctl.minDistance = 18; ctl.maxDistance = 560; ctl.enableZoom = false;
  const engage = () => { ctl.enableZoom = true; };                        // no scroll hijack until the user engages the scene
  cv.addEventListener("pointerdown", engage);
  const P3 = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);   // world (east, north, up) -> three (x, up, -north)
  const TB = HALF * .95;   // orbit target bounds: anywhere over the modelled ground, edge trees included

  // overcast storm light: a halftone cloud dome, soft gradient IBL, a weak sun for form, fog matched to the horizon
  const sky = buildSky(sc), env = overcastEnv(R); sc.environment = env; sc.environmentIntensity = .4;
  const fog = new THREE.Fog(0x1b232c, 700, 2600); sc.fog = fog;
  const hemi = new THREE.HemisphereLight(0xc9d4df, 0x3a3830, .6); sc.add(hemi);
  const sunDir = new THREE.Vector3(-170, 230, 110).normalize();
  const sun = new THREE.DirectionalLight(0xf3ede4, 2); sun.position.copy(sunDir).multiplyScalar(300); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -190, right: 190, top: 190, bottom: -190, near: 10, far: 800 });
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -.0004; sun.shadow.normalBias = .04; sun.shadow.radius = 3; sc.add(sun);

  // terrain (USGS 3DEP) draped with the satellite mosaic; the border dissolves into dots
  const satTex = new THREE.TextureLoader().load(D.scene.img, () => { dirty = true; }); satTex.colorSpace = THREE.SRGBColorSpace; satTex.anisotropy = R.capabilities.getMaxAnisotropy();
  const n = TER.n, tg = new THREE.PlaneGeometry(2 * HALF, 2 * HALF, n - 1, n - 1); tg.rotateX(-Math.PI / 2);
  { const p = tg.attributes.position; for (let i = 0; i < p.count; i++) p.setY(i, terrH(p.getX(i), -p.getZ(i))); tg.computeVertexNormals(); }
  const gdet = groundDetail(); gdet.repeat.set(HALF, HALF);
  const gmat = new THREE.MeshStandardMaterial({ map: satTex, normalMap: gdet, normalScale: new THREE.Vector2(.6, .6), roughness: .97 }); halftoneEdge(gmat, HALF);
  const ground = new THREE.Mesh(tg, gmat); ground.receiveShadow = true; sc.add(ground);
  let lo = 1e9; for (const v of TER.a) lo = Math.min(lo, v);

  // houses, lawn and trees need the imagery pixels (roof colours, lawn mask, driveways) and load async
  let houseWalls: { color: THREE.Color }[] = [], grass: Awaited<ReturnType<typeof buildGrass>> = null, trees: Awaited<ReturnType<typeof buildTrees>> | null = null;
  let dead = false, worldReady = false;
  const HS = D.house, satP = loadSat(D.scene.img, HALF);
  function procedural() {
    worldReady = false;
    satP.then(async sat => {
      if (dead) return; let t = performance.now();
      houseWalls = buildHouses(M, sc, sat).houseWalls; faces.forEach((f, k) => f.mat = houseWalls[k]); R.shadowMap.needsUpdate = true; refresh(); log("houses", since(t));
      t = performance.now(); grass = await buildGrass(M, sc, sat, () => !dead); if (dead) return; log("grass (time-sliced)", since(t)); dirty = true;
      t = performance.now(); const tr = await buildTrees(M, sc); if (dead) return; trees = tr; paintVeg(); R.shadowMap.needsUpdate = true; log("trees", since(t));
    }).catch(e => console.warn("site dressing failed", e)).finally(() => { worldReady = true; dirty = true; });
  }

  // Google Earth mesh (Photorealistic 3D Tiles) when a key is configured; the procedural world is the fallback
  let earth: ReturnType<typeof createEarth> | null = null;
  function earthLook(on: boolean) {
    R.toneMapping = on ? THREE.LinearToneMapping : THREE.ACESFilmicToneMapping;   // photo textures are display-ready: only darken them
    ground.visible = !on; dirty = true;
  }
  // house walls for the load view: one halftone pressure panel per wall. Google modes use the true outline (each wall
  // with its own facing); the procedural house uses its 4 rectangle walls and also tints their siding.
  type Face = { fb: number; x: number; y: number; nx: number; ny: number; len: number; cp?: (number | null)[]; mat?: { color: THREE.Color }; panel?: THREE.ShaderMaterial };
  const gy0 = terrH(0, 0), panels: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>[] = [];
  const rectFaces = (): Face[] => [[HS.heading, HS.L / 2, HS.W], [HS.heading + 180, HS.L / 2, HS.W], [HS.heading + 90, HS.W / 2, HS.L], [HS.heading - 90, HS.W / 2, HS.L]]
    .map(([fb, d, len]) => { const r = fb * Math.PI / 180, nx = Math.sin(r), ny = Math.cos(r); return { fb, x: nx * d, y: ny * d, nx, ny, len }; });
  let faces: Face[] = rectFaces();
  function outlineFaces(): Face[] {
    if (HS.walls?.length) return HS.walls.map((w: Face) => ({ ...w }));   // engine-simplified walls, with CFD Cp per wind sector
    let p: number[][] = (HS.ring as number[][] | undefined)?.slice() ?? [];
    if (p.length > 1 && Math.hypot(p[0][0] - p.at(-1)![0], p[0][1] - p.at(-1)![1]) < .05) p.pop();
    // drop near-collinear and stub vertices so a wall is a wall, not a dozen survey points
    for (let changed = true; changed && p.length > 3;) {
      changed = false;
      for (let i = 0; i < p.length && p.length > 3; i++) {
        const a = p[(i + p.length - 1) % p.length], b = p[i], c = p[(i + 1) % p.length];
        const ux = b[0] - a[0], uy = b[1] - a[1], vx = c[0] - b[0], vy = c[1] - b[1], lu = Math.hypot(ux, uy), lv = Math.hypot(vx, vy);
        if (lu < .7 || lv < .7 || Math.abs(Math.asin(clamp((ux * vy - uy * vx) / (lu * lv), -1, 1))) < .2 && ux * vx + uy * vy > 0) { p.splice(i, 1); changed = true; }
      }
    }
    if (p.length < 3) return rectFaces();
    let A = 0; p.forEach((a, i) => { const b = p[(i + 1) % p.length]; A += a[0] * b[1] - b[0] * a[1]; });
    return p.map((a, i) => {
      const b = p[(i + 1) % p.length], dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy), sg = A > 0 ? 1 : -1;
      const nx = sg * dy / len, ny = -sg * dx / len;                                   // outward normal
      return { fb: (Math.atan2(nx, ny) * 180 / Math.PI + 360) % 360, x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, nx, ny, len };
    });
  }
  // halftone pressure panel: dot size = |Cp|, rings run inward for push and outward for suction, bright hairline frame
  const pnlU = { uTime: { value: 0 }, uOn: { value: 0 } };
  function makePanels(base: number, h: number, out: number) {
    for (const m of panels) { sc.remove(m); m.geometry.dispose(); m.material.dispose(); } panels.length = 0;
    for (const f of faces) {
      const mat = new THREE.ShaderMaterial({
        uniforms: { ...pnlU, uCol: { value: new THREE.Color() }, uK: { value: 0 }, uSign: { value: 1 }, uSize: { value: new THREE.Vector2(f.len, h) } },
        transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: false,
        vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: `uniform vec3 uCol; uniform vec2 uSize; uniform float uK, uSign, uTime, uOn; varying vec2 vUv;
          void main() {
            vec2 m = vUv * uSize, g = m / .45, f = fract(g) - .5;
            vec2 c = ((floor(g) + .5) * .45 - uSize * .5) / (uSize * .5);
            float wave = .5 + .5 * sin(length(c) * 6. + uSign * uTime * 2.6);
            float r = .5 * (.22 + .78 * uK) * (.72 + .28 * wave), d = length(f), aa = fwidth(d);
            float dotm = 1. - smoothstep(r - aa, r + aa, d);
            vec2 e2 = min(m, uSize - m); float frame = 1. - smoothstep(0., .1, min(e2.x, e2.y));
            gl_FragColor = vec4(uCol * (1. + 2. * frame + 1.2 * uK * dotm), (dotm * (.5 + .4 * uK) + frame * .9 + .06) * uOn);
            #include <colorspace_fragment>
          }`,
      });
      const m = new THREE.Mesh(new THREE.PlaneGeometry(f.len, h), mat);
      m.position.set(f.x + f.nx * out, base + h / 2, -(f.y + f.ny * out)); m.rotation.y = Math.atan2(f.nx, -f.ny); m.renderOrder = 3; m.visible = false;
      sc.add(m); panels.push(m); f.panel = mat;
    }
  }
  const wantG3d = () => !!(o.googleKey && o.googleMode === "map3d" && o.gmapHost && !g3dFailed);
  let g3dFailed = false;
  if (o.googleKey) {
    const h = HS.eave_m; let gb = 1e9; for (const q of (HS.ring ?? [[0, 0]]) as number[][]) gb = Math.min(gb, terrH(q[0], q[1]));
    faces = outlineFaces(); makePanels(gb - 1.25, h + 1.5, .25);
    earthLook(true);
    if (wantG3d()) startMap3d(); else earth = createEarth(M, sc, cam, R, o.googleKey, satP, es => {
      o.onEarth?.(es);
      if (es.phase === "error" && earth) { earth.dispose(); earth = null; if (o.gmapHost) startMap3d(); else noGoogle(); }
    });
    earth?.setMode(st.houseMode);
  } else { makePanels(gy0, HS.eave_m, .12); procedural(); }
  function noGoogle() { faces = rectFaces(); makePanels(gy0, HS.eave_m, .12); wallTags.forEach(t => t.el.hidden = true); earthLook(false); procedural(); refresh(); }

  // Google's 3D map component underneath (free Demo Key); our canvas draws the simulation on top with Google's camera
  let g3d: Awaited<ReturnType<typeof createMap3d>> | null = null, pendingIntro = false, occ: THREE.Group | null = null;
  function startMap3d() {
    const host = o.gmapHost!;
    createMap3d(M, host, o.googleKey!, why => { o.onEarth?.({ phase: "error", error: why }); stopMap3d(); }).then(m => {
      if (dead) return m.dispose();
      g3d = m; host.hidden = false; ctl.enabled = false; cv.style.pointerEvents = "none";
      sky.mesh.visible = false; sc.fog = null; R.setClearColor(0, 0); ground.visible = false;
      host.addEventListener("pointermove", onMove); host.addEventListener("pointerleave", onLeave);
      sc.add(occ = occluders()); o.onEarth?.({ phase: "map3d" });
      // the intro runs here if it was waiting for this map (or for a Google tiles mesh that failed and fell back to it)
      if (pendingIntro || introWait) { pendingIntro = false; endWait(); introCalled = false; intro(); } else flyTo(lastPose ?? cams.site, 0);
    }, e => { o.onEarth?.({ phase: "error", error: String(e.message || e) }); stopMap3d(); });
  }
  function stopMap3d() {
    if (g3dFailed || dead) return;   // the auth-failure callback and the rejected promise can both land here
    g3d?.dispose(); g3d = null; g3dFailed = true; if (occ) { sc.remove(occ); occ.traverse(m => (m as THREE.Mesh).geometry?.dispose()); occ = null; }
    if (o.gmapHost) { o.gmapHost.hidden = true; o.gmapHost.style.filter = ""; o.gmapHost.removeEventListener("pointermove", onMove); o.gmapHost.removeEventListener("pointerleave", onLeave); }
    ctl.enabled = true; cv.style.pointerEvents = ""; sc.fog = fog; sky.mesh.visible = true; noGoogle();
    if (pendingIntro) { pendingIntro = false; introCalled = false; intro(); }
  }
  // depth-only stand-ins for buildings and crowns, so wind lines pass behind Google's houses and trees
  function occluders() {
    const g = new THREE.Group(), mat = new THREE.MeshBasicMaterial({ colorWrite: false }), q = new THREE.Quaternion(), y = new THREE.Vector3(0, 1, 0);
    const box = (b: { cx: number; cy: number; L: number; W: number; heading: number }, h: number) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(b.L, h + 1, b.W), mat); m.position.set(b.cx, terrH(b.cx, b.cy) + h / 2 - .5, -b.cy); m.rotation.y = (90 - b.heading) * Math.PI / 180; m.renderOrder = -1; g.add(m);
    };
    box(HS, (HS.eave_m + HS.ridge_m) / 2); for (const b of D.buildings) box(b, b.h || 7);
    const tr = D.trees.filter((t: Tree) => t.h >= 4), im = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), mat, tr.length), m4 = new THREE.Matrix4();
    tr.forEach((t: Tree, k: number) => { const r = t.cw / 2 * .8; m4.compose(new THREE.Vector3(t.x, terrH(t.x, t.y) + t.h - r, -t.y), q.setFromAxisAngle(y, 0), new THREE.Vector3(r, Math.min(r, t.h / 2), r)); im.setMatrixAt(k, m4); });
    im.renderOrder = -1; g.add(im); return g;
  }

  // USGS LiDAR vegetation returns, coloured from the imagery; trees cut since the survey are already removed.
  // Decoded the first time a layer needs them (not at load): 100k+ points and their tree lookup cost a visible stall.
  let veg: THREE.Points | null = null, vegBase: Float32Array | null = null, vegTree: Int32Array | null = null;
  function ensureVeg() {
    if (veg || !D.scene.veg?.n) return; const t = performance.now();
    const V = D.scene.veg, raw = Uint8Array.from(atob(V.xyz), c => c.charCodeAt(0)), xyz = new Int16Array(raw.buffer), rgb = Uint8Array.from(atob(V.rgb), c => c.charCodeAt(0));
    const N = V.n, pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const x = xyz[3 * i] / 10, y = xyz[3 * i + 1] / 10, z = xyz[3 * i + 2] / 10; pos[3 * i] = x; pos[3 * i + 1] = z; pos[3 * i + 2] = -y;
      const sh = .5 + .7 * clamp((z - terrH(x, y)) / 22, 0, 1);              // simple self-shading: darker deep in the crown
      col[3 * i] = Math.pow(rgb[3 * i] / 255, 2.2) * sh * 1.3; col[3 * i + 1] = Math.pow(rgb[3 * i + 1] / 255, 2.2) * sh * 1.35; col[3 * i + 2] = Math.pow(rgb[3 * i + 2] / 255, 2.2) * sh;
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const dot = document.createElement("canvas"); dot.width = dot.height = 32; { const x = dot.getContext("2d")!; x.fillStyle = "#fff"; x.beginPath(); x.arc(16, 16, 14, 0, 7); x.fill(); }
    veg = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.0, vertexColors: true, map: new THREE.CanvasTexture(dot), alphaTest: .5 }));
    sc.add(veg); vegBase = col.slice(); log(`LiDAR points decoded (${N})`, since(t));
  }
  function ensureVegTree() {
    if (vegTree || !veg) return; const pos = veg.geometry.attributes.position.array as Float32Array, N = pos.length / 3, t0 = performance.now();
    vegTree = new Int32Array(N).fill(-1); const cell = 4, grid = new Map<string, number[]>();
    D.trees.forEach((t: Tree, k: number) => { const key = Math.floor(t.x / cell) + "," + Math.floor(t.y / cell); if (!grid.has(key)) grid.set(key, []); grid.get(key)!.push(k); });
    for (let i = 0; i < N; i++) {
      const x = pos[3 * i], y = -pos[3 * i + 2], gx = Math.floor(x / cell), gy = Math.floor(y / cell); let best = -1, bd = 1e9;
      for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) for (const k of grid.get((gx + a) + "," + (gy + b)) || []) {
        const t = D.trees[k], d = Math.hypot(t.x - x, t.y - y);
        if (d < t.cw / 2 + 1.5 && d < bd) { bd = d; best = k; }
      }
      vegTree[i] = best;
    }
    log("LiDAR points matched to trees", since(t0));
  }
  function paintVeg() {
    const risk = st.layer.risk && st.view !== "storm";
    trees?.paint(risk, new THREE.Color(css("--warn")), new THREE.Color(css("--bad")));
    const want = st.layer.lidar || (!!earth && risk); if (want) ensureVeg();
    if (!veg) return; veg.visible = want; const c = veg.geometry.attributes.color as THREE.BufferAttribute, arr = c.array as Float32Array;
    if (!risk) { arr.set(vegBase!); c.needsUpdate = true; return; }
    ensureVegTree();
    // THREE.Color already converts hex/CSS colours to linear (ColorManagement): no second conversion
    const lin = (h: string | number) => new THREE.Color(h), bad = lin(css("--bad")), warn = lin(css("--warn")), calm = lin("#55635c"), small = lin("#2c3431");
    for (let i = 0; i < vegTree!.length; i++) {
      const t = D.trees[vegTree![i]], p = t ? (t.p_fail || 0) : 0;
      const C = !t || t.h < 8 ? small : p < .002 ? calm : p < .02 ? calm.clone().lerp(warn, (p - .002) / .018) : warn.clone().lerp(bad, clamp((p - .02) / .1, 0, 1));
      arr[3 * i] = C.r; arr[3 * i + 1] = C.g; arr[3 * i + 2] = C.b;
    }
    c.needsUpdate = true;
  }

  // wind map: terrain-following sheet at a chosen height, coloured by the local gust from the CFD field
  const SN = 100, sg = new THREE.PlaneGeometry(280, 280, SN - 1, SN - 1); sg.rotateX(-Math.PI / 2);
  sg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(SN * SN * 3), 3));
  const sheet = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: .42, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }));
  sheet.renderOrder = 2; sc.add(sheet);
  function paintSheet() {
    sheet.visible = !!(W && st.layer.slice && st.view !== "storm"); if (!sheet.visible) return;
    const p = sg.attributes.position, c = sg.attributes.color, rgb = [0, 0, 0], sec = secOf(st.idx)!, h = st.sliceH;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = -p.getZ(i), g0 = terrH(x, y); p.setY(i, g0 + h);
      const o = wind(sec, x, y, g0 + h), K = Math.hypot(o[0], o[1], o[2]);
      if (K < .02) { c.setXYZ(i, .004, .005, .006); continue; }
      rampLin(HRS[st.idx].gust[2] * gustRatio(K, o[3], sec, h), rgb); c.setXYZ(i, rgb[0], rgb[1], rgb[2]);
    }
    p.needsUpdate = c.needsUpdate = true;
  }

  // airflow: streamlines through the CFD mean field, coloured by local gust, pulses moving at the local speed
  const stream = createStreamlines(M, sc, HS.eave_m, b => log(`streamlines traced: ${b.lines} lines, ${b.segs} segments, ${b.ms} ms of slices`));
  const rain = buildRain(sc);

  // lightning: a forked channel far out in the storm, the cloud deck lit around it, a brief light flicker
  const lm = (w: number, r: number, g: number, b: number) => { const m = new LineMaterial({ linewidth: w, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }); m.color.setRGB(r, g, b); return m; };
  const boltMat = lm(2, 3.2, 3.4, 4.2), haloMat = lm(11, .45, .5, .9);
  let boltGeo = new LineSegmentsGeometry(); const bolt = new LineSegments2(boltGeo, boltMat), halo = new LineSegments2(boltGeo, haloMat);
  bolt.frustumCulled = halo.frustumCulled = false; bolt.visible = halo.visible = false; bolt.renderOrder = halo.renderOrder = 7; sc.add(halo, bolt);
  let flash = 0, strikeT = -1, nextStrike = 0;
  // the channel comes down where a ray through the upper part of the frame meets the ground beyond the modelled world (so
  // it is in view and nothing should hide it); a camera looking steeply down has no such point and gets an off-screen
  // strike: only the flash
  const ray = new THREE.Vector3();
  function strike() {
    const g = lo / 100 - 2; let t = 0;
    for (let k = 0; k < 8 && !t; k++) {
      ray.set((Math.random() - .5) * 1.4, .45 + Math.random() * .5, .5).unproject(cam).sub(cam.position).normalize();
      const d = ray.y < -.01 ? (cam.position.y - g) / -ray.y : 0, hx = cam.position.x + ray.x * d, hz = cam.position.z + ray.z * d;
      if (d > 380 && d < 1800 && Math.max(Math.abs(hx), Math.abs(hz)) > HALF + 30) t = d;
    }
    if (!t) { const f = cam.getWorldDirection(ray), yaw = Math.atan2(f.x, f.z) + (Math.random() - .5) * 1.2; ray.set(Math.sin(yaw), 0, Math.cos(yaw)); t = 650 + Math.random() * 450; }
    const gx = cam.position.x + ray.x * t, gz = cam.position.z + ray.z * t;
    const top = new THREE.Vector3(gx + (Math.random() - .5) * 140, 300 + Math.random() * 140, gz + (Math.random() - .5) * 140), bot = new THREE.Vector3(gx, g, gz);
    boltGeo.dispose(); boltGeo = new LineSegmentsGeometry(); boltGeo.setPositions(boltPath(top, bot)); bolt.geometry = halo.geometry = boltGeo;
    sky.uni.uFlashDir.value.copy(top).sub(cam.position).normalize();
  }
  function lightning(dt: number, now: number) {
    const th = HRS[st.idx].thunder ?? 0;
    if (RM || th < 30) nextStrike = 0;
    else {
      if (!nextStrike) nextStrike = now + 1200 + Math.random() * 2500;
      // mean gap 14 s at 30% thunder down to 4 s at 90%; never closer than 2.5 s (and two pulses per strike: well under 3 flashes/s)
      if (now > nextStrike && strikeT < 0) { strike(); strikeT = 0; nextStrike = now + 1000 * Math.max(2.5, -Math.log(1 - Math.random()) * (14 - 10 * clamp((th - 30) / 60, 0, 1))); }
    }
    if (strikeT >= 0) { strikeT += dt; flash = Math.exp(-strikeT / .07) + (strikeT > .14 ? .75 * Math.exp(-(strikeT - .14) / .12) : 0); if (strikeT > .9) { strikeT = -1; flash = 0; } }
    boltMat.opacity = Math.min(1, flash * 1.2); haloMat.opacity = flash * .5; bolt.visible = halo.visible = flash > .01;
  }

  // atmosphere from the selected hour (eased so scrubbing the timeline crossfades): 0 = bright overcast, 1 = black storm
  const SKY = { top0: new THREE.Color("#2b333c"), hor0: new THREE.Color("#5f6a75"), cl0: new THREE.Color("#c6ced6"),
    top1: new THREE.Color(css("--sky-top") || "#0b0f14"), hor1: new THREE.Color(css("--sky-bot") || "#1b232c"), cl1: new THREE.Color("#7c8794") };
  const A = { dark: -1, fog: 0 }, fwd = new THREE.Vector3(); let lastFilter = "";
  function atmos(dt: number, now: number) {
    const h = HRS[st.idx], wet = clamp(h.rain[0] / .2, 0, 1), th = clamp((h.thunder ?? 0) / 80, 0, 1), pop = clamp((h.pop ?? 0) / 100, 0, 1), gw = clamp(h.gust[2] / 70, 0, 1);
    const dk = clamp(.15 + .25 * pop + .35 * wet + .3 * th + .1 * gw, 0, 1), fg = clamp(.08 + .6 * wet + .25 * th + .1 * pop, 0, 1);
    if (A.dark < 0 || RM) { A.dark = dk; A.fog = fg; } else { const k = 1 - Math.exp(-dt * 2.5); A.dark += (dk - A.dark) * k; A.fog += (fg - A.fog) * k; }
    const d = A.dark, U = sky.uni, toward = (h.dir + 180) * Math.PI / 180;
    U.uTop.value.lerpColors(SKY.top0, SKY.top1, d); U.uHor.value.lerpColors(SKY.hor0, SKY.hor1, d); U.uCloud.value.lerpColors(SKY.cl0, SKY.cl1, d);
    U.uDark.value = d; U.uFlash.value = flash; U.uTime.value = RM ? 0 : now / 1000;
    const v = .004 + h.sust * .0006; U.uDrift.value.set(-Math.sin(toward) * v, Math.cos(toward) * v);   // the deck drifts downwind
    fog.color.copy(U.uHor.value); fog.near = 700 - 520 * A.fog; fog.far = 2600 - 1650 * A.fog;
    sun.intensity = 2.3 - 1.5 * d; hemi.intensity = .65 - .2 * d + flash * 2.4; sc.environmentIntensity = .42 - .15 * d;   // the house stays readable in the worst hour
    // Google's photo textures are graded down toward the report's night palette so the airflow reads on top of them
    R.toneMappingExposure = (earth ? .8 - .3 * d : .8 - .1 * d) + flash * .18;
    if (g3d && o.gmapHost) {   // Google's own map: grade it with a CSS filter (and flash it) instead
      const f = `brightness(${(.72 - .3 * d + flash * .8).toFixed(3)}) saturate(${(.7 - .35 * d).toFixed(3)}) contrast(${(1.05 + .05 * d).toFixed(3)})`;
      if (f !== lastFilter) o.gmapHost.style.filter = lastFilter = f;
    }
    // rain: drop count follows the hourly rate; the streaks slant with the 10 m wind blowing toward its heading
    const rr = h.rain[0], cnt = st.layer.rain && !RM ? clamp(rr * 50000, rr > .002 ? 500 : 0, 14000) : 0;
    rain.count(cnt); if (!cnt) return;
    const ws = h.sust * .447 * .8, ru = rain.uni, c = ru.uCenter.value;
    ru.uVel.value.set(Math.sin(toward) * ws, -9, -Math.cos(toward) * ws); ru.uOpacity.value = .16 + .22 * clamp(rr / .3, 0, 1); rain.step(dt);
    if (g3d) { cam.getWorldDirection(fwd); c.copy(cam.position).addScaledVector(fwd, fwd.y < -.05 ? Math.min(400, -cam.position.y / fwd.y) : 150); } else c.copy(ctl.target);
    const bx = clamp(cam.position.distanceTo(c) * 1.1, 70, 320); ru.uBox.value.set(bx, 70 + bx * .15, bx); c.y += ru.uBox.value.y * .5 - 6;
  }

  // house loads: ASCE 7 low-rise wall pressure coefficients by wind angle, velocity pressure from the CFD roof-height gust
  const tags: { el: HTMLDivElement; pos: THREE.Vector3 }[] = [];
  function tag(cls: string) { const el = document.createElement("div"); el.className = "tag3d sv-tag " + cls; o.overlay.appendChild(el); const t = { el, pos: new THREE.Vector3() }; tags.push(t); return t; }
  const houseTag = tag("home"); houseTag.el.textContent = bldName(HS); houseTag.pos.copy(P3(0, 0, gy0 + HS.ridge_m + 3));
  const wallTags = Array.from({ length: 12 }, () => tag("push")), roofTag = tag("pull");
  wallTags.forEach(t => t.el.hidden = true); roofTag.el.hidden = true;
  const CPW = [[0, .8], [45, .45], [70, -.3], [90, -.7], [135, -.5], [180, -.5]];
  const cpAt = (d: number) => { d = Math.abs(d); for (let i = 1; i < CPW.length; i++) if (d <= CPW[i][0]) { const [a, ca] = CPW[i - 1], [b, cb] = CPW[i]; return ca + (cb - ca) * (d - a) / (b - a); } return -.5; };
  // diverging pressure colour: neutral at 0, warm for push, cool for suction (|Cp| 1 = deepest)
  const PUSH = ["#cfd5da", "#ff9e64", "#ff5a3d"].map(c => new THREE.Color(c)), PULL = ["#cfd5da", "#7fb0ff", "#3f74ff"].map(c => new THREE.Color(c));
  const diverge = (cp: number, out: THREE.Color) => { const t = clamp(Math.abs(cp), 0, 1), S = cp > 0 ? PUSH : PULL; return t < .5 ? out.lerpColors(S[0], S[1], t * 2) : out.lerpColors(S[1], S[2], t * 2 - 1); };
  const base0 = new THREE.Color(0xffffff), pc = new THREE.Color();
  let panelOn = 0;
  function loads() {
    const show = st.view === "house", h = HRS[st.idx], si = W ? W.hour_sector[st.idx] : 0, rc = HS.roof_cp?.[si];
    // psf: q_h from the approach-flow gust at mean roof height, the reference the CFD Cp uses (ASCE 7 / AIJ); hours whose
    // direction fell back to ASCE 7 tables use the local roof-height gust
    const vRoof = h.appr ? h.appr[1] : h.roof ? h.roof[1] : h.gust[2], q = .00256 * vRoof * vRoof;
    // mean Cp from the fine house CFD for this hour's wind sector; ASCE 7 table when the CFD didn't run
    const ps = faces.map(f => q * (f.cp?.[si] ?? cpAt(((f.fb - h.dir) % 360 + 540) % 360 - 180)));
    // label only the hardest-pushed and hardest-sucked walls (>= 4 m), and nothing that rounds to zero
    const big = faces.map((f, k) => k).filter(k => faces[k].len >= 4);
    const hi = big.reduce((a, k) => a < 0 || ps[k] > ps[a] ? k : a, -1), lo = big.reduce((a, k) => a < 0 || ps[k] < ps[a] ? k : a, -1);
    const lab = [hi, lo].filter(k => k >= 0 && Math.abs(ps[k]) >= .05 && (k === hi ? ps[k] > 0 : ps[k] < 0));
    wallTags.forEach(t => t.el.hidden = true);
    faces.forEach((f, k) => {
      const p = ps[k], cp = q ? p / q : 0;
      diverge(cp, pc);
      if (f.panel) { const u = f.panel.uniforms; u.uCol.value.copy(pc); u.uK.value = clamp(Math.abs(cp), 0, 1); u.uSign.value = p > 0 ? 1 : -1; }
      f.mat?.color.copy(base0).lerp(pc, show ? .3 : 0);
      const li = lab.indexOf(k); if (li < 0) return; const t = wallTags[li];
      t.pos.copy(P3(f.x + f.nx, f.y + f.ny, gy0 + HS.eave_m * .6)); t.el.className = "tag3d sv-tag " + (p > 0 ? "push" : "pull");
      t.el.textContent = `${p > 0 ? "Push +" : "Suction −"}${Math.abs(p).toFixed(1)} psf`; t.el.hidden = !show;
    });
    const up = q * (rc ? Math.abs(Math.min(rc[1], 0)) : 2.0);
    roofTag.pos.copy(P3(0, 0, gy0 + HS.ridge_m + 1.2)); roofTag.el.textContent = up < .05 ? "Wind loads near zero this hour" : `Roof edges up to −${up.toFixed(1)} psf uplift`;
    roofTag.el.hidden = !show; houseTag.el.hidden = show; panelOn = show ? 1 : 0;
  }

  // selected tree: a teal ring of dots at the crown's footprint and a dotted column up the trunk, rippling
  const mkU = { uTime: { value: 0 }, uOn: { value: 0 }, uPR: { value: pr }, uCol: { value: new THREE.Color(css("--accent") || "#6fe3d3") } };
  const marker = new THREE.Points(new THREE.BufferGeometry(), new THREE.ShaderMaterial({
    uniforms: mkU, transparent: true, depthTest: false, depthWrite: false, fog: false,
    vertexShader: `attribute float aPh; uniform float uTime, uOn, uPR; varying float vA;
      void main() { float w = .5 + .5 * sin(uTime * 3. - aPh); vA = uOn * (.45 + .55 * w); gl_PointSize = uPR * (2.5 + 2.5 * w);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`,
    fragmentShader: `uniform vec3 uCol; varying float vA;
      void main() { float d = length(gl_PointCoord - .5); if (d > .5) discard; gl_FragColor = vec4(uCol * 1.6, vA * (1. - smoothstep(.3, .5, d)));
        #include <colorspace_fragment>
      }`,
  }));
  marker.frustumCulled = false; marker.renderOrder = 8; marker.visible = false; sc.add(marker);
  let markOn = 0;
  function setMarker(i: number) {
    const t = D.trees[i] as Tree | undefined; markOn = t ? 1 : 0; if (!t) return;
    const pos: number[] = [], ph: number[] = [], g = terrH(t.x, t.y), rr = t.cw / 2 + 1;
    for (let k = 0; k < 56; k++) { const a = k / 56 * 6.283, x = t.x + Math.cos(a) * rr, y = t.y + Math.sin(a) * rr; pos.push(x, terrH(x, y) + .4, -y); ph.push(a * 3); }
    for (let z = .6; z < t.h + 2.5; z += 1.1) { pos.push(t.x, g + z, -t.y); ph.push(-z * .7); }
    marker.geometry.dispose(); marker.geometry = new THREE.BufferGeometry();
    marker.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); marker.geometry.setAttribute("aPh", new THREE.Float32BufferAttribute(ph, 1));
  }

  // camera: frame the lot with the peak-hour wind coming from behind the viewer's left shoulder
  const pd = (HRS[M.peakIdx].dir || 200) * Math.PI / 180, hk = Math.max(1, Math.hypot(HS.L, HS.W) / 28);   // hk: big buildings need a wider house view
  const cams: Record<"site" | "house", Pose> = {
    site: [P3(Math.sin(pd + .55) * 220, Math.cos(pd + .55) * 220, 140), P3(0, 0, 2)],
    house: [P3(Math.sin(pd + .7) * 52 * hk, Math.cos(pd + .7) * 52 * hk, 30 * hk), P3(0, 0, 4)],
  };
  // a tree, seen from beyond it so the house sits behind it in frame; if a building would block that view, the camera
  // swings round the tree to the nearest bearing with a clear line of sight
  const blds: Rect[] = [{ cx: 0, cy: 0, L: HS.L, W: HS.W, heading: HS.heading, h: HS.ridge_m }, ...D.buildings];
  function treePose(i: number): Pose | null {
    const t = D.trees[i] as Tree | undefined; if (!t) return null;
    const g = terrH(t.x, t.y), a0 = (Math.hypot(t.x, t.y) > 1 ? Math.atan2(t.x, t.y) : pd) + .45, dist = clamp(t.h * 2.3 + t.cw, 24, 110);
    const tx = clamp(t.x, -TB, TB), ty = clamp(t.y, -TB, TB), tz = g + t.h * .55;
    const clear = (a: number) => {   // sight line from the trunk base (not just the crown) to the camera
      for (let k = 1; k <= 12; k++) {
        const f = k / 12, x = tx + Math.sin(a) * dist * f, y = ty + Math.cos(a) * dist * f, z = g + 1 + (tz + dist * .5 - g - 1) * f;
        if (blds.some(b => z < terrH(b.cx, b.cy) + (b.h || 7) + 1 && inRect(b, x, y, .5))) return false;
      }
      return true;
    };
    const a = a0 + ([0, .5, -.5, 1, -1, 1.6, -1.6, 2.3, -2.3, 3.1].find(da => clear(a0 + da)) ?? 0);
    const tgt = P3(tx, ty, tz);
    return [tgt.clone().add(new THREE.Vector3(Math.sin(a) * dist, dist * .5, -Math.cos(a) * dist)), tgt];
  }
  cam.position.copy(cams.site[0]); ctl.target.copy(cams.site[1]);
  let flight: { t0: THREE.Vector3; t1: THREE.Vector3; s0: THREE.Spherical; s1: THREE.Spherical; k: number; dur: number } | null = null, lastPose: Pose | null = null;
  const tmp = new THREE.Vector3(), sph = new THREE.Spherical();
  // a portrait screen sees a narrow slice across: pull the camera back along its line of sight so the subject still fits
  // and slide the view forward a little, so the subject sits in the open lower half instead of under the headline
  const fit = (p: Pose): Pose => {
    const k = clamp((1.3 / cam.aspect) ** .6, 1, 2.1); if (k <= 1) return p;
    const d = tmp.subVectors(p[0], p[1]).multiplyScalar(k), fwd = new THREE.Vector3(-d.x, 0, -d.z).normalize().multiplyScalar(d.length() * .3 * (k - 1));
    return [d.add(p[1]).add(fwd).clone(), p[1].clone().add(fwd)];
  };
  // eased flight: the target glides, the camera swings around it in spherical coordinates (distance eased in log space)
  function flyTo(pose: Pose, dur: number) {
    lastPose = pose; dirty = true; const p = fit(pose);
    if (g3d) { g3d.fly(p[0], p[1], RM ? 0 : dur); return; }
    if (RM || !(dur > 0)) { flight = null; cam.position.copy(p[0]); ctl.target.copy(p[1]); ctl.update(); return; }
    const s0 = new THREE.Spherical().setFromVector3(tmp.subVectors(cam.position, ctl.target)), s1 = new THREE.Spherical().setFromVector3(tmp.subVectors(p[0], p[1]));
    s1.theta = s0.theta + ((s1.theta - s0.theta) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;   // shortest way round
    flight = { t0: ctl.target.clone(), t1: p[1].clone(), s0, s1, k: 0, dur };
  }
  function stepFlight(dt: number) {
    const f = flight!; f.k = Math.min(1, f.k + dt / f.dur); const e = ease(f.k);
    ctl.target.lerpVectors(f.t0, f.t1, e);
    sph.set(Math.exp(Math.log(f.s0.radius) + (Math.log(f.s1.radius) - Math.log(f.s0.radius)) * e), f.s0.phi + (f.s1.phi - f.s0.phi) * e, f.s0.theta + (f.s1.theta - f.s0.theta) * e);
    cam.position.setFromSpherical(sph).add(ctl.target);
    if (f.k >= 1) flight = null;
  }
  const flyDur = (p: Pose) => clamp(.9 + (cam.position.distanceTo(p[0]) + ctl.target.distanceTo(p[1])) / 420, 1.1, 2.4);

  // intro: start high and wide, swing ~70 deg round while descending to the default view as the airflow draws on.
  // The procedural world builds in the background first (up to 2.6 s, slowly circling) so the flight never hitches.
  let introCalled = false, introWait = 0, revealed = false;
  const reveal = (dur: number) => { if (!revealed) { revealed = true; stream.reveal(dur); } };
  function introStart(): Pose {
    const [p, t] = cams.site, s = new THREE.Spherical().setFromVector3(p.clone().sub(t));
    s.radius = Math.min(ctl.maxDistance * .97, s.radius * 2.3); s.phi = .3; s.theta -= 1.2;
    return [new THREE.Vector3().setFromSpherical(s).add(t), t.clone()];
  }
  function intro() {
    if (introCalled) return; introCalled = true;
    if (RM) { flyTo(cams.site, 0); reveal(0); return; }
    if (wantG3d() && !g3d) { pendingIntro = true; return; }   // runs when Google's map is up
    const s = introStart();
    if (g3d) { g3d.fly(s[0], s[1], 0); flyTo(cams.site, 3.8); reveal(3.4); return; }
    flight = null; cam.position.copy(s[0]); ctl.target.copy(s[1]); ctl.update();
    ctl.autoRotate = true; ctl.autoRotateSpeed = .6; introWait = performance.now(); dirty = true;
  }
  const endWait = () => { introWait = 0; ctl.autoRotate = false; };
  function startIntroFlight() { endWait(); flyTo(cams.site, 3.8); reveal(3.3); }
  // any user grab ends a flight (and the intro) where the camera is
  const onStart = () => { flight = null; endWait(); reveal(1.2); };
  const onChange = () => { dirty = true; };
  ctl.addEventListener("start", onStart); ctl.addEventListener("change", onChange);

  function focus(t: FocusTarget) {
    const p = t.kind === "tree" ? treePose(t.i) : cams[t.kind === "house" ? "house" : "site"]; if (!p) return;
    endWait(); reveal(1.2); setMarker(t.kind === "tree" ? t.i : -1); flyTo(p, flyDur(p));
  }

  // hover: LiDAR trees
  const big = D.trees.filter((t: Tree) => t.h >= 8), v3 = new THREE.Vector3();
  const onMove = (e: PointerEvent) => {
    if (st.view === "storm") return; const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best: Tree | null = null, bd = 20;
    for (const t of big) {
      v3.copy(P3(t.x, t.y, terrH(t.x, t.y) + t.h * .75)).project(cam); if (v3.z > 1) continue;
      const d = Math.hypot((v3.x + 1) / 2 * r.width - mx, (1 - v3.y) / 2 * r.height - my); if (d < bd) { bd = d; best = t; }
    }
    o.onHover(best, mx, my);
  };
  const onLeave = () => o.onHover(null, 0, 0);
  cv.addEventListener("pointermove", onMove); cv.addEventListener("pointerleave", onLeave);

  // post: ambient occlusion (N8AO) -> bloom (only HDR pulse heads, reveal front and lightning pass its threshold) -> SMAA -> output
  const composer = new EffectComposer(R), ao = new N8AOPass(sc, cam, 1, 1);
  Object.assign(ao.configuration, { aoRadius: 2.2, distanceFalloff: 1.0, intensity: 2.6, halfRes: true, gammaCorrection: false });
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), .65, .45, 1.4);
  composer.addPass(ao); composer.addPass(bloom); composer.addPass(new SMAAPass()); composer.addPass(new OutputPass());
  let VW = 1, VH = 1;
  function resize() {
    const r = cv.getBoundingClientRect(); if (!r.width || !r.height) return; VW = r.width; VH = r.height;
    R.setSize(VW, VH, false); composer.setSize(VW, VH); cam.aspect = VW / VH; cam.updateProjectionMatrix();
    stream.resize(VW, VH); boltMat.resolution.set(VW, VH); haloMat.resolution.set(VW, VH); sky.uni.uCell.value = 5 * pr; mkU.uPR.value = pr; dirty = true;
  }
  const ro = new ResizeObserver(resize); ro.observe(cv); resize();
  function refresh() { paintSheet(); paintVeg(); loads(); dirty = true; }
  refresh();

  // adaptive resolution: drop the pixel ratio a step while frames run slower than ~45 fps, restore when there is headroom;
  // a restore that immediately has to be undone locks the lower ratio for the session
  let ema = 16.7, prAt = 0, restoredAt = -1e9, prLocked = false, frames = 0;
  function setPR(p: number) { pr = p; prAt = performance.now(); ema = 16.7; R.setPixelRatio(p); composer.setPixelRatio(p); resize(); log(`pixel ratio -> ${p}`); }
  function adapt(now: number, gap: number) {
    if (++frames < 90 || gap > 250) return;   // skip shader-compile warm-up and stalls (tab switches, builds)
    ema += (Math.min(gap, 60) - ema) * .05;
    if (pr > 1 && ema > 22 && now - prAt > 1500) { if (now - restoredAt < 9000) prLocked = true; setPR(Math.max(1, pr - .5)); }
    else if (!prLocked && pr < PRMAX && ema < 13 && now - prAt > 5000) { restoredAt = now; setPR(Math.min(PRMAX, pr + .5)); }
  }

  const pf = { n: 0, cpu: 0, gap: 0, max: 0, t: 0, first: false };
  log(`setup ${since(T0)} (synchronous part of createSiteScene)`);
  let last = performance.now(), raf = 0, credT = 1, earthT = 0;
  function loop(t: number) {
    raf = requestAnimationFrame(loop); const gap = t - last, dt = Math.min(.05, gap / 1000); last = t;
    if (document.hidden || st.view === "storm" || st.view === "plan") return;
    const c0 = performance.now(), anim = !RM, sec = anim ? t / 1000 : 0;
    if (g3d) g3d.sync(cam);
    if (introWait && ((earth ? earth.root.visible : worldReady) || t - introWait > (earth ? 4000 : 2600))) startIntroFlight();   // Google tiles: wait for the calibrated mesh
    if (flight) { stepFlight(dt); dirty = true; }
    if (stream.update(st.idx, !!(W && st.layer.flow), dt, anim)) dirty = true;
    atmos(dt, t); lightning(dt, t);
    if (strikeT >= 0) dirty = true;
    const pk = anim ? 1 - Math.exp(-dt * 7) : 1;
    pnlU.uOn.value += (panelOn - pnlU.uOn.value) * pk; pnlU.uTime.value = sec; panels.forEach(m => m.visible = pnlU.uOn.value > .005);
    mkU.uOn.value += (markOn - mkU.uOn.value) * pk; mkU.uTime.value = sec; marker.visible = mkU.uOn.value > .01;
    if (!g3d) { if (!flight) { ctl.target.x = clamp(ctl.target.x, -TB, TB); ctl.target.z = clamp(ctl.target.z, -TB, TB); } if (ctl.update(dt)) dirty = true; }
    earth?.update();
    if (o.credits && earth && (credT += dt) > 1) { credT = 0; o.credits.textContent = earth.attributions(); }
    // render on demand when nothing animates (reduced motion): only after a change, or every 250 ms while Google tiles stream
    if (!anim && !dirty && !g3d && !(earth && t - earthT > 250)) return;
    dirty = false; earthT = t; R.info.reset();
    const h = HRS[st.idx], toward = (h.dir + 180) * Math.PI / 180;
    grass?.update(sec, Math.sin(toward), Math.cos(toward), h.yard ? h.yard[1] : h.gust[2] * .5);
    trees?.update(sec, h.gust[2] * .8);
    sky.mesh.position.copy(cam.position);
    if (g3d) R.render(sc, cam); else composer.render(dt);
    const half = tags.map(g => g.el.hidden ? 0 : g.el.offsetWidth / 2);   // read layout once, before any writes
    tags.forEach((g, i) => {
      if (g.el.hidden) return; v3.copy(g.pos).project(cam); g.el.style.visibility = v3.z < 1 ? "visible" : "hidden";
      const x = clamp((v3.x + 1) / 2 * VW, half[i] + 8, VW - half[i] - 8);   // a label near the edge stays whole on screen
      g.el.style.transform = `translate(${x.toFixed(1)}px, ${((1 - v3.y) / 2 * VH).toFixed(1)}px) translate(-50%, -110%)`;
    });
    cam.getWorldDirection(fwd); const az = Math.atan2(-fwd.x, -fwd.z);
    o.needle.setAttribute("transform", `rotate(${(az * 180 / Math.PI + 180).toFixed(1)})`);
    if (anim) adapt(t, gap);
    if (PERF) {
      const now = performance.now();
      if (!pf.first) { pf.first = true; pf.t = now; log(`first frame drawn ${since(T0)} after create`); requestAnimationFrame(() => log(`first frame on screen ~${since(T0)} after create`)); return; }
      pf.n++; pf.cpu += now - c0; pf.gap += gap; pf.max = Math.max(pf.max, gap);
      if (now - pf.t > 2000) {
        const i = R.info.render;
        log(`${(1000 * pf.n / (now - pf.t)).toFixed(0)} fps | frame ${(pf.gap / pf.n).toFixed(1)} ms avg, ${pf.max.toFixed(0)} max | main-thread ${(pf.cpu / pf.n).toFixed(2)} ms | pixel ratio ${pr} | ${i.calls} draws, ${(i.triangles / 1e6).toFixed(2)}M tris, ${R.info.programs?.length} programs`);
        Object.assign(pf, { n: 0, cpu: 0, gap: 0, max: 0, t: now });
      }
    }
  }
  raf = requestAnimationFrame(loop);
  intro();   // idempotent: Report may call it again

  return {
    update(next: SceneState) {
      const viewChanged = next.view !== st.view; st = next; earth?.setMode(next.houseMode);
      if (viewChanged) { if (next.view === "site" || next.view === "house") { endWait(); reveal(1.2); markOn = 0; flyTo(cams[next.view], 1.4); } o.onHover(null, 0, 0); }
      refresh();
    },
    focus,
    intro,
    dispose() {
      dead = true; stream.dispose(); rain.dispose(); sky.dispose(); earth?.dispose(); g3d?.dispose(); cancelAnimationFrame(raf); ro.disconnect();
      if (o.gmapHost) { o.gmapHost.style.filter = ""; o.gmapHost.removeEventListener("pointermove", onMove); o.gmapHost.removeEventListener("pointerleave", onLeave); }
      cv.removeEventListener("pointerdown", engage); cv.removeEventListener("pointermove", onMove); cv.removeEventListener("pointerleave", onLeave);
      ctl.removeEventListener("start", onStart); ctl.removeEventListener("change", onChange);
      tags.forEach(t => t.el.remove()); ctl.dispose(); boltGeo.dispose();
      sc.traverse(obj => { const m = obj as THREE.Mesh; m.geometry?.dispose(); [m.material].flat().forEach(x => (x as THREE.Material | undefined)?.dispose()); });
      composer.passes.forEach(p => p.dispose()); composer.dispose(); env.dispose(); R.dispose();
    },
  };
}
