// Google Photorealistic 3D Tiles (the Google Earth mesh) as the site world.
// If the user's house is blurred in Google's imagery (privacy blur or a stale low-detail capture), the house is cut out
// of the mesh and replaced by the best-matching sharp neighbour's mesh, rotated to face the same street and scaled to
// the user's footprint (subdivision houses are usually built from the same few plans).
import * as THREE from "three";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { TilesRenderer } from "3d-tiles-renderer/three";
import { GLTFExtensionsPlugin, GoogleCloudAuthPlugin, LoadRegionPlugin, ReorientationPlugin, SphereRegion } from "3d-tiles-renderer/plugins";
import type { Model } from "./model";
import { frontFace, inRect, isLawn, rectAngle, satAt, type Rect, type Sat } from "./realism";

export type HouseMode = "auto" | "google" | "neighbour";
export type EarthStatus = { phase: "loading" | "ready" | "error" | "map3d"; blurred?: boolean; donor?: number; sharpness?: number; error?: string };

const HOLE_PAD = 1.4;   // m around the footprint: eaves, gutters and the blur halo

export function createEarth(M: Model, sc: THREE.Scene, cam: THREE.PerspectiveCamera, R: THREE.WebGLRenderer, key: string, sat: Promise<Sat>,
  onStatus: (s: EarthStatus) => void) {
  const { D, terrH } = M, HS = D.house;
  const house: Rect = { cx: 0, cy: 0, L: HS.L, W: HS.W, heading: HS.heading };
  const others: Rect[] = D.buildings.filter((b: Rect) => Math.hypot(b.cx, b.cy) < 110);

  const tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: key, autoRefreshToken: true }));
  const draco = new DRACOLoader().setDecoderPath("/draco/");
  tiles.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader: draco }));
  tiles.registerPlugin(new ReorientationPlugin({ lat: D.meta.lat * Math.PI / 180, lon: D.meta.lon * Math.PI / 180, height: 0, recenter: true }));
  // keep the lot and its neighbours at full detail wherever the camera is (sharpness test + donor mesh need it)
  const region = new LoadRegionPlugin(); tiles.registerPlugin(region);
  tiles.errorTarget = 12;

  // tiles frame after reorientation: x west, y up, z north. Scene frame: x east, y up, z south -> turn 180 deg about y
  const root = new THREE.Group(); root.rotation.y = Math.PI; root.visible = false; root.add(tiles.group); sc.add(root);

  // cut the user's house out of the Google mesh (a column over the footprint) when a donor replaces it
  const hole = { uOn: { value: 0 }, uC: { value: new THREE.Vector2(0, 0) }, uA: { value: rectAngle(house) }, uH: { value: new THREE.Vector2(HS.L / 2 + HOLE_PAD, HS.W / 2 + HOLE_PAD) } };
  tiles.addEventListener("load-model", ({ scene }: { scene: THREE.Object3D }) => {
    scene.traverse(o => {
      const m = (o as THREE.Mesh).material as THREE.MeshBasicMaterial | undefined; if (!m) return;
      m.onBeforeCompile = sh => {
        Object.assign(sh.uniforms, hole);
        sh.vertexShader = "varying vec3 vEW;\n" + sh.vertexShader.replace("#include <project_vertex>", "#include <project_vertex>\nvEW = (modelMatrix * vec4(transformed, 1.0)).xyz;");
        sh.fragmentShader = "varying vec3 vEW;\nuniform float uOn; uniform vec2 uC; uniform float uA; uniform vec2 uH;\n" + sh.fragmentShader.replace("void main() {", `void main() {
  if (uOn > .5) { vec2 d = vec2(vEW.x, -vEW.z) - uC; float lx = d.x * cos(uA) + d.y * sin(uA), lz = d.x * sin(uA) - d.y * cos(uA);
    if (abs(lx) < uH.x && abs(lz) < uH.y) discard; }`);
      };
      m.customProgramCacheKey = () => "earth-hole";
    });
  });

  let offset = 0, calibrated = false, analysed = 0, mode: HouseMode = "auto", donor: THREE.Group | null = null, status: EarthStatus = { phase: "loading" }, dead = false;
  const setStatus = (s: EarthStatus) => { status = s; onStatus(s); };
  tiles.addEventListener("load-error", (e: { error?: Error }) => {
    if (!calibrated) setStatus({ phase: "error", error: String(e.error?.message || e.error || "tile load failed") });
  });

  // ground calibration: Google heights are ellipsoidal; cast rays onto open lawn and match the 3DEP terrain there
  const ray = new THREE.Raycaster(), down = new THREE.Vector3(0, -1, 0);
  let lawn: [number, number][] = [];
  sat.then(s => {
    for (let k = 0; k < 400 && lawn.length < 48; k++) {
      const r = 18 + Math.random() * 60, a = Math.random() * 6.283, x = r * Math.cos(a), y = r * Math.sin(a);
      if (!isLawn(satAt(s, x, y)) || [house, ...others].some(b => inRect(b, x, y, 2)) || D.trees.some((t: { x: number; y: number; cw: number }) => Math.hypot(t.x - x, t.y - y) < t.cw / 2 + 1)) continue;
      lawn.push([x, y]);
    }
  });
  function calibrate() {
    if (!lawn.length) return false;
    root.position.y = 0; root.updateMatrixWorld(true);
    const d: number[] = [];
    for (const [x, y] of lawn) {
      ray.set(new THREE.Vector3(x, 4000, -y), down); ray.far = 9000;
      const h = ray.intersectObject(tiles.group, true)[0]; if (h) d.push(h.point.y - terrH(x, y));
    }
    if (d.length < 8) { root.position.y = -offset; return false; }
    d.sort((a, b) => a - b); offset = d[Math.floor(d.length * .3)];   // low-ish quantile: rays that hit shrubs read high
    root.position.y = -offset; root.updateMatrixWorld(true); return true;
  }

  // top-down render of the mesh around the lot -> per-footprint sharpness (mean |Laplacian| of luminance)
  const RT = 1024, SPAN = 230, rt = new THREE.WebGLRenderTarget(RT, RT), ortho = new THREE.OrthographicCamera(-SPAN / 2, SPAN / 2, SPAN / 2, -SPAN / 2, 1, 6000);
  ortho.position.set(0, 3000, 0); ortho.up.set(0, 0, -1); ortho.lookAt(0, 0, 0);
  function sharpness(): number[] {
    const was = hole.uOn.value; hole.uOn.value = 0;
    const px = new Uint8Array(RT * RT * 4), bg = sc.background, fog = sc.fog; sc.background = null; sc.fog = null;
    R.setRenderTarget(rt); R.setClearColor(0, 1); R.clear(); R.render(root, ortho); R.readRenderTargetPixels(rt, 0, 0, RT, RT, px); R.setRenderTarget(null);
    sc.background = bg; sc.fog = fog; hole.uOn.value = was;
    const lum = (i: number, j: number) => { const o = ((RT - 1 - j) * RT + i) * 4; return .3 * px[o] + .59 * px[o + 1] + .11 * px[o + 2]; };   // j: row from top
    return [house, ...others].map(b => {
      let s = 0, n = 0; const a = rectAngle(b);
      for (let u = -.42; u <= .42; u += .012) for (let v = -.42; v <= .42; v += .012) {
        const lx = u * b.L, lz = v * b.W, x = b.cx + lx * Math.cos(a) + lz * Math.sin(a), y = b.cy + lx * Math.sin(a) - lz * Math.cos(a);
        const i = Math.round((x + SPAN / 2) / SPAN * RT), j = Math.round((SPAN / 2 - y) / SPAN * RT);
        if (i < 1 || j < 1 || i > RT - 2 || j > RT - 2) continue;
        const c = lum(i, j); if (c === 0) continue;   // no mesh here
        s += Math.abs(4 * c - lum(i - 1, j) - lum(i + 1, j) - lum(i, j - 1) - lum(i, j + 1)); n++;
      }
      return n > 40 ? s / n : NaN;
    });
  }

  // donor: copy the visible mesh triangles over a neighbour's footprint into the user's footprint
  function transplant(s: Sat, di: number) {
    const b = others[di], fb = frontFace(s, b), fh = frontFace(s, house);
    const ab = rectAngle(b), ah = rectAngle(house);
    const nrm = (r: Rect, a: number, f: number) => { const v = [[Math.cos(a), Math.sin(a)], [-Math.cos(a), -Math.sin(a)], [Math.sin(a), -Math.cos(a)], [-Math.sin(a), Math.cos(a)]][f]; return { f: v, s: [-v[1], v[0]], depth: f < 2 ? r.L : r.W, width: f < 2 ? r.W : r.L }; };
    const B = nrm(b, ab, fb), H = nrm(house, ah, fh);
    const sd = THREE.MathUtils.clamp(H.depth / B.depth, .75, 1.33), sw = THREE.MathUtils.clamp(H.width / B.width, .75, 1.33), dz = -terrH(b.cx, b.cy);
    const map = (e: number, n: number, out: number[]) => {
      const pe = e - b.cx, pn = n - b.cy, d = (pe * B.f[0] + pn * B.f[1]) * sd, w = (pe * B.s[0] + pn * B.s[1]) * sw;
      out[0] = d * H.f[0] + w * H.s[0]; out[1] = d * H.f[1] + w * H.s[1];
    };
    const g = new THREE.Group(), v = new THREE.Vector3(), tri = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()], q = [0, 0];
    const texCache = new Map<THREE.Texture, THREE.Texture>();
    root.updateMatrixWorld(true);
    tiles.group.traverse(o => {
      const mesh = o as THREE.Mesh; if (!mesh.isMesh) return;
      const geo = mesh.geometry, pos = geo.attributes.position, uv = geo.attributes.uv, idx = geo.index; if (!uv) return;
      const P: number[] = [], U: number[] = [], nTri = (idx ? idx.count : pos.count) / 3;
      for (let t = 0; t < nTri; t++) {
        const ids = [0, 1, 2].map(k => idx ? idx.getX(3 * t + k) : 3 * t + k);
        ids.forEach((id, k) => tri[k].fromBufferAttribute(pos, id).applyMatrix4(mesh.matrixWorld));
        v.copy(tri[0]).add(tri[1]).add(tri[2]).divideScalar(3);
        if (!inRect(b, v.x, -v.z, HOLE_PAD)) continue;
        ids.forEach((id, k) => { map(tri[k].x, -tri[k].z, q); P.push(q[0], tri[k].y + dz, -q[1]); U.push(uv.getX(id), uv.getY(id)); });
      }
      if (!P.length) return;
      // copy the texture: the source tile may unload (and dispose its texture) when the camera moves
      const src = (mesh.material as THREE.MeshBasicMaterial).map;
      let tex = src ? texCache.get(src) : undefined;
      if (src && !tex) {
        const img = src.image as CanvasImageSource & { width: number; height: number }, c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
        c.getContext("2d")!.drawImage(img, 0, 0); tex = new THREE.CanvasTexture(c); tex.flipY = src.flipY; tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8; texCache.set(src, tex);
      }
      const ng = new THREE.BufferGeometry(); ng.setAttribute("position", new THREE.Float32BufferAttribute(P, 3)); ng.setAttribute("uv", new THREE.Float32BufferAttribute(U, 2)); ng.computeVertexNormals();
      g.add(new THREE.Mesh(ng, new THREE.MeshBasicMaterial({ map: tex || null, toneMapped: false, side: THREE.DoubleSide })));
    });
    return g;
  }

  function apply() {
    const want = status.donor != null && (mode === "neighbour" || (mode === "auto" && status.blurred));
    hole.uOn.value = want ? 1 : 0; if (donor) donor.visible = !!want;
  }

  async function analyse() {
    const s = await sat; if (dead) return;
    const sh = sharpness(), hs = sh[0], ref = sh.slice(1).filter(Number.isFinite).sort((a, b) => a - b), med = ref[Math.floor(ref.length / 2)];
    const blurred = Number.isFinite(hs) && ref.length >= 2 ? hs < .45 * med : false;
    // donor: sharp neighbour with the closest footprint (either orientation), nearest first on ties
    let di = -1, best = 1e9;
    others.forEach((b, k) => {
      if (!(sh[k + 1] > .7 * med)) return;
      const fit = Math.min(Math.abs(Math.log(b.L / HS.L)) + Math.abs(Math.log(b.W / HS.W)), Math.abs(Math.log(b.L / HS.W)) + Math.abs(Math.log(b.W / HS.L)));
      const score = fit + Math.hypot(b.cx, b.cy) / 400; if (fit < .45 && score < best) { best = score; di = k; }
    });
    if (donor) { sc.remove(donor); donor.traverse(o => { const m = o as THREE.Mesh; m.geometry?.dispose(); (m.material as THREE.MeshBasicMaterial | undefined)?.map?.dispose(); }); donor = null; }
    if (di >= 0) { donor = transplant(s, di); sc.add(donor); }
    setStatus({ phase: "ready", blurred, donor: di >= 0 ? di : undefined, sharpness: hs / med });
    apply();
  }

  let quiet: ReturnType<typeof setTimeout> | undefined;
  tiles.addEventListener("tiles-load-end", () => {
    clearTimeout(quiet);
    quiet = setTimeout(() => {
      if (dead) return;
      if (!calibrated && (calibrated = calibrate())) { root.visible = true; }
      if (calibrated && analysed < 3) { analysed++; analyse(); }   // re-run as finer LODs arrive
    }, 900);
  });

  let regionAdded = false;
  return {
    root,
    update() {
      if (!regionAdded && calibrated) {   // region lives in the tileset frame; place it once the ground height is known
        const c = new THREE.Vector3(0, 0, 0); tiles.group.worldToLocal(c);
        region.addRegion(new SphereRegion({ errorTarget: 6, sphere: new THREE.Sphere(c, 120) })); regionAdded = true;
      }
      cam.updateMatrixWorld(); tiles.setCamera(cam); tiles.setResolutionFromRenderer(cam, R); tiles.update();
    },
    setMode(m: HouseMode) { mode = m; apply(); },
    status: () => status,
    attributions() { const a: { value: string; type: string }[] = []; tiles.getAttributions(a); return a.filter(x => x.type === "string").map(x => x.value).join(" "); },
    dispose() { dead = true; clearTimeout(quiet); tiles.dispose(); draco.dispose(); rt.dispose(); if (donor) sc.remove(donor); sc.remove(root); },
  };
}
