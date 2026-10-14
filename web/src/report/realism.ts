// Realistic scene dressing: procedural PBR materials, houses with real roof colours, lawn grass, and
// LiDAR-sized procedural trees. Geometry stays tied to the data: footprints, eave/ridge heights,
// tree height and crown width all come from the engine; only surface detail is synthesised.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { clamp, type Model, type Tree } from "./model";

// ---------- satellite pixels (for roof colours, lawn mask, driveway detection) ----------
export type Sat = { w: number; px: Uint8ClampedArray; half: number };
export async function loadSat(src: string, half: number): Promise<Sat> {
  const img = new Image();
  await new Promise((ok, fail) => { img.onload = ok; img.onerror = fail; img.src = src; });   // not decode(): it can stall in background tabs
  const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
  const g = c.getContext("2d", { willReadFrequently: true })!; g.drawImage(img, 0, 0);
  return { w: img.width, px: g.getImageData(0, 0, img.width, img.height).data, half };
}
export function satAt(s: Sat, x: number, y: number): [number, number, number] {
  const i = clamp(Math.floor((s.half - y) / (2 * s.half) * s.w), 0, s.w - 1), j = clamp(Math.floor((x + s.half) / (2 * s.half) * s.w), 0, s.w - 1), o = (i * s.w + j) * 4;
  return [s.px[o], s.px[o + 1], s.px[o + 2]];
}
export const isLawn = ([r, g, b]: number[]) => g > b + 10 && g >= r * .95 && r < g * 1.2 && g > 55 && g < 215;
const isPaving = ([r, g, b]: number[]) => Math.max(r, g, b) - Math.min(r, g, b) < 24 && (r + g + b) / 3 > 125;

// ---------- canvas textures: albedo + normal from a height field ----------
function canvasTex(size: number, paint: (g: CanvasRenderingContext2D, h: CanvasRenderingContext2D) => void, normalStrength: number) {
  const mk = () => { const c = document.createElement("canvas"); c.width = c.height = size; return c; };
  const albedo = mk(), height = mk();
  paint(albedo.getContext("2d")!, height.getContext("2d")!);
  const hd = height.getContext("2d")!.getImageData(0, 0, size, size).data, nc = mk(), ng = nc.getContext("2d")!, nd = ng.createImageData(size, size);
  const H = (x: number, y: number) => hd[(((y + size) % size) * size + ((x + size) % size)) * 4] / 255;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const dx = (H(x + 1, y) - H(x - 1, y)) * normalStrength, dy = (H(x, y + 1) - H(x, y - 1)) * normalStrength, l = Math.hypot(dx, dy, 1), o = (y * size + x) * 4;
    nd.data[o] = (-dx / l * .5 + .5) * 255; nd.data[o + 1] = (dy / l * .5 + .5) * 255; nd.data[o + 2] = (1 / l * .5 + .5) * 255; nd.data[o + 3] = 255;
  }
  ng.putImageData(nd, 0, 0);
  const map = new THREE.CanvasTexture(albedo), normal = new THREE.CanvasTexture(nc);
  map.colorSpace = THREE.SRGBColorSpace;
  for (const t of [map, normal]) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; }
  return { map, normal };
}
const rnd = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();

// asphalt shingles, 3 m tile, 0.125 m exposure (5"), staggered tabs; grey so the roof colour comes from imagery
const SHINGLE_M = 3;
function shingles() {
  return canvasTex(512, (g, h) => {
    const rows = 24, rh = 512 / rows;
    g.fillStyle = "#777"; g.fillRect(0, 0, 512, 512); h.fillStyle = "#000"; h.fillRect(0, 0, 512, 512);
    for (let r = 0; r < rows; r++) {
      let x = (r % 2) * 21 - 21;
      while (x < 512) {
        const w = 38 + rnd() * 10, v = 105 + rnd() * 45;
        g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x, r * rh, w - 1.5, rh);
        for (let k = 0; k < 40; k++) { const q = v + (rnd() - .5) * 60; g.fillStyle = `rgb(${q},${q},${q})`; g.fillRect(x + rnd() * w, r * rh + rnd() * rh, 2, 2); }   // granules
        const gr = h.createLinearGradient(0, r * rh, 0, (r + 1) * rh); gr.addColorStop(0, "#333"); gr.addColorStop(1, "#ddd");
        h.fillStyle = gr; h.fillRect(x, r * rh, w - 1.5, rh);
        x += w;
      }
      g.fillStyle = "rgba(0,0,0,.35)"; g.fillRect(0, (r + 1) * rh - 1.5, 512, 1.5);
    }
  }, 3);
}
// lap siding, 1.8 m tile, 10 boards of 7"
const SIDING_M = 1.8;
function siding() {
  return canvasTex(256, (g, h) => {
    g.fillStyle = "#f2f0eb"; g.fillRect(0, 0, 256, 256);
    for (let b = 0; b < 10; b++) {
      const y0 = b * 25.6, gr = h.createLinearGradient(0, y0, 0, y0 + 25.6); gr.addColorStop(0, "#222"); gr.addColorStop(.92, "#eee"); gr.addColorStop(1, "#111");
      h.fillStyle = gr; h.fillRect(0, y0, 256, 25.6);
      g.fillStyle = "rgba(0,0,0,.10)"; g.fillRect(0, y0 + 23.5, 256, 2.1);
      for (let k = 0; k < 30; k++) { g.fillStyle = `rgba(0,0,0,${rnd() * .03})`; g.fillRect(rnd() * 256, y0 + rnd() * 24, 30 + rnd() * 60, 1); }   // wood-grain embossing
    }
  }, 4);
}
// running-bond brick, 1.6 m tile
const BRICK_M = 1.6;
function brick() {
  return canvasTex(512, (g, h) => {
    g.fillStyle = "#bdb5a8"; g.fillRect(0, 0, 512, 512); h.fillStyle = "#000"; h.fillRect(0, 0, 512, 512);
    const bh = 512 / 24, bw = 512 / 8;
    for (let r = 0; r < 24; r++) for (let c = -1; c < 9; c++) {
      const x = c * bw + (r % 2) * bw / 2, v = rnd();
      g.fillStyle = `rgb(${150 + v * 45},${70 + v * 30},${55 + v * 25})`; g.fillRect(x + 3, r * bh + 3, bw - 6, bh - 6);
      h.fillStyle = "#bbb"; h.fillRect(x + 3, r * bh + 3, bw - 6, bh - 6);
    }
  }, 3);
}
// sectional garage door panels
function garageTex() {
  return canvasTex(256, (g, h) => {
    g.fillStyle = "#f3f3f1"; g.fillRect(0, 0, 256, 256); h.fillStyle = "#888"; h.fillRect(0, 0, 256, 256);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      h.fillStyle = "#bbb"; h.fillRect(c * 64 + 6, r * 64 + 8, 52, 48);
      g.fillStyle = "rgba(0,0,0,.06)"; g.fillRect(c * 64 + 6, r * 64 + 54, 52, 2);
    }
    for (let r = 1; r < 4; r++) { g.fillStyle = "rgba(0,0,0,.25)"; g.fillRect(0, r * 64 - 1, 256, 2); h.fillStyle = "#000"; h.fillRect(0, r * 64 - 1, 256, 2); }
  }, 2);
}
// fine noise normal map for the ground (breaks up the flat satellite drape)
export function groundDetail() {
  return canvasTex(256, (g, h) => {
    g.fillStyle = "#fff"; g.fillRect(0, 0, 256, 256);
    const id = h.createImageData(256, 256); for (let i = 0; i < id.data.length; i += 4) { const v = 90 + rnd() * 80; id.data[i] = id.data[i + 1] = id.data[i + 2] = v; id.data[i + 3] = 255; }
    h.putImageData(id, 0, 0);
  }, 1.2).normal;
}

// ---------- houses ----------
export type Rect = { cx: number; cy: number; L: number; W: number; heading: number; h?: number };
const SIDING_COLORS = ["#ece7dc", "#d9d4c8", "#c9ccc8", "#b9c0c4", "#e3dccd", "#cfc6b4", "#a9b2ad", "#f4f1ea"];

// Quad helper: four local points (CCW seen from outside), uv in metres along t and b, divided by tile size.
function quads(list: { p: THREE.Vector3[]; t: THREE.Vector3; b: THREE.Vector3; tile: number; group: number }[]) {
  const pos: number[] = [], uv: number[] = [], groups: [number, number, number][] = [];
  for (const q of list) {
    const start = pos.length / 3, o = q.p[0];
    const tri = q.p.length === 4 ? [0, 1, 2, 0, 2, 3] : [0, 1, 2];
    for (const k of tri) { const p = q.p[k], d = p.clone().sub(o); pos.push(p.x, p.y, p.z); uv.push(d.dot(q.t) / q.tile, d.dot(q.b) / q.tile); }
    groups.push([start, tri.length, q.group]);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  groups.sort((a, b) => a[2] - b[2]); let cur = -1, st = 0, cnt = 0;   // merge consecutive groups of the same material
  for (const [s, c, m] of groups) { if (m === cur && s === st + cnt) cnt += c; else { if (cur >= 0) g.addGroup(st, cnt, cur); cur = m; st = s; cnt = c; } }
  if (cur >= 0) g.addGroup(st, cnt, cur);
  g.computeVertexNormals(); return g;
}
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);

// local axes of a footprint rect in world (east, north): local +x -> (cos a, sin a), local +z -> (sin a, -cos a)
export const rectAngle = (b: Rect) => (90 - b.heading) * Math.PI / 180;
export function inRect(b: Rect, x: number, y: number, pad = 0) {
  const a = rectAngle(b), dx = x - b.cx, dy = y - b.cy, lx = dx * Math.cos(a) + dy * Math.sin(a), lz = dx * Math.sin(a) - dy * Math.cos(a);
  return Math.abs(lx) < b.L / 2 + pad && Math.abs(lz) < b.W / 2 + pad;
}
// front face (0:+x 1:-x 2:+z 3:-z) = the one looking onto the most paving (driveway / street) in the imagery
export function frontFace(sat: Sat, b: Rect) {
  const a = rectAngle(b), L = b.L, W = b.W, N = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  let front = 2, best = -1;
  N.forEach((n, k) => {
    let c = 0;
    for (let d = 3; d < 14; d += 1.5) for (let s = -.4; s <= .4; s += .2) {
      const lx = n[0] ? n[0] * (L / 2 + d) : s * L, lz = n[1] ? n[1] * (W / 2 + d) : s * W;
      if (isPaving(satAt(sat, b.cx + lx * Math.cos(a) + lz * Math.sin(a), b.cy + lx * Math.sin(a) - lz * Math.cos(a)))) c++;
    }
    if (c > best) { best = c; front = k; }
  });
  return front;
}

export function buildHouses(M: Model, sc: THREE.Scene, sat: Sat) {
  const { D, terrH } = M, HS = D.house;
  const tx = { shingle: shingles(), siding: siding(), brick: brick(), garage: garageTex() };
  const trim = new THREE.MeshStandardMaterial({ color: 0xf4f4f0, roughness: .6, side: THREE.DoubleSide });
  const plinth = new THREE.MeshStandardMaterial({ color: 0x8c8a85, roughness: .95 });
  const frames: THREE.Matrix4[] = [], glass: THREE.Matrix4[] = [], doors: { m: THREE.Matrix4; c: THREE.Color }[] = [], garages: THREE.Matrix4[] = [];
  let houseWalls: THREE.MeshStandardMaterial[] = [];
  const dummy = new THREE.Object3D();

  const all: (Rect & { isHouse: boolean })[] = [{ cx: 0, cy: 0, L: HS.L, W: HS.W, heading: HS.heading, isHouse: true }, ...D.buildings.map((b: Rect) => ({ ...b, isHouse: false }))];
  for (const b of all) {
    const seed = Math.abs(Math.round(b.cx * 7.3 + b.cy * 3.1));
    const eave = b.isHouse ? HS.eave_m : Math.max(3, b.h! - 1.6), ridge = b.isHouse ? HS.ridge_m : b.h! + 1.6;
    const stories = b.isHouse ? HS.stories : Math.max(1, Math.round(eave / 3));
    const base = terrH(b.cx, b.cy) - .15, a = (90 - b.heading) * Math.PI / 180, L = b.L, W = b.W;
    const grp = new THREE.Group(); grp.position.set(b.cx, base, -b.cy); grp.rotation.y = a; sc.add(grp); grp.updateMatrixWorld();
    // local axes in world (east, north): local +x -> (cos a, sin a), local +z -> (sin a, -cos a)
    const toWorld = (lx: number, lz: number) => [b.cx + lx * Math.cos(a) + lz * Math.sin(a), b.cy + lx * Math.sin(a) - lz * Math.cos(a)];

    // roof colour: mean imagery colour over the inner footprint (the real roof)
    let rs = 0, gs = 0, bs = 0, n = 0;
    for (let i = -3; i <= 3; i++) for (let j = -3; j <= 3; j++) { const [x, y] = toWorld(i / 3.6 * (L / 2 - .8), j / 3.6 * (W / 2 - .8)), c = satAt(sat, x, y); rs += c[0]; gs += c[1]; bs += c[2]; n++; }
    const roofCol = new THREE.Color().setRGB(rs / n / 255 * 1.08, gs / n / 255 * 1.08, bs / n / 255 * 1.08, THREE.SRGBColorSpace);
    const shingle = new THREE.MeshStandardMaterial({ map: tx.shingle.map, normalMap: tx.shingle.normal, color: roofCol, roughness: .92, side: THREE.DoubleSide });

    const isBrick = !b.isHouse && seed % 5 < 2;
    const wallMat = isBrick
      ? new THREE.MeshStandardMaterial({ map: tx.brick.map, normalMap: tx.brick.normal, roughness: .9 })
      : new THREE.MeshStandardMaterial({ map: tx.siding.map, normalMap: tx.siding.normal, color: new THREE.Color(SIDING_COLORS[b.isHouse ? 0 : seed % SIDING_COLORS.length]), roughness: .75 });
    const tile = isBrick ? BRICK_M : SIDING_M;

    // walls: faces 0:+x 1:-x 2:+z 3:-z (house loads colour each face separately)
    const y0 = -2, Y = V(0, 1, 0);
    const walls = quads([
      { p: [V(L / 2, y0, W / 2), V(L / 2, y0, -W / 2), V(L / 2, eave, -W / 2), V(L / 2, eave, W / 2)], t: V(0, 0, -1), b: Y, tile, group: 0 },
      { p: [V(-L / 2, y0, -W / 2), V(-L / 2, y0, W / 2), V(-L / 2, eave, W / 2), V(-L / 2, eave, -W / 2)], t: V(0, 0, 1), b: Y, tile, group: 1 },
      { p: [V(-L / 2, y0, W / 2), V(L / 2, y0, W / 2), V(L / 2, eave, W / 2), V(-L / 2, eave, W / 2)], t: V(1, 0, 0), b: Y, tile, group: 2 },
      { p: [V(L / 2, y0, -W / 2), V(-L / 2, y0, -W / 2), V(-L / 2, eave, -W / 2), V(L / 2, eave, -W / 2)], t: V(-1, 0, 0), b: Y, tile, group: 3 },
    ]);
    const wallMats = b.isHouse ? [0, 1, 2, 3].map(() => wallMat.clone()) : [wallMat, wallMat, wallMat, wallMat];
    if (b.isHouse) houseWalls = wallMats;
    const wm = new THREE.Mesh(walls, wallMats); wm.castShadow = wm.receiveShadow = true; grp.add(wm);
    const pl = new THREE.Mesh(new THREE.BoxGeometry(L + .06, 2.45, W + .06), plinth); pl.position.y = -.78; pl.receiveShadow = true; grp.add(pl);

    // roof: gable along the long axis or hip, 0.45 m overhang, fascia and soffit
    const o = .45, gable = b.isHouse ? true : seed % 3 !== 0, pitch = (ridge - eave) / (W / 2), ye = eave - o * pitch, ex = L / 2 + (gable ? .25 : o), ez = W / 2 + o;
    const rq: Parameters<typeof quads>[0] = [];
    if (gable) {
      rq.push({ p: [V(-ex, ye, ez), V(ex, ye, ez), V(ex, ridge, 0), V(-ex, ridge, 0)], t: V(1, 0, 0), b: V(0, ridge - ye, -ez).normalize(), tile: SHINGLE_M, group: 0 });
      rq.push({ p: [V(ex, ye, -ez), V(-ex, ye, -ez), V(-ex, ridge, 0), V(ex, ridge, 0)], t: V(-1, 0, 0), b: V(0, ridge - ye, ez).normalize(), tile: SHINGLE_M, group: 0 });
      for (const s of [1, -1]) {   // gable-end wall triangles (siding) and rake trim
        rq.push({ p: s > 0 ? [V(L / 2, eave, W / 2), V(L / 2, eave, -W / 2), V(L / 2, ridge, 0)] : [V(-L / 2, eave, -W / 2), V(-L / 2, eave, W / 2), V(-L / 2, ridge, 0)], t: V(0, 0, -s), b: Y, tile, group: 3 });
        rq.push({ p: s > 0 ? [V(ex, ye - .22, ez), V(ex, ye, ez), V(ex, ridge, 0), V(ex, ridge - .22, 0)] : [V(-ex, ye, ez), V(-ex, ye - .22, ez), V(-ex, ridge - .22, 0), V(-ex, ridge, 0)], t: V(0, 0, 1), b: Y, tile: 1, group: 1 });
        rq.push({ p: s > 0 ? [V(ex, ye, -ez), V(ex, ye - .22, -ez), V(ex, ridge - .22, 0), V(ex, ridge, 0)] : [V(-ex, ye - .22, -ez), V(-ex, ye, -ez), V(-ex, ridge, 0), V(-ex, ridge - .22, 0)], t: V(0, 0, 1), b: Y, tile: 1, group: 1 });
      }
      for (const s of [1, -1]) {   // fascia + soffit on the eaves
        rq.push({ p: s > 0 ? [V(-ex, ye - .22, ez), V(ex, ye - .22, ez), V(ex, ye, ez), V(-ex, ye, ez)] : [V(ex, ye - .22, -ez), V(-ex, ye - .22, -ez), V(-ex, ye, -ez), V(ex, ye, -ez)], t: V(1, 0, 0), b: Y, tile: 1, group: 1 });
        rq.push({ p: s > 0 ? [V(-ex, ye - .22, W / 2), V(ex, ye - .22, W / 2), V(ex, ye - .22, ez), V(-ex, ye - .22, ez)] : [V(ex, ye - .22, -W / 2), V(-ex, ye - .22, -W / 2), V(-ex, ye - .22, -ez), V(ex, ye - .22, -ez)], t: V(1, 0, 0), b: V(0, 0, 1), tile: 1, group: 2 });
      }
    } else {
      const run = Math.min((ridge - eave) / pitch, Math.min(L, W) / 2 - .01) + o, sq = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
      const bot = sq.map(([s, t]) => V(s * (L / 2 + o), ye, t * ez)), top = sq.map(([s, t]) => V(s * Math.max(L / 2 + o - run, 0), ridge, t * Math.max(ez - run, 0)));
      for (let k = 0; k < 4; k++) {
        const k2 = (k + 1) % 4, t = bot[k2].clone().sub(bot[k]).normalize(), mid = top[k].clone().add(top[k2]).multiplyScalar(.5).sub(bot[k].clone().add(bot[k2]).multiplyScalar(.5));
        const bv = mid.sub(t.clone().multiplyScalar(mid.dot(t))).normalize();
        rq.push({ p: [bot[k], bot[k2], top[k2], top[k]].map(v => v.clone()), t, b: bv, tile: SHINGLE_M, group: 0 });
        const d = V(0, -.22, 0);   // fascia band and soffit
        rq.push({ p: [bot[k].clone().add(d), bot[k2].clone().add(d), bot[k2].clone(), bot[k].clone()], t, b: Y, tile: 1, group: 1 });
      }
      rq.push({ p: [V(-L / 2 - o, ye - .22, ez), V(L / 2 + o, ye - .22, ez), V(L / 2 + o, ye - .22, -ez), V(-L / 2 - o, ye - .22, -ez)], t: V(1, 0, 0), b: V(0, 0, -1), tile: 1, group: 2 });
    }
    rq.push({ p: [V(0, 0, 0), V(0, 0, 0), V(0, 0, 0)], t: V(1, 0, 0), b: Y, tile: 1, group: 3 });   // keep group 3 present
    const roof = new THREE.Mesh(quads(rq), [shingle, trim, trim, wallMat]); roof.castShadow = roof.receiveShadow = true; grp.add(roof);

    // front = the face looking onto the most paving (driveway / street) in the imagery
    const faces = [{ n: [1, 0], len: W }, { n: [-1, 0], len: W }, { n: [0, 1], len: L }, { n: [0, -1], len: L }];
    const front = frontFace(sat, b);

    // openings: windows per storey, garage + door on the front face
    const place = (k: number, along: number, y: number, list: THREE.Matrix4[], sx = 1, sy = 1) => {
      const f = faces[k], d = (f.n[0] ? L : W) / 2 + .02;
      dummy.position.set(f.n[0] ? f.n[0] * d : along, y, f.n[1] ? f.n[1] * d : along);
      dummy.rotation.set(0, f.n[0] ? f.n[0] * Math.PI / 2 : (f.n[1] > 0 ? 0 : Math.PI), 0); dummy.scale.set(sx, sy, 1);
      dummy.updateMatrix(); list.push(grp.matrixWorld.clone().multiply(dummy.matrix));
    };
    faces.forEach((f, k) => {
      const len = f.len, cnt = Math.max(1, Math.floor((len - 2) / 3.2)), garageHere = k === front;
      const gW = len > 9 ? 4.9 : len > 6 ? 2.7 : 0, gx = len / 2 - gW / 2 - 1;
      for (let s = 0; s < stories; s++) for (let w = 0; w < cnt; w++) {
        const off = (w - (cnt - 1) / 2) * (len - 2) / cnt;
        if (garageHere && s === 0 && (Math.abs(off - gx) < gW / 2 + .9 || Math.abs(off) < 1.2)) continue;
        place(k, off, s * 3 + 1.55, frames); place(k, off, s * 3 + 1.55, glass);
      }
      if (garageHere) {
        if (gW) place(k, gx, 1.1, garages, gW, 2.2);
        const dm: THREE.Matrix4[] = []; place(k, gW ? -1.5 : 0, 1.05, dm);
        doors.push({ m: dm[0], c: new THREE.Color(["#3b2a22", "#1f2f45", "#5a1f1f", "#2c3b2f", "#222"][seed % 5]) });
      }
    });
  }

  // instanced openings (one draw call each)
  const fGeo = mergeGeometries([new THREE.BoxGeometry(1.05, 1.5, .1), new THREE.BoxGeometry(.05, 1.4, .06).translate(0, 0, .06), new THREE.BoxGeometry(.95, .05, .06).translate(0, .1, .06)]);
  const inst = (geo: THREE.BufferGeometry, mat: THREE.Material, list: THREE.Matrix4[]) => {
    const m = new THREE.InstancedMesh(geo, mat, list.length); list.forEach((x, i) => m.setMatrixAt(i, x)); m.castShadow = true; m.receiveShadow = true; sc.add(m); return m;
  };
  inst(fGeo, trim, frames);
  inst(new THREE.PlaneGeometry(.92, 1.37).translate(0, 0, .055), new THREE.MeshPhysicalMaterial({ color: 0x1b252e, roughness: .04, metalness: 0, envMapIntensity: 1.6, clearcoat: 1 }), glass);
  inst(new THREE.PlaneGeometry(1, 1).translate(0, 0, .03), new THREE.MeshStandardMaterial({ map: tx.garage.map, normalMap: tx.garage.normal, roughness: .5 }), garages);
  const dm = inst(mergeGeometries([new THREE.BoxGeometry(1.0, 2.15, .08).translate(0, 0, .04)]), new THREE.MeshStandardMaterial({ roughness: .45 }), doors.map(d => d.m));
  doors.forEach((d, i) => dm.setColorAt(i, d.c));
  inst(new THREE.BoxGeometry(1.2, 2.3, .06), trim, doors.map(d => d.m));   // door casing
  return { houseWalls };
}

// ---------- lawn grass ----------
// Instanced blades on pixels the imagery classifies as lawn, outside buildings and tree crowns. Blades bend
// downwind with the hour's yard-level gust (shader), coloured from the imagery underneath. Built in ~6 ms slices so
// the intro flight stays smooth; buildings and crowns are rasterised once into a 0.5 m mask instead of tested per blade.
export async function buildGrass(M: Model, sc: THREE.Scene, sat: Sat, alive: () => boolean, radius = 75, target = 420000) {
  const { D, terrH } = M, HS = D.house;
  const RES = .5, N = Math.ceil(2 * radius / RES) + 1, mask = new Uint8Array(N * N);
  const ix = (v: number) => Math.floor((v + radius) / RES);
  const mark = (i: number, j: number) => { if (i >= 0 && j >= 0 && i < N && j < N) mask[j * N + i] = 1; };
  for (const b of [{ cx: 0, cy: 0, L: HS.L, W: HS.W, heading: HS.heading }, ...D.buildings] as Rect[]) {
    const R = Math.hypot(b.L / 2 + 1, b.W / 2 + 1); if (Math.hypot(b.cx, b.cy) - R > radius) continue;
    for (let i = ix(b.cx - R); i <= ix(b.cx + R); i++) for (let j = ix(b.cy - R); j <= ix(b.cy + R); j++)
      if (inRect(b, (i + .5) * RES - radius, (j + .5) * RES - radius, 1)) mark(i, j);
  }
  for (const t of D.trees as Tree[]) {
    const r = t.cw / 2 * .8; if (Math.hypot(t.x, t.y) - r > radius) continue;
    for (let i = ix(t.x - r); i <= ix(t.x + r); i++) for (let j = ix(t.y - r); j <= ix(t.y + r); j++)
      if (Math.hypot((i + .5) * RES - radius - t.x, (j + .5) * RES - radius - t.y) < r) mark(i, j);
  }

  const blade = new THREE.BufferGeometry(), segs = 4, bp: number[] = [], buv: number[] = [], bn: number[] = [], idx: number[] = [];
  for (let s = 0; s <= segs; s++) { const v = s / segs, w = .03 * (1 - v * .85); bp.push(-w, v, 0, w, v, 0); buv.push(0, v, 1, v); bn.push(0, 1, 0, 0, 1, 0); }   // normals up: lit like the lawn
  for (let s = 0; s < segs; s++) { const a = s * 2; idx.push(a, a + 1, a + 3, a, a + 3, a + 2, a, a + 3, a + 1, a, a + 2, a + 3); }   // both windings, normals stay up
  blade.setAttribute("position", new THREE.Float32BufferAttribute(bp, 3)); blade.setAttribute("uv", new THREE.Float32BufferAttribute(buv, 2)); blade.setAttribute("normal", new THREE.Float32BufferAttribute(bn, 3)); blade.setIndex(idx);

  const mats = new Float32Array(target * 16), cols = new Float32Array(target * 3), d = new THREE.Object3D(), c = new THREE.Color();
  let n = 0, i = 0;
  while (i < target * 1.8 && n < target) {
    const until = performance.now() + 6;
    for (let k = 0; i < target * 1.8 && n < target && (k & 511 || performance.now() < until); i++, k++) {
      const r = radius * Math.sqrt(rnd()), th = rnd() * 6.2832, x = r * Math.cos(th), y = r * Math.sin(th), px = satAt(sat, x, y);
      const mi = ix(x), mj = ix(y);
      if (!isLawn(px) || mask[mj * N + mi]) continue;
      const h = .1 + rnd() * .14;
      d.position.set(x, terrH(x, y) - .01, -y); d.rotation.set((rnd() - .5) * .5, rnd() * 6.28, (rnd() - .5) * .5); d.scale.set(1, h, 1); d.updateMatrix();
      d.matrix.toArray(mats, n * 16);
      const v = .85 + rnd() * .3; c.setRGB(px[0] / 255 * v, px[1] / 255 * v * 1.04, px[2] / 255 * v, THREE.SRGBColorSpace); c.toArray(cols, n * 3);
      n++;
    }
    await new Promise(ok => setTimeout(ok, 0));
    if (!alive()) { blade.dispose(); return null; }
  }
  const uni = { uTime: { value: 0 }, uBend: { value: new THREE.Vector2(0, 0) } };
  const mat = new THREE.MeshStandardMaterial({ roughness: .85 });
  mat.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, uni);
    sh.vertexShader = "uniform float uTime; uniform vec2 uBend;\n" + sh.vertexShader.replace("#include <project_vertex>", `
      vec4 mvPosition = vec4(transformed, 1.0);
      #ifdef USE_INSTANCING
        mvPosition = instanceMatrix * mvPosition;
      #endif
      float k = uv.y * uv.y, ph = dot(mvPosition.xz, vec2(.31, .27));
      vec2 sway = uBend * (0.75 + 0.25 * sin(uTime * 2.3 + ph)) + 0.02 * vec2(sin(uTime * 3.1 + ph * 2.0), cos(uTime * 2.7 + ph));
      mvPosition.xz += sway * k * 0.12;
      mvPosition = modelViewMatrix * mvPosition;
      gl_Position = projectionMatrix * mvPosition;`);
    sh.fragmentShader = sh.fragmentShader.replace("#include <color_fragment>", "#include <color_fragment>\n diffuseColor.rgb *= mix(0.55, 1.0, vUv.y);");
  };
  mat.defines = { USE_UV: "" };
  const mesh = new THREE.InstancedMesh(blade, mat, n);
  (mesh.instanceMatrix.array as Float32Array).set(mats.subarray(0, n * 16));
  mesh.instanceColor = new THREE.InstancedBufferAttribute(cols.slice(0, n * 3), 3);
  mesh.receiveShadow = true; mesh.frustumCulled = false; sc.add(mesh);
  return {
    // wind: direction the wind blows toward (world east/north) and yard-level gust in mph
    update(t: number, toE: number, toN: number, gustMph: number) { uni.uTime.value = t; const s = clamp(gustMph / 25, 0, 2.2); uni.uBend.value.set(toE * s, -toN * s); },
  };
}

// ---------- LiDAR-sized procedural trees ----------
// ez-tree species presets, instanced. Each LiDAR tree gets the variant that fits its shape, scaled to its
// measured height and crown width; leaves take the canopy colour from the imagery. Sway follows the gust.
export async function buildTrees(M: Model, sc: THREE.Scene) {
  const { Tree } = await import("@dgreenheck/ez-tree");
  const { D, terrH } = M;
  const names = ["oak_large", "ash_large", "oak_medium", "aspen_medium", "pine_medium", "bush_1"];
  const variants = names.map((name, i) => {
    const t = new Tree(); t.loadPreset(name); t.options.seed = 1000 + i * 77;
    t.options.leaves.size *= 1.7;   // (ez-tree indexes leaves with Uint16: raising count overflows)
    t.options.leaves.tint = 0xffffff;   // full mid-October crowns; colour comes per tree
    t.generate();
    const bb = new THREE.Box3().setFromObject(t), size = bb.getSize(new THREE.Vector3());
    return { t, h: size.y, w: Math.max(size.x, size.z), y0: bb.min.y };
  });
  const pick = (tr: Tree, k: number) => tr.h < 4 ? 5 : tr.cw / tr.h < .42 ? 4 : [0, 1, 2, 3][k % 4];
  const byV = names.map(() => [] as { tr: Tree; k: number }[]);
  D.trees.forEach((tr: Tree, k: number) => byV[pick(tr, k)].push({ tr, k }));
  const d = new THREE.Object3D(), leaves: { mesh: THREE.InstancedMesh; items: { tr: Tree; k: number }[] }[] = [], mats: THREE.Material[] = [];
  byV.forEach((items, vi) => {
    if (!items.length) return;
    const v = variants[vi];
    const br = new THREE.InstancedMesh(v.t.branchesMesh.geometry, v.t.branchesMesh.material as THREE.Material, items.length);
    const lmat = v.t.leavesMesh.material as THREE.MeshPhongMaterial, orig = lmat.onBeforeCompile.bind(lmat);
    lmat.onBeforeCompile = (sh, r) => {   // ez-tree's leaf sway shader skips instanceMatrix; put it back
      orig(sh, r);
      sh.vertexShader = sh.vertexShader.replace("mvPosition = modelViewMatrix * mvPosition;", "#ifdef USE_INSTANCING\n mvPosition = instanceMatrix * mvPosition;\n#endif\n mvPosition = modelViewMatrix * mvPosition;");
    };
    lmat.emissive.setRGB(.05, .075, .025); lmat.emissiveMap = lmat.map; lmat.shininess = 6;   // light through thin leaves (Phong has no sky ambient)
    const lv = new THREE.InstancedMesh(v.t.leavesMesh.geometry, lmat, items.length);
    lv.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: lmat.map, alphaTest: lmat.alphaTest });   // leaf-shaped shadows
    items.forEach(({ tr, k }, i) => {
      const sxz = clamp(tr.cw / v.w, .25, 3), sy = tr.h / v.h;
      d.position.set(tr.x, terrH(tr.x, tr.y) - v.y0 * sy - .2, -tr.y); d.rotation.set(0, (k * 2.399) % 6.283, 0); d.scale.set(sxz, sy, sxz); d.updateMatrix();
      br.setMatrixAt(i, d.matrix); lv.setMatrixAt(i, d.matrix);
    });
    for (const m of [br, lv]) { m.castShadow = m.receiveShadow = true; sc.add(m); }
    leaves.push({ mesh: lv, items }); mats.push(lv.material as THREE.Material);
  });
  const base = (tr: Tree) => { const c = tr.color ?? 0x4a6a3a; const k = new THREE.Color().setRGB(((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255, THREE.SRGBColorSpace); return k.multiplyScalar(3.2).lerp(new THREE.Color(.5, .62, .28), .45); };   // imagery canopy is shadowed: lift it
  const paint = (risk: boolean, warn: THREE.Color, bad: THREE.Color) => {
    const calm = new THREE.Color("#7f9a7c"), small = new THREE.Color("#556058");
    for (const { mesh, items } of leaves) {
      items.forEach(({ tr }, i) => {
        const p = tr.p_fail || 0;
        mesh.setColorAt(i, !risk ? base(tr) : tr.h < 8 ? small : p < .002 ? calm : p < .02 ? calm.clone().lerp(warn, (p - .002) / .018) : warn.clone().lerp(bad, clamp((p - .02) / .1, 0, 1)));
      });
      mesh.instanceColor!.needsUpdate = true;
    }
  };
  paint(false, new THREE.Color(), new THREE.Color());
  return {
    paint,
    update(t: number, gustMph: number) {
      const s = clamp(gustMph / 30, .08, 1.6) * .35;
      for (const m of mats) { const sh = m.userData.shader; if (sh) { sh.uniforms.uTime.value = t; sh.uniforms.uWindStrength.value.set(s, 0, s); sh.uniforms.uWindFrequency.value = .6 + gustMph / 40; } }
    },
  };
}

// ---------- storm atmosphere ----------
const NOISE = `
float h21(vec2 p) { vec3 q = fract(vec3(p.xyx) * .1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float vn(vec2 p) { vec2 i = floor(p), f = fract(p), u = f * f * (3. - 2. * f);
  return mix(mix(h21(i), h21(i + vec2(1., 0.)), u.x), mix(h21(i + vec2(0., 1.)), h21(i + vec2(1., 1.)), u.x), u.y); }
const mat2 ROT = mat2(.8, .6, -.6, .8);
float fbm3(vec2 p) { float s = 0., a = .5; for (int k = 0; k < 3; k++) { s += a * vn(p); p = ROT * p * 2.02 + 7.3; a *= .5; } return s / .875; }
float fbm5(vec2 p) { float s = 0., a = .5; for (int k = 0; k < 5; k++) { s += a * vn(p); p = ROT * p * 2.02 + 7.3; a *= .5; } return s / .96875; }`;

// Sky dome: an overcast gradient with a drifting cloud deck drawn as a screen-space halftone (the brand's dot motif),
// darker and fuller as the hour gets wetter; lightning lights the clouds around the strike. Follows the camera.
export function buildSky(sc: THREE.Scene) {
  const uni = {
    uTop: { value: new THREE.Color() }, uHor: { value: new THREE.Color() }, uCloud: { value: new THREE.Color() }, uDrift: { value: new THREE.Vector2() },
    uTime: { value: 0 }, uDark: { value: .3 }, uCell: { value: 6 }, uFlash: { value: 0 }, uFlashDir: { value: new THREE.Vector3(0, .3, -1).normalize() },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms: uni, side: THREE.BackSide, depthWrite: false, depthTest: false, fog: false,
    vertexShader: `varying vec3 vDir;
      void main() { vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }`,
    fragmentShader: `uniform vec3 uTop, uHor, uCloud, uFlashDir; uniform vec2 uDrift; uniform float uTime, uDark, uCell, uFlash;
      varying vec3 vDir;${NOISE}
      void main() {
        vec3 d = normalize(vDir);
        vec3 col = mix(uHor, uTop, smoothstep(-.03, .6, d.y));
        vec2 uv = d.xz / (d.y + .22) * .75 + uDrift * uTime;                  // cloud deck seen in perspective
        vec2 w = vec2(fbm3(uv * .8 + uTime * .004), fbm3(uv * .8 + vec2(4.7, 1.9)));
        float n = fbm5(uv * 1.35 + w * 1.3);
        float cov = mix(.6, .36, uDark);
        float dens = smoothstep(cov - .06, cov + .24, n) * smoothstep(-.01, .14, d.y);
        vec2 f = fract(gl_FragCoord.xy / uCell) - .5;
        float r = .58 * sqrt(dens), aa = 1.2 / uCell;
        float dotm = 1. - smoothstep(r - aa, r + aa, length(f));
        float fl = uFlash * (.2 + 2.6 * pow(max(dot(d, uFlashDir), 0.), 10.));
        col *= 1. + .45 * uFlash;
        col += dotm * uCloud * (.4 + .6 * dens) * (1. + 5. * fl);
        gl_FragColor = vec4(col, 1.);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), mat);
  mesh.scale.setScalar(1800); mesh.renderOrder = -10; mesh.frustumCulled = false; sc.add(mesh);
  return { mesh, uni, dispose() { sc.remove(mesh); mesh.geometry.dispose(); mat.dispose(); } };
}

// Image-based light for an overcast day: a soft vertical gradient (no sun disc), prefiltered once.
export function overcastEnv(R: THREE.WebGLRenderer) {
  const g = new THREE.SphereGeometry(1, 32, 16), c = new Float32Array(g.attributes.position.count * 3), p = g.attributes.position;
  const top = new THREE.Color(.55, .6, .66), hor = new THREE.Color(.36, .39, .42), gnd = new THREE.Color(.09, .09, .08), k = new THREE.Color();
  for (let i = 0; i < p.count; i++) { const y = p.getY(i); (y > 0 ? k.lerpColors(hor, top, y) : k.lerpColors(hor, gnd, Math.min(1, -y * 3))).toArray(c, i * 3); }
  g.setAttribute("color", new THREE.BufferAttribute(c, 3));
  const s = new THREE.Scene(), m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })); m.scale.setScalar(50); s.add(m);
  const pm = new THREE.PMREMGenerator(R), tex = pm.fromScene(s, .02).texture;
  pm.dispose(); g.dispose(); (m.material as THREE.Material).dispose();
  return tex;
}

// Rain: streaks advected on the GPU (zero per-frame CPU), wrapped in a box that follows the view, falling at 9 m/s and
// drifting with the hour's wind so they slant the way the storm blows; call step(dt) each frame. Fades near the camera and at the box edge.
export function buildRain(sc: THREE.Scene, N = 14000) {
  const seed = new Float32Array(N * 6), aux = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const x = Math.random(), y = Math.random(), z = Math.random(), w = Math.random();
    seed.set([x, y, z, x, y, z], i * 6); aux.set([w, 0, w, 1], i * 4);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(seed, 3)); g.setAttribute("aux", new THREE.BufferAttribute(aux, 2));
  const uni = { uOff: { value: new THREE.Vector3() }, uBox: { value: new THREE.Vector3(240, 90, 240) }, uCenter: { value: new THREE.Vector3() }, uVel: { value: new THREE.Vector3(0, -9, 0) },
    uColor: { value: new THREE.Color(.62, .7, .78) }, uOpacity: { value: .3 } };
  const mat = new THREE.ShaderMaterial({
    uniforms: uni, transparent: true, depthWrite: false, fog: false,
    vertexShader: `attribute vec2 aux; uniform vec3 uOff, uBox, uCenter, uVel; varying float vA;
      void main() {
        vec3 lo = uCenter - .5 * uBox;
        vec3 p = position * uBox + uOff * (.85 + .3 * aux.x);                  // uOff: wind-drift integrated on the CPU, so changing wind never jumps
        p = lo + mod(p - lo, uBox);
        // tail: 50-80 ms of motion blur, stretched with distance so far-off rain still reads as streaks, not dots
        p -= uVel * aux.y * (.05 + .03 * aux.x) * max(1., distance(p, cameraPosition) / 30.);
        vec4 mv = modelViewMatrix * vec4(p, 1.);
        vA = (1. - .9 * aux.y) * smoothstep(1.5, 9., -mv.z) * (1. - smoothstep(.36, .5, length((p.xz - uCenter.xz) / uBox.xz)));
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; varying float vA;
      void main() { gl_FragColor = vec4(uColor, uOpacity * vA);
        #include <colorspace_fragment>
      }`,
  });
  const lines = new THREE.LineSegments(g, mat); lines.frustumCulled = false; lines.renderOrder = 6; lines.visible = false; sc.add(lines);
  return {
    lines, uni,
    step(dt: number) { uni.uOff.value.addScaledVector(uni.uVel.value, dt); },
    count(n: number) { n = Math.max(0, Math.min(N, Math.round(n))); g.setDrawRange(0, n * 2); lines.visible = n > 0; },
    dispose() { sc.remove(lines); g.dispose(); mat.dispose(); },
  };
}

// A lightning channel from cloud base to ground: midpoint-displaced main stroke plus a few forked branches, as
// line-segment pairs for LineSegmentsGeometry.
export function boltPath(a: THREE.Vector3, b: THREE.Vector3, rnd: () => number = Math.random) {
  const out: number[] = [];
  const seg = (p: THREE.Vector3, q: THREE.Vector3, depth: number, jag: number) => {
    if (!depth) { out.push(p.x, p.y, p.z, q.x, q.y, q.z); return; }
    const L = p.distanceTo(q), m = p.clone().lerp(q, .5 + (rnd() - .5) * .12);
    m.x += (rnd() - .5) * L * jag; m.z += (rnd() - .5) * L * jag; m.y += (rnd() - .5) * L * jag * .25;
    seg(p, m, depth - 1, jag); seg(m, q, depth - 1, jag);
  };
  seg(a, b, 6, .42);
  const main = out.length / 6, H = a.y - b.y;
  for (let k = 0; k < 3; k++) {
    const i = Math.floor((.12 + rnd() * .5) * main) * 6, s = new THREE.Vector3(out[i], out[i + 1], out[i + 2]), drop = H * (.18 + rnd() * .25);
    const ang = rnd() * 6.283, e = s.clone().add(new THREE.Vector3(Math.cos(ang) * drop * .7, -drop, Math.sin(ang) * drop * .7));
    seg(s, e, 4, .5);
  }
  return out;
}

// Terrain edge dissolves into a world-space halftone: dots shrink toward the border, so the model reads as a cut-out
// of the world printed in the brand's dot screen rather than a slab with walls.
export function halftoneEdge(mat: THREE.Material, half: number, band = 34, cell = 2.4) {
  mat.onBeforeCompile = sh => {
    sh.uniforms.uHalf = { value: half }; sh.uniforms.uBand = { value: band }; sh.uniforms.uCellW = { value: cell };
    sh.vertexShader = "varying vec2 vXZ;\n" + sh.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\nvXZ = (modelMatrix * vec4(transformed, 1.0)).xz;");
    sh.fragmentShader = "varying vec2 vXZ;\nuniform float uHalf, uBand, uCellW;\n" + sh.fragmentShader.replace("#include <clipping_planes_fragment>", `#include <clipping_planes_fragment>
      float edgeD = uHalf - max(abs(vXZ.x), abs(vXZ.y));
      if (edgeD < uBand) { vec2 f = fract(vXZ / uCellW) - .5; if (length(f) > .74 * pow(max(edgeD, 0.) / uBand, .75)) discard; }`);
  };
}
