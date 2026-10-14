// The loading screen's hero: 1,400 dots that morph between one dot-drawn scene per engine stage. Dots are the brand motif
// (halftone clouds, LiDAR points, the 1,000 storms), so the whole wait reads as one living thing.
// CPU: every dot is a damped spring pulled toward its slot in the current scene; scenes animate their slots. On a stage change
// the dots nearest the scene are ranked left to right onto the new scene's slots (slots ranked the same way), so one shape
// flows into the next with a slight counter-clockwise swirl instead of exploding. Unused dots drift around as faint dust.
// GPU: one drawArrays of round points, each streaked along its velocity (motion blur), with a soft halo on lit accent dots.
// The pointer pushes dots aside (read from window, so the canvas can sit under the UI with pointer-events: none).
// Reduced motion: no loop; one still frame per stage, resize and scroll.
// ponytail: no WebGL -> a no-op controller and a blank canvas (same as halftone.ts); the typography still carries the screen.

export const MODELS = ["HRRR", "NBM", "GFS", "ECMWF", "ICON", "GEM", "UKMO"];
export type DotMorph = { setStage(i: number): void; setMark(model: number): void; dispose(): void };

const N = 1400, F = 10, TAU = Math.PI * 2;          // dots; floats per dot in the vertex buffer

// Scene units: the scene box is [-1, 1] on its short side, y down; r is a dot radius in the same units.
// Palette index c: 0 ink, 1 accent, 2 warn, 3 muted. g: halo 0..1. Each frame writes x, y, r, a for every slot (c, g default 0).
type Out = { x: Float32Array; y: Float32Array; r: Float32Array; a: Float32Array; c: Uint8Array; g: Float32Array };
type Scene = { n: number; frame(t: number, T: number, o: Out, mark: number): void };   // t: s since the scene began, T: clock
type Pt = [number, number];

const rnd = (i: number) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const fract = (x: number) => x - Math.floor(x);
const ss = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function seg(a: Pt, b: Pt, step: number, out: Pt[] = []) {
  const n = Math.max(1, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
  for (let i = 0; i < n; i++) out.push([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n]);
  return out;
}
function path(ps: Pt[], step: number, closed: boolean, out: Pt[] = []) {
  for (let i = 0; i < ps.length - 1; i++) seg(ps[i], ps[i + 1], step, out);
  if (closed) seg(ps[ps.length - 1], ps[0], step, out); else out.push(ps[ps.length - 1]);
  return out;
}
function hex(x0: number, y0: number, x1: number, y1: number, s: number, keep: (x: number, y: number) => boolean) {
  const out: Pt[] = [];
  for (let r = 0, y = y0; y <= y1; r++, y += s * .866) for (let x = x0 + (r & 1) * s / 2; x <= x1; x += s) if (keep(x, y)) out.push([x, y]);
  return out;
}

// 5-row dot-matrix glyphs, just the letters the model names need
const GLYPH: Record<string, string> = {
  H: "101/101/111/101/101", R: "110/101/110/101/101", N: "1001/1101/1011/1001/1001", B: "110/101/110/101/110",
  M: "10001/11011/10101/10001/10001", G: "011/100/101/101/011", F: "111/100/110/100/100", S: "011/100/010/001/110",
  E: "111/100/110/100/111", C: "011/100/100/100/011", W: "10001/10001/10101/11011/10001", I: "111/010/010/010/111",
  O: "010/101/101/101/010", K: "101/101/110/101/101", U: "101/101/101/101/111",
};
function text(s: string, cx: number, y: number, p: number) {
  const gs = [...s].map(ch => GLYPH[ch].split("/")), out: Pt[] = [];
  let x = cx - (gs.reduce((w, g) => w + g[0].length + 1, -1) - 1) * p / 2;
  for (const g of gs) {
    g.forEach((row, r) => [...row].forEach((b, c) => { if (b === "1") out.push([x + c * p, y + r * p]); }));
    x += (g[0].length + 1) * p;
  }
  return out;
}

// 0 Warming up: radar rings with a sweeping beam; dots glow as it passes and blips light up
function radar(): Scene {
  const ang: number[] = [], rad: number[] = [], kind: number[] = [];
  const add = (a: number, r: number, k: number) => { ang.push(a); rad.push(r); kind.push(k); };
  for (const r of [.28, .54, .8, 1.04]) { const n = Math.round(TAU * r / .034); for (let i = 0; i < n; i++) add(i / n * TAU, r, 0); }
  for (let s = .07; s <= 1.04; s += .052) for (let q = 0; q < 4; q++) add(q * TAU / 4, s, 1);
  for (let i = 1; i <= 32; i++) add(0, i / 32 * 1.04, 2);
  for (let i = 0; i < 7; i++) add(rnd(i + 40) * TAU, .2 + .75 * rnd(i + 50), 3);
  const n = ang.length;
  return { n, frame(_t, T, o) {
    const sw = T * 1.5;
    for (let i = 0; i < n; i++) {
      const k = kind[i], r = rad[i], a = k === 2 ? sw : ang[i];
      o.x[i] = Math.cos(a) * r; o.y[i] = Math.sin(a) * r;
      const d = ((sw - ang[i]) % TAU + TAU) % TAU, glow = Math.exp(-d * 1.8);
      if (k === 0) { o.a[i] = .14 + .76 * glow; o.r[i] = .007 + .005 * glow; }
      else if (k === 1) { o.a[i] = .08 + .3 * glow; o.r[i] = .005; }
      else if (k === 2) { o.a[i] = .95 - .45 * r; o.r[i] = .009; o.c[i] = 1; o.g[i] = .45; }
      else { const b = Math.exp(-d * .7); o.a[i] = .04 + .96 * b; o.r[i] = .011 + .01 * b; o.c[i] = 1; o.g[i] = b; }
    }
  } };
}

// 1 Found your address: a halftone-shaded map pin over a perspective street grid, ripples spreading from its tip
function pin(): Scene {
  const cy = -.34, R = .36, tip = .38, tan = cy + R / 2;                 // tip half-angle 30 deg: R / (tip - cy) = .5
  const body = hex(-.4, -.72, .4, .4, .045, (x, y) => {
    const d = x * x + (y - cy) ** 2;
    if (d < .018) return false;                                          // the hole
    if (d < R * R) return true;
    const v = tip - y; return y > tan && v > .866 * Math.hypot(x, v);
  });
  const ground: Pt[] = [];
  for (let j = 0; j < 6; j++) { const f = j / 5, y = .3 + .66 * f ** 1.4, w = .55 + .6 * f; for (let x = -w; x <= w + 1e-6; x += .05) ground.push([x, y]); }
  for (let u = -1; u <= 1.001; u += .25) seg([u * .55, .3], [u * 1.15, .96], .06, ground);
  const RIP = 3, RN = 60, nb = body.length, ng = ground.length;
  return { n: nb + ng + RIP * RN, frame(_t, T, o) {
    const bob = .018 * Math.sin(T * 1.6) - .02;
    for (let i = 0; i < nb; i++) {
      const [x, y] = body[i], s = ss(-1.1, 1, -(x * .6 + (y - cy) * .8) / R);
      o.x[i] = x; o.y[i] = y + bob; o.r[i] = .006 + .011 * s; o.a[i] = .5 + .5 * s; o.c[i] = 1;
    }
    for (let i = 0; i < ng; i++) { const j = nb + i; o.x[j] = ground[i][0]; o.y[j] = ground[i][1]; o.r[j] = .0045; o.a[j] = .16; }
    for (let k = 0; k < RIP; k++) {
      const ph = fract(T * .32 + k / RIP), rx = .1 + .95 * ph, fade = .7 * (1 - ph) ** 1.5 * ss(0, .06, ph);
      for (let i = 0; i < RN; i++) {
        const j = nb + ng + k * RN + i, a = i / RN * TAU;
        o.x[j] = Math.cos(a) * rx; o.y[j] = tip + .02 + Math.sin(a) * rx * .3; o.r[j] = .007; o.a[j] = fade;
      }
    }
  } };
}

// 2 Weighing seven weather models: LED-meter bars with dot-matrix names; the most trusted model (once known) lights up
function bars(): Scene {
  const P: number[][] = [];                                              // x, y, bar, row, kind (0 meter, 1 name, 2 baseline)
  MODELS.forEach((m, b) => {
    const cx = -.84 + b * .28;
    for (let r = 0; r < 20; r++) for (let c = -1; c <= 1; c++) P.push([cx + c * .04, .4 - r * .041, b, r, 0]);
    for (const [x, y] of text(m, cx, .52, .0115)) P.push([x, y, b, 0, 1]);
  });
  for (let x = -1; x <= 1.001; x += .04) P.push([x, .465, -1, 0, 2]);
  const h = new Float32Array(MODELS.length);
  return { n: P.length, frame(_t, T, o, mark) {
    for (let b = 0; b < h.length; b++)
      h[b] = b === mark ? .97 : (.22 + .62 * (.5 + .5 * Math.sin(T * (1.1 + .17 * b) + b * 1.9) * Math.sin(T * .37 + b))) * (mark >= 0 ? .8 : 1);
    for (let i = 0; i < P.length; i++) {
      const [x, y, b, r, k] = P[i], hot = b >= 0 && b === mark;
      o.x[i] = x; o.y[i] = y;
      if (k === 0) {
        const on = h[b] * 20, lit = r < on, top = lit && r >= on - 1;
        o.a[i] = lit ? (top ? 1 : .78) : .07; o.r[i] = lit ? (top ? .014 : .011) : .006;
        if (hot) { o.c[i] = 1; o.g[i] = top ? .8 : 0; }
      } else if (k === 1) { o.a[i] = hot ? 1 : .5; o.r[i] = .0047; if (hot) o.c[i] = 1; }
      else { o.a[i] = .18; o.r[i] = .004; }
    }
  } };
}

// 3 Flood zone, elevation and soil: three soil strata under falling rain; the wetting front soaks down through the topsoil
function soil(): Scene {
  const b1 = (x: number) => .3 + .04 * Math.sin(2.3 * x + 1), b2 = (x: number) => .62 + .05 * Math.sin(1.7 * x + 2.2);
  const P: number[][] = [];                                              // x, y, layer (3 = surface), jitter
  hex(-1, .05, 1, .97, .05, () => true).forEach(([x, y], i) => {
    const L = y < b1(x) ? 0 : y < b2(x) ? 1 : 2;
    if (L < 2 || rnd(i) < .42) P.push([x, y, L, rnd(i + 3)]);
  });
  for (let x = -1; x <= 1.001; x += .028) P.push([x, -.005 + .012 * Math.sin(9 * x), 3, 0]);
  const RS = [.0085, .011, .016], AS = [.42, .28, .2], ns = P.length, RAIN = 150;
  return { n: ns + RAIN, frame(t, T, o) {
    const wet = .32 * (1 - Math.exp(-t / 5));
    for (let i = 0; i < ns; i++) {
      const [x, y, L, j] = P[i];
      o.x[i] = x; o.y[i] = y;
      if (L === 3) { o.a[i] = .55; o.r[i] = .006; continue; }
      const w = L === 0 ? ss(wet + .03, wet - .03, y) : 0;
      o.r[i] = RS[L] + .003 * j + .003 * w; o.a[i] = AS[L] + .45 * w;
    }
    for (let k = 0; k < RAIN; k++) {
      const i = ns + k, ph = fract(T * (.8 + .4 * rnd(k + 11)) + rnd(k + 17));
      o.x[i] = -1.1 + 2.1 * rnd(k + 23) + .2 * ph; o.y[i] = -1.05 + 1.05 * ph; o.r[i] = .006;
      o.a[i] = .6 * ss(0, .12, ph) * ss(1, .9, ph);
    }
  } };
}

// 4 Finding your house: building footprints seen from above; a targeting bracket checks a few, then locks onto yours
function house(): Scene {
  // footprints: [cx, cy, w, h]; index 0 is yours, at the centre
  const B: number[][] = [[0, 0, .34, .26], [-.62, -.5, .28, .22], [0, -.55, .3, .2], [.62, -.48, .26, .24], [-.66, .05, .24, .3],
    [.64, .06, .3, .26], [-.6, .58, .3, .22], [.02, .6, .26, .22], [.62, .6, .28, .2]];
  const P: number[][] = [];                                              // x, y, building
  B.forEach(([cx, cy, w, h], k) => hex(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, .042, () => true).forEach(([x, y]) => P.push([x, y, k])));
  const road: Pt[] = []; seg([-1, -.25], [1.001, -.25], .05, road); seg([-1, .32], [1.001, .32], .05, road); seg([-.32, -1], [-.32, 1.001], .05, road); seg([.32, -1], [.32, 1.001], .05, road);
  const ARM = 10, nb = P.length, nr = road.length, HOPS = [3, 6, 4, 0];  // candidates checked, ending on yours
  const at = (k: number) => B[k], pad = .07;
  return { n: nb + nr + 4 * 2 * ARM, frame(t, T, o) {
    const c = t % 8, hop = Math.min(HOPS.length - 1, Math.floor(c / 1.1)), k0 = HOPS[Math.max(0, hop - 1)], k1 = HOPS[hop];
    const e = hop === 0 ? 1 : ss(hop * 1.1, hop * 1.1 + .45, c), lock = ss(3.7, 4.1, c) * ss(8, 7.6, c);
    const [ax, ay, aw, ah] = at(k0), [bx, by, bw, bh] = at(k1), L = (u: number, v: number) => u + (v - u) * e;
    const cx = L(ax, bx), cy = L(ay, by), hw = L(aw, bw) / 2 + pad * (1 - .5 * lock), hh = L(ah, bh) / 2 + pad * (1 - .5 * lock);
    for (let i = 0; i < nb; i++) {
      const [x, y, k] = P[i], mine = k === 0, look = k === k1 && e > .9 && !lock ? 1 : 0;
      o.x[i] = x; o.y[i] = y;
      const pulse = mine ? lock * (.5 + .5 * Math.sin(T * 4 - Math.hypot(x, y) * 18)) : 0;
      o.r[i] = .008 + .004 * look + .006 * pulse; o.a[i] = .22 + .35 * look + (mine ? .6 * lock : 0);
      if (mine && lock > 0) { o.c[i] = lock; o.g[i] = .4 * pulse; }
    }
    for (let i = 0; i < nr; i++) { const j = nb + i; o.x[j] = road[i][0]; o.y[j] = road[i][1]; o.r[j] = .004; o.a[j] = .12; }
    // four corner brackets, each two arms of ARM dots
    let j = nb + nr;
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) for (const axis of [0, 1]) for (let q = 0; q < ARM; q++) {
      const len = q / ARM * .11;
      o.x[j] = cx + sx * hw - (axis ? 0 : sx * len); o.y[j] = cy + sy * hh - (axis ? sy * len : 0);
      o.r[j] = .008; o.a[j] = .95; o.c[j] = 1; o.g[j] = lock * .6; j++;
    }
  } };
}

// 5 Satellite imagery and the neighbours: hip roofs seen from above, imaged by a passing satellite's swath; yours gets outlined
function satellite(): Scene {
  const P: number[][] = [];                                              // x, y, shade, kind (0 roof, 1 tree, 2 your outline, 3 satellite)
  for (let gy = -1; gy <= 1; gy++) for (let gx = -1; gx <= 1; gx++) {
    const me = !gx && !gy, k = (gy + 1) * 3 + gx + 1;
    const cx = gx * .64 + (me ? 0 : (rnd(k) - .5) * .1), cy = gy * .56 + (me ? 0 : (rnd(k + 9) - .5) * .08);
    const w = me ? .44 : .3 + .12 * rnd(k + 3), d = me ? .32 : .22 + .08 * rnd(k + 5), rot = me ? 0 : (rnd(k + 7) - .5) * .3;
    const cs = Math.cos(rot), sn = Math.sin(rot);
    for (const [u, v] of hex(-w / 2, -d / 2, w / 2, d / 2, .036, () => true)) {
      const du = w / 2 - Math.abs(u), dv = d / 2 - Math.abs(v);     // hip roof: the nearest eave picks the facet
      P.push([cx + u * cs - v * sn, cy + u * sn + v * cs, dv < du ? (v < 0 ? .95 : .3) : (u < 0 ? .7 : .45), 0]);
    }
    if (me) for (const [u, v] of path([[-.25, -.19], [.25, -.19], [.25, .19], [-.25, .19]], .03, true)) P.push([u, v, 1, 2]);
  }
  for (let k = 0; k < 10; k++) {
    if (rnd(k + 80) < .2) continue;
    const tx = (k % 5 - 2) * .32 + (rnd(k + 60) - .5) * .08, ty = (k < 5 ? -.28 : .28) + (rnd(k + 70) - .5) * .06, rr = .05 + .03 * rnd(k + 90);
    for (let i = 0; i < 13; i++) {
      const r = rr * Math.sqrt((i + .5) / 13), a = i * 2.39996;
      P.push([tx + Math.cos(a) * r, ty + Math.sin(a) * r, .5 - .5 * (Math.cos(a) * .6 + Math.sin(a) * .8) * r / rr, 1]);
    }
  }
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) P.push([a * .018, b * .018, 1, 3]);
  for (let a = 0; a < 5; a++) for (let b = 0; b < 2; b++) for (const s of [-1, 1]) P.push([s * (.045 + a * .018), (b - .5) * .018, 1, 3]);
  return { n: P.length, frame(t, _T, o) {
    const sx = -1.35 + 2.7 * fract(t / 5);
    for (let i = 0; i < P.length; i++) {
      const [x, y, s, k] = P[i];
      if (k === 3) { o.x[i] = sx + x; o.y[i] = -.95 + y; o.a[i] = .9; o.r[i] = .0075; continue; }
      o.x[i] = x; o.y[i] = y;
      const band = Math.exp(-(((x - sx) / .12) ** 2)), seen = x < sx ? 1 : .35;
      if (k === 2) { const on = t > 2.5; o.a[i] = on ? .95 : .06; o.r[i] = .0075; o.c[i] = 1; o.g[i] = on ? .35 : 0; }
      else if (k === 1) { o.a[i] = (.18 + .3 * s) * (.4 + .6 * seen) + .3 * band; o.r[i] = .006 + .003 * s; }
      else { o.a[i] = (.22 + .55 * s) * (.4 + .6 * seen) + .35 * band; o.r[i] = .004 + .011 * s; }
    }
  } };
}

// 6 Measuring your trees with LiDAR: a rotating point-cloud tree, a scan plane lighting the points it hits, a height ruler
function tree(): Scene {
  const P: number[][] = [];                                              // x, y, z, kind (0 crown, 1 wood, 2 ground, 3 scan line, 4 ruler)
  const blobs = [[0, -.36, 0, .42, .34], [-.24, -.14, .06, .3, .25], [.25, -.12, -.06, .31, .25]];
  for (let i = 0; i < 560; i++) {
    const [bx, by, bz, rh, rv] = blobs[i % 4 < 2 ? 0 : i % 4 - 1];
    const u = 2 * rnd(i * 3) - 1, ph = TAU * rnd(i * 3 + 1), s = Math.sqrt(1 - u * u), q = 1 - .35 * rnd(i * 3 + 2) ** 2;
    P.push([bx + s * Math.cos(ph) * rh * q, by + u * rv * q, bz + s * Math.sin(ph) * rh * q, 0]);
  }
  for (let i = 0; i < 60; i++) { const a = rnd(i + 700) * TAU; P.push([Math.cos(a) * .045, .06 + .5 * i / 60, Math.sin(a) * .045, 1]); }
  for (const [ex, ey, ez] of [[-.24, -.2, .06], [.25, -.18, -.06], [0, -.4, 0], [-.1, -.3, -.15]])
    for (let i = 1; i <= 10; i++) { const f = i / 10; P.push([ex * f, .1 + (ey - .1) * f, ez * f, 1]); }
  for (let k = 0; k < 120; k++) { const r = .75 * Math.sqrt((k + .5) / 120), a = k * 2.39996; P.push([Math.cos(a) * r, .56, Math.sin(a) * r, 2]); }
  for (let i = 0; i < 40; i++) P.push([-.95 + 1.9 * i / 39, 0, 0, 3]);
  for (let i = 0; i <= 26; i++) P.push([.88, .56 - 1.28 * i / 26, 0, 4]);
  for (let i = -2; i <= 2; i++) { P.push([.88 + i * .02, .56, 0, 4]); P.push([.88 + i * .02, -.72, 0, 4]); }
  return { n: P.length, frame(t, T, o) {
    const yaw = T * .35, cs = Math.cos(yaw), sn = Math.sin(yaw), ys = -.78 + 1.4 * fract(t / 3.2), first = t < 3.2;
    for (let i = 0; i < P.length; i++) {
      const [x, y, z, k] = P[i];
      if (k === 3) { o.x[i] = x; o.y[i] = ys; o.a[i] = .22; o.r[i] = .005; o.c[i] = 1; continue; }
      if (k === 4) { o.x[i] = x; o.y[i] = y; o.a[i] = .3; o.r[i] = .005; continue; }
      const rx = x * cs - z * sn, rz = x * sn + z * cs, f = 2.6 / (2.6 + rz), near = ss(.6, -.6, rz);
      o.x[i] = rx * f; o.y[i] = y * f;
      const hit = k < 2 && Math.abs(y - ys) < .035, scanned = k === 2 || !first || y < ys;
      if (hit) { o.a[i] = 1; o.r[i] = .012 * f; o.c[i] = 1; o.g[i] = .9; }
      else { o.a[i] = (k === 0 ? .7 : k === 1 ? .6 : .22) * (.45 + .55 * near) * (scanned ? 1 : .25); o.r[i] = (k === 2 ? .005 : .0075) * f; }
    }
  } };
}

// 7 Simulating the wind: streamlines squeeze over the roof (and speed up there), with eddies tumbling in the lee
function flow(): Scene {
  const G = .6, LID = 1.6, X0 = -1.25, X1 = 1.25, M = 256;
  const bump = (x: number) => .66 * Math.exp(-((x / (x < 0 ? .36 : .62)) ** 2));
  const cum = new Float32Array(M + 1), inv = new Float32Array(M + 1);   // phase -> x, so dots run faster where lines squeeze
  for (let i = 1; i <= M; i++) cum[i] = cum[i - 1] + 1 - bump(X0 + (X1 - X0) * (i - .5) / M) / LID;
  for (let i = 0, j = 0; i <= M; i++) {
    const want = cum[M] * i / M;
    while (j < M - 1 && cum[j + 1] < want) j++;
    inv[i] = X0 + (X1 - X0) * (j + Math.min(1, (want - cum[j]) / (cum[j + 1] - cum[j]))) / M;
  }
  const xAt = (p: number) => { const f = p * M, i = Math.min(M - 1, Math.floor(f)); return inv[i] + (inv[i + 1] - inv[i]) * (f - i); };
  const LINES = 11, PER = 54;
  const hs = hex(-.4, -.02, .4, .6, .045, (x, y) => y > .25 ? Math.abs(x) < .31 : Math.abs(x) < .4 * (y + .02) / .27 - .03);
  const edge = path([[-.31, .6], [-.31, .25], [-.37, .25], [0, -.005], [.37, .25], [.31, .25], [.31, .6]], .03, false);
  const ground = seg([-1.2, .615], [1.2, .615], .045), nh = hs.length, ng = ground.length, nf = LINES * PER, ne = edge.length;
  return { n: nf + nh + ng + 20 + ne, frame(_t, T, o) {
    for (let k = 0; k < LINES; k++) {
      const g = .08 + k * .13;
      for (let i = 0; i < PER; i++) {
        const j = k * PER + i, p = fract(i / PER + T * .1 + rnd(k + 300)), x = xAt(p), b = bump(x), sp = 1 / (1 - b / LID);
        const wake = x > .3 && g < .55 ? ss(.3, .55, x) * ss(1.25, .8, x) * (1 - g / .55) : 0;
        o.x[j] = x; o.y[j] = G - (b + g * (1 - b / LID)) + .035 * wake * Math.sin(T * 4.2 + p * 60 + k);
        o.a[j] = (.3 + .7 * (sp - 1)) * (1 - .45 * g / 1.4) * ss(0, .06, p) * ss(1, .94, p) + .12;
        o.r[j] = .006 + .006 * (sp - 1);
      }
    }
    for (let i = 0; i < nh; i++) { const j = nf + i, [x, y] = hs[i]; o.x[j] = x; o.y[j] = y; o.r[j] = .006 + .004 * (.5 - .4 * x); o.a[j] = .3; o.c[j] = 3; }
    for (let i = 0; i < ng; i++) { const j = nf + nh + i; o.x[j] = ground[i][0]; o.y[j] = ground[i][1]; o.r[j] = .0045; o.a[j] = .22; }
    for (let i = 0; i < 20; i++) {
      const j = nf + nh + ng + i, e = i < 10 ? 0 : 1, a = (i % 10) / 10 * TAU + T * 2.2, r = e ? .055 : .07;
      o.x[j] = (e ? .86 : .58) + Math.cos(a) * r; o.y[j] = (e ? .53 : .49) + Math.sin(a) * r; o.r[j] = .006; o.a[j] = .35;
    }
    for (let i = 0; i < ne; i++) { const j = nf + nh + ng + 20 + i; o.x[j] = edge[i][0]; o.y[j] = edge[i][1]; o.r[j] = .0065; o.a[j] = .8; }
  } };
}

// 8 Playing out 1,000 storms: a 50 x 20 grid fills in as simulated storms land; some come up amber
function storms(): Scene {
  const n = 1000, rank = new Float32Array(n), amber = new Uint8Array(n);
  Array.from({ length: n }, (_, i) => i).sort((a, b) => rnd(a + 500) - rnd(b + 500)).forEach((d, r) => { rank[d] = r / n; });
  for (let i = 0; i < n; i++) amber[i] = rnd(i + 900) < .12 ? 1 : 0;
  return { n, frame(t, _T, o) {
    const fill = 1 - Math.exp(-t / 4.5);
    for (let i = 0; i < n; i++) {
      o.x[i] = -1 + (i % 50) * (2 / 49); o.y[i] = -.4 + Math.floor(i / 50) * (.8 / 19);
      if (rank[i] < fill) {
        const pop = Math.exp(-(t + 4.5 * Math.log(1 - rank[i])) * 6);   // fades from the moment this storm landed
        o.r[i] = .011 + .012 * pop; o.a[i] = Math.min(1, (amber[i] ? .95 : .7) + .3 * pop);
        if (amber[i]) { o.c[i] = 2; o.g[i] = .25 + .6 * pop; }
      } else { o.r[i] = .0055; o.a[i] = .08; }
    }
  } };
}

// 9 Building your 3D report: a wireframe house assembling edge by edge, turning over a dot ground grid
function model3d(): Scene {
  const w = .56, d = .4, g = .42, e = -.06, rg = -.42;                   // half width, half depth, ground, eaves, ridge
  type V3 = [number, number, number];
  const E: [V3, V3][] = [
    [[-w, g, -d], [w, g, -d]], [[w, g, -d], [w, g, d]], [[w, g, d], [-w, g, d]], [[-w, g, d], [-w, g, -d]],
    [[-w, g, -d], [-w, e, -d]], [[w, g, -d], [w, e, -d]], [[w, g, d], [w, e, d]], [[-w, g, d], [-w, e, d]],
    [[-w, e, -d], [w, e, -d]], [[w, e, -d], [w, e, d]], [[w, e, d], [-w, e, d]], [[-w, e, d], [-w, e, -d]],
    [[-w, rg, 0], [-w, e, -d]], [[-w, rg, 0], [-w, e, d]], [[w, rg, 0], [w, e, -d]], [[w, rg, 0], [w, e, d]], [[-w, rg, 0], [w, rg, 0]],
  ];
  const P: number[][] = [];                                              // x, y, z, edge, kind (0 edge, 1 corner, 2 grid)
  E.forEach(([a, b], k) => {
    const n = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / .035));
    for (let i = 1; i < n; i++) { const f = i / n; P.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, k, 0]); }
  });
  for (const x of [-w, w]) { for (const z of [-d, d]) for (const y of [g, e]) P.push([x, y, z, 0, 1]); P.push([x, rg, 0, 12, 1]); }
  for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) P.push([-1.05 + i * .21, g, -1.05 + j * .21, 0, 2]);
  const cp = Math.cos(.42), sp = Math.sin(.42);
  return { n: P.length, frame(t, T, o) {
    const yaw = T * .5 + .6, cy = Math.cos(yaw), sy = Math.sin(yaw);
    for (let i = 0; i < P.length; i++) {
      const [x, y, z, k, kind] = P[i];
      const rx = x * cy - z * sy, rz = x * sy + z * cy, py = y * cp - rz * sp, pz = y * sp + rz * cp, f = 3 / (3 + pz), near = ss(.8, -.8, pz);
      o.x[i] = rx * f * 1.1; o.y[i] = py * f + .08;
      const on = ss(k * .16, k * .16 + .35, t);
      if (kind === 2) { o.a[i] = .16 * (.4 + .6 * near); o.r[i] = .0045 * f; }
      else if (kind === 1) { o.a[i] = on; o.r[i] = .014 * f; o.c[i] = 1; o.g[i] = .7 * on; }
      else { o.a[i] = .85 * (.45 + .55 * near) * on; o.r[i] = .008 * f; }
    }
  } };
}

// Failed: the dots give up and settle into a dim heap
function pile(): Scene {
  const n = 700;
  return { n, frame(_t, _T, o) {
    for (let i = 0; i < n; i++) {
      const x = (rnd(i + 1) * 2 - 1) * .95;
      o.x[i] = x; o.y[i] = .62 - .34 * (1 - x * x / .9025) * rnd(i + 2); o.a[i] = .25; o.r[i] = .008; o.c[i] = 3;
    }
  } };
}

const VS = `
precision highp float;
attribute vec2 a_pos;
attribute vec2 a_vel;
attribute vec2 a_sz;
attribute vec4 a_col;
uniform vec2 u_res;
uniform float u_dpr, u_maxPt;
varying vec4 v_col;
varying vec4 v_shape;   // streak half-length, radius, sprite size (device px), halo
varying vec2 v_dir;
void main() {
  float sp = length(a_vel), rd = a_sz.x * u_dpr, a = a_col.a;
  float ln = min(sp * .02, 18.) * u_dpr;                          // motion streak
  if (rd < .65) { a *= rd * rd / .4225; rd = .65; }               // sub-pixel dot: keep it round, trade area for alpha
  a /= 1. + ln / (rd * 5.);                                       // a streak spreads the same light over more pixels
  float size = min(2. * (ln + rd + a_sz.y * (rd * 2. + 3. * u_dpr)) + 2., u_maxPt);
  v_dir = sp > 1. ? a_vel / sp : vec2(1., 0.);
  v_col = vec4(a_col.rgb, a);
  v_shape = vec4(ln, rd, size, a_sz.y);
  gl_PointSize = size;
  gl_Position = a < .004 ? vec4(2., 2., 2., 1.) : vec4(a_pos / u_res * vec2(2., -2.) + vec2(-1., 1.), 0., 1.);
}`;

const FS = `
precision mediump float;
varying vec4 v_col;
varying vec4 v_shape;
varying vec2 v_dir;
void main() {
  vec2 p = (gl_PointCoord - .5) * v_shape.z;
  float u = dot(p, v_dir), w = dot(p, vec2(-v_dir.y, v_dir.x));
  float d = length(vec2(u - clamp(u, -v_shape.x, v_shape.x), w));
  float rh = v_shape.y * 2. + 3.;
  float k = max(clamp(.5 - (d - v_shape.y), 0., 1.), v_shape.w * .4 * exp(-d * d / (rh * rh * .35))) * v_col.a;
  if (k < .003) discard;
  gl_FragColor = vec4(v_col.rgb * k, k);
}`;

const NOOP: DotMorph = { setStage() {}, setMark() {}, dispose() {} };

export function createDotMorph(canvas: HTMLCanvasElement, box: HTMLElement): DotMorph {
  try { return build(canvas, box) ?? NOOP; }
  catch (e) { console.warn("dot morph disabled:", e); return NOOP; }
}

function build(canvas: HTMLCanvasElement, box: HTMLElement): DotMorph | null {
  const ctx = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
    powerPreference: "low-power" });
  if (!ctx) return null;
  const gl: WebGLRenderingContext = ctx;
  const scenes = [radar(), pin(), bars(), soil(), house(), satellite(), tree(), flow(), storms(), model3d()], broken = pile();
  const mq = matchMedia("(prefers-reduced-motion: reduce)");

  // palette from the CSS tokens (ink, accent, warn, muted) as 0..1 sRGB
  const pal: number[] = [], css = getComputedStyle(document.documentElement), c2 = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  for (const k of ["--ink", "--accent", "--warn", "--muted"]) {
    if (!c2) { pal.push(1, 1, 1); continue; }
    c2.clearRect(0, 0, 1, 1); c2.fillStyle = css.getPropertyValue(k).trim() || "#fff"; c2.fillRect(0, 0, 1, 1);
    const d = c2.getImageData(0, 0, 1, 1).data; pal.push(d[0] / 255, d[1] / 255, d[2] / 255);
  }

  const f32 = () => new Float32Array(N);
  const o: Out = { x: f32(), y: f32(), r: f32(), a: f32(), c: new Uint8Array(N), g: f32() };
  const px = f32(), py = f32(), vx = f32(), vy = f32(), al = f32(), rad = f32(), gw = f32(), cr = f32(), cg = f32(), cb = f32();
  const lx = f32(), ly = f32(), delay = f32(), slotOf = new Int16Array(N).fill(-1), data = new Float32Array(N * F);
  const du = f32(), dv = f32(), dsp = f32(), da = f32();                // dust: start, row, speed, brightness
  for (let i = 0; i < N; i++) { du[i] = rnd(i * 3 + 1); dv[i] = rnd(i * 3 + 2); dsp[i] = .006 + .01 * rnd(i * 3 + 3); da[i] = .05 + .11 * rnd(i * 3 + 4); }

  let prog: WebGLProgram | null = null, buf: WebGLBuffer | null = null, uRes: WebGLUniformLocation | null = null;
  let uDpr: WebGLUniformLocation | null = null, scene = scenes[0], mark = -1, T = 0, t0 = 0, cx = 0, cy = 0, h = 0, W = 0, H = 0;
  let dpr = 1, raf = 0, last = 0, gone = false, ready = false, pending = true, still = mq.matches;
  let mx = 0, my = 0, mt = 0, mOn = false, ms = 0, mvx = 0, mvy = 0;   // pointer: position, inside, strength, velocity

  function init() {
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!; gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader");
      return s;
    };
    prog = gl.createProgram()!;
    const vs = sh(gl.VERTEX_SHADER, VS), fs = sh(gl.FRAGMENT_SHADER, FS);
    gl.attachShader(prog, vs); gl.attachShader(prog, fs);
    gl.linkProgram(prog); gl.deleteShader(vs); gl.deleteShader(fs);   // freed with the program
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) || "link");
    gl.useProgram(prog);
    uRes = gl.getUniformLocation(prog, "u_res"); uDpr = gl.getUniformLocation(prog, "u_dpr");
    gl.uniform1f(gl.getUniformLocation(prog, "u_maxPt"), gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1]);
    buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW);
    for (const [name, size, off] of [["a_pos", 2, 0], ["a_vel", 2, 2], ["a_sz", 2, 4], ["a_col", 4, 6]] as const) {
      const l = gl.getAttribLocation(prog, name);
      if (l >= 0) { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, F * 4, off * 4); }
    }
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.clearColor(0, 0, 0, 0);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  // The scene box is wherever the layout put it this frame (it scrolls on phones; the canvas is fixed full-screen).
  function measure() {
    const c = canvas.getBoundingClientRect(), b = box.getBoundingClientRect();
    W = c.width; H = c.height; cx = b.left - c.left + b.width / 2; cy = b.top - c.top + b.height / 2; h = Math.min(b.width, b.height) / 2;
    const d = Math.min(devicePixelRatio || 1, 2), bw = Math.round(W * d), bh = Math.round(H * d);
    if (canvas.width !== bw || canvas.height !== bh || d !== dpr) { dpr = d; canvas.width = bw; canvas.height = bh; gl.viewport(0, 0, bw, bh); }
    return W > 0 && H > 0 && h > 0;
  }

  // The scene's dots: the n dots nearest the box, ranked left to right onto the slots ranked the same way, released in a sweep.
  function assign() {
    const n = scene.n;
    o.c.fill(0); o.g.fill(0); scene.frame(T - t0, T, o, mark);
    const d2 = (i: number) => (px[i] - cx) ** 2 + (py[i] - cy) ** 2;
    const use = Array.from({ length: N }, (_, i) => i).sort((a, b) => d2(a) - d2(b)).slice(0, n).sort((a, b) => px[a] - px[b]);
    const slots = Array.from({ length: n }, (_, j) => j).sort((a, b) => o.x[a] - o.x[b]);
    slotOf.fill(-1);
    use.forEach((d, r) => { slotOf[d] = slots[r]; delay[d] = .3 * r / n + .1 * rnd(d + 77); });
  }

  function step(dt: number, snapAll: boolean) {
    if (!measure()) return false;
    if (!ready) {                                                        // first frame: every dot starts as dust, unlit
      for (let i = 0; i < N; i++) {
        px[i] = lx[i] = cx + (-1.75 + 3.5 * du[i]) * h; py[i] = ly[i] = cy + (-1.35 + 2.7 * dv[i]) * h;
        cr[i] = pal[0]; cg[i] = pal[1]; cb[i] = pal[2];
      }
      ready = true;
    }
    if (pending) { assign(); pending = false; }
    T += dt;
    const t = T - t0;
    o.c.fill(0); o.g.fill(0); scene.frame(t, T, o, mark);

    const set = ss(.9, 2, t), k = 90 + 330 * set, damp = 2 * (.68 + .2 * set) * Math.sqrt(k), swirl = .55 * k * (1 - ss(.15, 1, t));
    const snap2 = (.3 * h) ** 2, settled = t > 1.2;
    const fUp = 1 - Math.exp(-dt * 16), fDn = 1 - Math.exp(-dt * 4.5), fR = 1 - Math.exp(-dt * 9), fC = 1 - Math.exp(-dt * 5);
    ms += ((mOn && !still ? 1 : 0) - ms) * (1 - Math.exp(-dt * 8));
    const decay = Math.exp(-dt * 5); mvx *= decay; mvy *= decay;
    const PR = Math.max(70, h * .35), PR2 = PR * PR, push = k * 60 * ms, drag = k * .035 * ms;

    for (let i = 0; i < N; i++) {
      const j = slotOf[i];
      let X, Y, A, R, C, G;
      if (j >= 0) { X = o.x[j]; Y = o.y[j]; A = o.a[j]; R = o.r[j]; C = o.c[j]; G = o.g[j]; }
      else {                                                             // dust drifts east with the wind, fading at the edges
        const u = fract(du[i] + T * dsp[i]), v = dv[i];
        X = -1.75 + 3.5 * u; Y = -1.35 + 2.7 * v + .03 * Math.sin(T * .4 + i);
        A = da[i] * ss(0, .1, u) * ss(1, .9, u) * ss(0, .15, v) * ss(1, .85, v); R = .0045; C = 0; G = 0;
      }
      const tx = cx + X * h, ty = cy + Y * h, jx = tx - lx[i], jy = ty - ly[i];
      lx[i] = tx; ly[i] = ty;
      if (snapAll || (settled && jx * jx + jy * jy > snap2)) {          // a slot that wrapped around (rain, flow) jumps, unlit
        px[i] = tx; py[i] = ty; vx[i] = vy[i] = 0;
        if (!snapAll) al[i] = 0;
      }
      let ax, ay;
      const dx = tx - px[i], dy = ty - py[i];
      if (j >= 0 && t < delay[i]) { ax = -3 * vx[i]; ay = -3 * vy[i]; }
      else { ax = k * dx - damp * vx[i] - swirl * dy; ay = k * dy - damp * vy[i] + swirl * dx; }
      if (ms > .01) {
        const ex = px[i] - mx, ey = py[i] - my, e2 = ex * ex + ey * ey;
        if (e2 < PR2 && e2 > 1e-3) {
          const e = Math.sqrt(e2), f = (1 - e / PR) ** 2;
          ax += ex / e * f * push + mvx * f * drag; ay += ey / e * f * push + mvy * f * drag;
        }
      }
      vx[i] += ax * dt; vy[i] += ay * dt; px[i] += vx[i] * dt; py[i] += vy[i] * dt;

      const c3 = C * 3;
      if (snapAll) { al[i] = A; rad[i] = R; gw[i] = G; cr[i] = pal[c3]; cg[i] = pal[c3 + 1]; cb[i] = pal[c3 + 2]; }
      else {
        al[i] += (A - al[i]) * (A > al[i] ? fUp : fDn); gw[i] += (G - gw[i]) * (G > gw[i] ? fUp : fDn); rad[i] += (R - rad[i]) * fR;
        cr[i] += (pal[c3] - cr[i]) * fC; cg[i] += (pal[c3 + 1] - cg[i]) * fC; cb[i] += (pal[c3 + 2] - cb[i]) * fC;
      }
      const b = i * F;
      data[b] = px[i]; data[b + 1] = py[i]; data[b + 2] = vx[i]; data[b + 3] = vy[i]; data[b + 4] = rad[i] * h; data[b + 5] = gw[i];
      data[b + 6] = cr[i]; data[b + 7] = cg[i]; data[b + 8] = cb[i]; data[b + 9] = al[i];
    }
    return true;
  }

  function draw() {
    if (!prog || gl.isContextLost()) return;
    gl.uniform2f(uRes, W, H); gl.uniform1f(uDpr, dpr);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, N);
  }

  const loop = (now: number) => {
    raf = requestAnimationFrame(loop);
    // the settled spring (k 420, damping 36) is only stable for dt < 0.044 s with this integrator: cap at 1/30 so a slow
    // device runs the morph in slow motion instead of blowing every dot up to NaN
    const dt = Math.max(0, Math.min((now - last) / 1000, 1 / 30)); last = now;
    if (step(dt, false)) draw();
  };
  const start = () => { if (!raf && !still && !gone && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(loop); } };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };
  const frame1 = () => { if (still && !gone && prog) { T = t0 + 6; if (step(0, true)) draw(); } };   // reduced motion: settled frame

  const move = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, now = performance.now();
    if (mOn) {
      const s = Math.max(.008, (now - mt) / 1000), lim = (v: number) => Math.max(-3000, Math.min(3000, v));
      mvx = lim(mvx * .5 + .5 * (x - mx) / s); mvy = lim(mvy * .5 + .5 * (y - my) / s);
    }
    mx = x; my = y; mt = now; mOn = true;
  };
  const out = (e: PointerEvent) => { if (!e.relatedTarget) mOn = false; };
  const blur = () => { mOn = false; };
  const vis = () => document.hidden ? stop() : start();
  const motion = () => { still = mq.matches; if (still) { stop(); frame1(); } else start(); };
  const lost = (e: Event) => { e.preventDefault(); stop(); prog = null; };
  const restored = () => { init(); frame1(); start(); };

  init();   // first: if it throws, nothing below is attached, so the caller's no-op handle leaks nothing
  const ro = new ResizeObserver(frame1);
  ro.observe(canvas); ro.observe(box);
  addEventListener("scroll", frame1, { passive: true });
  addEventListener("pointermove", move, { passive: true });
  addEventListener("pointerout", out);
  addEventListener("blur", blur);
  document.addEventListener("visibilitychange", vis);
  mq.addEventListener("change", motion);
  canvas.addEventListener("webglcontextlost", lost);
  canvas.addEventListener("webglcontextrestored", restored);
  start();

  return {
    setStage(i) {
      const s = i < 0 ? broken : scenes[Math.min(i, scenes.length - 1)];
      if (s === scene && !pending) return;
      scene = s; t0 = T; pending = true;
      frame1();
    },
    setMark(k) { if (k !== mark) { mark = k; frame1(); } },
    dispose() {
      if (gone) return;
      gone = true; stop(); ro.disconnect();
      removeEventListener("scroll", frame1); removeEventListener("pointermove", move); removeEventListener("pointerout", out);
      removeEventListener("blur", blur);
      document.removeEventListener("visibilitychange", vis);
      mq.removeEventListener("change", motion);
      canvas.removeEventListener("webglcontextlost", lost);
      canvas.removeEventListener("webglcontextrestored", restored);
      // No WEBGL_lose_context: React StrictMode remounts on the same canvas, and a lost context can't be fetched again.
      if (!gl.isContextLost()) { gl.deleteBuffer(buf); gl.deleteProgram(prog); gl.clear(gl.COLOR_BUFFER_BIT); }
    },
  };
}
