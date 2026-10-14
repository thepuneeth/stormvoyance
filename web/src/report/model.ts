// Decoded report data + the small physics helpers the viewer shares (same formulas as stormvoyance.py).
// ponytail: the engine bundle is untyped here (`any`); add a schema when a second consumer appears.
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Bundle = any;
export type Hour = any;
export type Tree = any;
export type Sector = { dir: number; z0: number; dx: number; z_base: number; nw: number; nz: number; exposure: number; lattice: number[];
  U: Int8Array; SG: Uint8Array; N3: number };

export const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
export const ft = (m: number) => Math.round(m * 3.281);
// what to call the simulated building: "Your house" for homes, else its OSM name or construction class
export const bldName = (H: { cls?: string; name?: string | null }) => !H.cls || H.cls === "WSF" ? "Your house" : H.name || "Your building";
export const pct = (p: number | null | undefined) => p == null ? "n/a" : p < .005 ? "<1%" : p < .1 ? (p * 100).toFixed(1).replace(/\.0$/, "") + "%" : p > .99 ? ">99%" : Math.round(p * 100) + "%";
const C16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
export const compass = (d: number) => C16[Math.round((((d % 360) + 360) % 360) / 22.5) % 16];
export const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const bearing = (la1: number, lo1: number, la2: number, lo2: number) => {
  const r = Math.PI / 180, y = Math.sin((lo2 - lo1) * r) * Math.cos(la2 * r),
    x = Math.cos(la1 * r) * Math.sin(la2 * r) - Math.sin(la1 * r) * Math.cos(la2 * r) * Math.cos((lo2 - lo1) * r);
  return (Math.atan2(y, x) / r + 360) % 360;
};
const bytes = (s: string) => { const b = atob(s), a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; };

// gust colour ramp (mph): sequential; hue turns warm at the 40 mph "branches start breaking" range
export const RAMP: [number, number[]][] = [[0, [34, 62, 104]], [10, [38, 122, 160]], [20, [64, 186, 170]], [30, [196, 224, 112]], [40, [250, 192, 72]], [50, [244, 122, 60]], [60, [224, 60, 82]], [80, [168, 48, 160]]];
export function ramp(v: number, out: number[]) {
  let i = 1; while (i < RAMP.length - 1 && v > RAMP[i][0]) i++;
  const [a, ca] = RAMP[i - 1], [b, cb] = RAMP[i], t = clamp((v - a) / (b - a), 0, 1);
  for (let k = 0; k < 3; k++) out[k] = (ca[k] + (cb[k] - ca[k]) * t) / 255;
  return out;
}
export const rampLin = (v: number, out: number[]) => { ramp(v, out); out[0] **= 2.2; out[1] **= 2.2; out[2] **= 2.2; return out; };  // three.js vertex colours are linear

const I_OPEN = 1 / Math.log(10 / 0.03);
// local 3-s gust / open-terrain gust = exposure x (mean + 3 sigma) / (1 + 3 I_open)
export function gustRatio(K: number, sig: number, sec: Sector, zAgl = 10) {
  sig = Math.max(sig, K / Math.log(Math.max(zAgl, 1) / sec.z0 + 1));
  return sec.exposure * (K + 3 * sig) / (1 + 3 * I_OPEN);
}

export type Model = ReturnType<typeof makeModel>;

export function makeModel(D: Bundle) {
  // runs made before the engine fix list the house itself among the neighbours; drop it
  D.buildings = D.buildings.filter((b: { cx: number; cy: number }) => Math.hypot(b.cx, b.cy) > 1);
  const S = D.sim, HRS: Hour[] = S.hours, NT = HRS.length, TZ = D.meta.tz, HALF = D.scene.half;
  const fmt = (iso: string, o: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, ...o }).format(new Date(iso));
  const hourLbl = (iso: string) => fmt(iso, { weekday: "short", hour: "numeric" });
  const peakIdx = Math.max(0, HRS.findIndex(h => h.t === S.peak.hour));

  const t = D.scene.terrain, TER = { n: t.n as number, dx: t.dx as number, a: new Int16Array(bytes(t.cm).buffer) };
  function terrH(x: number, y: number) {
    const n = TER.n, fi = clamp((HALF - y) / TER.dx - .5, 0, n - 1.001), fj = clamp((x + HALF) / TER.dx - .5, 0, n - 1.001), i = fi | 0, j = fj | 0, p = fi - i, q = fj - j, a = TER.a;
    return ((a[i * n + j] * (1 - q) + a[i * n + j + 1] * q) * (1 - p) + (a[(i + 1) * n + j] * (1 - q) + a[(i + 1) * n + j + 1] * q) * p) / 100;
  }

  const W = D.wind as null | { sectors: Sector[]; hour_sector: number[]; G_open: number[]; steps: number };
  if (W) for (const s of W.sectors as any[]) if (s.u) { s.U = new Int8Array(bytes(s.u).buffer); s.SG = bytes(s.sig); s.u = s.sig = null; s.N3 = s.nw * s.nw * s.nz; }
  const secOf = (i: number) => W ? W.sectors[W.hour_sector[i]] : null;

  const OUT = new Float32Array(4);
  // world x east, y north, z above house ground -> [east, north, up, sigma], in units of the inflow wind at 10 m
  function wind(sec: Sector, x: number, y: number, z: number) {
    const nw = sec.nw, nz = sec.nz, fc = clamp(nw / 2 + x / sec.dx - .5, 0, nw - 1.001), fr = clamp(nw / 2 - y / sec.dx - .5, 0, nw - 1.001), fk = clamp((z - sec.z_base) / sec.dx - .5, 0, nz - 1.001);
    const c = fc | 0, r = fr | 0, k = fk | 0, a = fc - c, b = fr - r, e = fk - k, U = sec.U, G = sec.SG, N3 = sec.N3;
    OUT[0] = OUT[1] = OUT[2] = OUT[3] = 0;
    for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 2; dc++) for (let dk = 0; dk < 2; dk++) {
      const w = (dr ? b : 1 - b) * (dc ? a : 1 - a) * (dk ? e : 1 - e), o = ((r + dr) * nw + c + dc) * nz + k + dk;
      OUT[0] += w * U[o]; OUT[1] += w * U[N3 + o]; OUT[2] += w * U[2 * N3 + o]; OUT[3] += w * G[o];
    }
    OUT[0] /= 50; OUT[1] /= 50; OUT[2] /= 50; OUT[3] /= 200; return OUT;
  }
  function localGust(x: number, y: number, zAgl: number, i: number) {
    const sec = secOf(i); if (!sec) return HRS[i].gust[2];
    const o = wind(sec, x, y, terrH(x, y) + zAgl);
    return HRS[i].gust[2] * gustRatio(Math.hypot(o[0], o[1], o[2]), o[3], sec, zAgl);
  }
  return { D, S, HRS, NT, HALF, W, TER, fmt, hourLbl, peakIdx, terrH, secOf, wind, localGust };
}

// ---- report UI helpers (pure; no DOM) ----

// Smooth line through points that never overshoots between them (monotone cubic, Fritsch-Carlson tangents, as d3's
// curveMonotoneX), so a gust curve never dips below zero or bulges past a peak that wasn't forecast. Works right-to-left too.
export function monotonePath(P: [number, number][]): string {
  const n = P.length, f = (v: number) => +v.toFixed(1);
  if (!n) return "";
  if (n < 3) return P.map((p, i) => (i ? "L" : "M") + f(p[0]) + " " + f(p[1])).join("");
  const s: number[] = [], m: number[] = [];
  for (let i = 0; i < n - 1; i++) s.push((P[i + 1][1] - P[i][1]) / ((P[i + 1][0] - P[i][0]) || 1e-9));
  m[0] = s[0]; m[n - 1] = s[n - 2];
  for (let i = 1; i < n - 1; i++) {
    const h0 = P[i][0] - P[i - 1][0], h1 = P[i + 1][0] - P[i][0], p = (s[i - 1] * h1 + s[i] * h0) / ((h0 + h1) || 1e-9);
    m[i] = (Math.sign(s[i - 1]) + Math.sign(s[i])) * Math.min(Math.abs(s[i - 1]), Math.abs(s[i]), .5 * Math.abs(p)) || 0;
  }
  let d = `M${f(P[0][0])} ${f(P[0][1])}`;
  for (let i = 0; i < n - 1; i++) {
    const [x0, y0] = P[i], [x1, y1] = P[i + 1], dx = (x1 - x0) / 3;
    d += `C${f(x0 + dx)} ${f(y0 + dx * m[i])} ${f(x1 - dx)} ${f(y1 - dx * m[i + 1])} ${f(x1)} ${f(y1)}`;
  }
  return d;
}

// Value at probability q of an hour's gust quantiles [p5, p25, p50, p75, p95], linear between them. q is clamped to
// .05..95: nothing outside the forecast's own 90% range is drawn.
const QS = [.05, .25, .5, .75, .95];
export function quantileAt(g: number[], q: number) {
  q = clamp(q, .05, .95);
  let i = 1; while (i < 4 && q > QS[i]) i++;
  return g[i - 1] + (g[i] - g[i - 1]) * (q - QS[i - 1]) / (QS[i] - QS[i - 1]);
}

// Index into D.trees of the tree most likely to hit the house, ties (usually all zero) broken by chance of failing. -1 if none.
export function riskiestTree(trees: Tree[]) {
  let b = -1;
  trees.forEach((t, i) => {
    const h = t.p_hit || 0, f = t.p_fail || 0, bh = b < 0 ? -1 : trees[b].p_hit || 0, bf = b < 0 ? -1 : trees[b].p_fail || 0;
    if (h > bh || (h === bh && f > bf)) b = i;
  });
  return b;
}

// Not wired into the UI. Throws on the first broken helper.
export function reportSelfTest(): true {
  const P: [number, number][] = [[0, 0], [1, 10], [2, 10], [3, 30], [4, 2], [5, 2]];
  const nums = monotonePath(P).slice(1).split(/[C ]/).map(Number);
  for (let i = 0; i < P.length - 1; i++) {                       // every Bezier stays inside its two end points' y range
    const [y0, y1] = [P[i][1], P[i + 1][1]], c = nums.slice(2 + i * 6, 2 + i * 6 + 6), b = [y0, c[1], c[3], y1];
    for (let t = 0; t <= 1; t += .05) {
      const y = (1 - t) ** 3 * b[0] + 3 * (1 - t) ** 2 * t * b[1] + 3 * (1 - t) * t * t * b[2] + t ** 3 * b[3];
      if (y < Math.min(y0, y1) - 1e-6 || y > Math.max(y0, y1) + 1e-6) throw new Error(`monotonePath overshoots in segment ${i}: ${y}`);
    }
  }
  if (!monotonePath([...P].reverse()).startsWith("M5 2")) throw new Error("monotonePath reversed");
  const g = [10, 20, 30, 40, 60];
  for (const [q, v] of [[0, 10], [.05, 10], [.15, 15], [.5, 30], [.85, 50], [1, 60]]) if (Math.abs(quantileAt(g, q) - v) > 1e-9) throw new Error(`quantileAt ${q}`);
  if (riskiestTree([{ p_hit: 0, p_fail: .2 }, { p_hit: .01, p_fail: .01 }, { p_hit: .01, p_fail: .05 }]) !== 2 || riskiestTree([]) !== -1) throw new Error("riskiestTree");
  return true;
}
