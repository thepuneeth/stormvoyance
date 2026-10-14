// CFD-post style streamlines: RK2-integrated through the steady LBM mean field from seed rakes, drawn as screen-space
// lines coloured by local gust. Comet pulses travel at the local air speed (the dash coordinate is time of flight, not
// distance, the way ParaView / Fluent animate pathlines): bright HDR heads (they bloom), fading tails, width that
// follows local speed, and a fade near the camera. reveal() "draws the field on" along time of flight from the inflow
// edge, like dye released at t = 0. Tracing is time-sliced across frames and cached per wind sector, so the first
// frame never waits for it and scrubbing between sectors is instant once each has been traced.
import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { clamp, gustRatio, ramp, type Model, type Sector } from "./model";

type Job = { sec: Sector; seeds: number[]; i: number; H: number; p: number[]; k: number[]; s: number[]; z: number[]; t: number[]; seg: number[]; ends: number[]; ms: number };
type Built = { sec: Sector; geo: LineSegmentsGeometry; col: THREE.InstancedInterleavedBuffer; seg: Int32Array; k: Float32Array; s: Float32Array; z: Float32Array;
  pc: Float32Array; tMax: number; painted: number; lines: number; ms: number };

const DASH = 6, GAP = 30;   // pulse length and spacing, time-of-flight units (m travelled at the inflow speed)
const ds = 1.2;             // integration step, m

// LineMaterial + reveal front, comet pulses, per-vertex width and near-camera fade, patched into three's own shader.
function lineMaterial(p: { linewidth: number; opacity: number; comet: boolean }) {
  const m = new LineMaterial({ linewidth: p.linewidth, opacity: p.opacity, vertexColors: true, transparent: true, depthWrite: false, toneMapped: false,
    dashed: true, dashSize: p.comet ? DASH : 1e9, gapSize: p.comet ? GAP : 0, blending: p.comet ? THREE.AdditiveBlending : THREE.NormalBlending });
  Object.assign(m.uniforms, { uReveal: { value: 0 }, uFront: { value: 0 }, uHead: { value: p.comet ? 3.5 : 0 }, uNear: { value: new THREE.Vector4(3, 24, 340, 720) } });
  if (p.comet) m.defines.COMET = "";
  const V: [string, string][] = [
    ["uniform vec2 resolution;", "uniform vec2 resolution;\nattribute float instanceWidthStart;\nattribute float instanceWidthEnd;\nvarying float vDepth;"],
    ["offset *= linewidth;", "offset *= linewidth * ( ( position.y < 0.5 ) ? instanceWidthStart : instanceWidthEnd );"],
    ["// this is an approximation", "// this is an approximation\nvDepth = - mvPosition.z;"],
  ];
  const F: [string, string][] = [
    ["uniform float linewidth;", "uniform float linewidth;\nuniform float uReveal, uFront, uHead;\nuniform vec4 uNear;\nvarying float vDepth;"],
    ["if ( mod( vLineDistance + dashOffset, dashSize + gapSize ) > dashSize ) discard;", `
      if ( vLineDistance > uReveal ) discard;
      float glow = 1.0 + uFront * exp( ( vLineDistance - uReveal ) * 0.08 );
      #ifdef COMET
        float u = mod( vLineDistance + dashOffset, dashSize + gapSize );
        if ( u > dashSize ) discard;
        float c = u / dashSize;                       // 0 tail .. 1 head
        alpha *= c * c * ( 1.0 - vUv.x * vUv.x );     // fading tail, soft cross-section
        glow += uHead * pow( c, 10.0 );
      #endif
      alpha *= smoothstep( uNear.x, uNear.y, vDepth ) * ( 1.0 - 0.7 * smoothstep( uNear.z, uNear.w, vDepth ) );   // fade at the lens and into the distance`],
    ["gl_FragColor = vec4( diffuseColor.rgb, alpha );", `vec3 lc = diffuseColor.rgb;
      lc = mix( lc, lc / max( max( lc.r, lc.g ), max( lc.b, 1e-3 ) ), clamp( glow - 1.0, 0.0, 1.0 ) );   // hot heads and the drawing front keep their hue
      gl_FragColor = vec4( lc * glow, alpha );`],
  ];
  // all anchors or none: a three.js upgrade that moves them leaves plain lines rather than a shader that fails to compile
  if (V.every(([a]) => m.vertexShader.includes(a)) && F.every(([a]) => m.fragmentShader.includes(a))) {
    for (const [a, b] of V) m.vertexShader = m.vertexShader.replace(a, b);
    for (const [a, b] of F) m.fragmentShader = m.fragmentShader.replace(a, b);
  } else console.warn("streamlines: LineMaterial shader changed; drawing plain lines");
  return m;
}

const ease = (t: number) => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

export function createStreamlines(M: Model, sc: THREE.Scene, eave: number, onBuilt?: (b: { lines: number; segs: number; ms: number }) => void) {
  const { HRS, W, terrH, secOf, wind } = M;
  const base = lineMaterial({ linewidth: 1.25, opacity: .34, comet: false }), pulse = lineMaterial({ linewidth: 3.4, opacity: .9, comet: true });
  const a = new LineSegments2(new LineSegmentsGeometry(), base), b = new LineSegments2(a.geometry, pulse);
  a.renderOrder = 4; b.renderOrder = 5; a.frustumCulled = b.frustumCulled = false; a.visible = b.visible = false; sc.add(a, b);
  const cache = new Map<Sector, Built>(), jobs = new Map<Sector, Job>();
  let cur: Built | null = null, shown = 0;
  const rv = { t: 0, dur: 0, held: true, since: 0 };   // reveal: held at nothing until reveal() (the intro) releases it

  function job(sec: Sector): Job {
    let j = jobs.get(sec); if (j) return j;
    const H = sec.nw * sec.dx / 2 - 2, r = sec.dir * Math.PI / 180, fx = -Math.sin(r), fy = -Math.cos(r), seeds: number[] = [];   // flow toward
    const seed = (along: number, across: number, zAgl: number) => seeds.push(along * fx - across * fy, along * fy + across * fx, zAgl);
    for (const z of [1.5, 4, 8, 14, 22]) for (let c = -H * .92; c <= H * .92; c += 7) seed(-H * .92, c, z);            // domain rake
    for (const z of [1.5, eave * .6, eave + 2]) for (let c = -24; c <= 24; c += 3) seed(-30, c, z);            // dense rake on the house
    j = { sec, seeds, i: 0, H, p: [], k: [], s: [], z: [], t: [], seg: [], ends: [], ms: 0 }; jobs.set(sec, j); return j;
  }
  function trace(j: Job, x: number, y: number, zAgl: number) {
    // each line starts at a random pulse phase: otherwise every head on a rake leaves in step and they pile into one white band
    const sec = j.sec, n0 = j.k.length; let z = terrH(x, y) + zAgl, t = Math.random() * (DASH + GAP);
    for (let n = 0; n < 380; n++) {
      const gz = terrH(x, y); z = Math.max(z, gz + .3);
      const o = wind(sec, x, y, z), K = Math.hypot(o[0], o[1], o[2]);
      j.p.push(x, z, -y); j.k.push(K); j.s.push(o[3]); j.z.push(z - gz); j.t.push(t);
      if (K < .04 || Math.abs(x) > j.H || Math.abs(y) > j.H) break;          // stagnation / inside a building / left the domain
      const ux = o[0] / K, uy = o[1] / K, uz = o[2] / K;                     // RK2 (midpoint) on the unit direction
      const m = wind(sec, x + ux * ds / 2, y + uy * ds / 2, z + uz * ds / 2), Km = Math.hypot(m[0], m[1], m[2]) || K;
      x += m[0] / Km * ds; y += m[1] / Km * ds; z += m[2] / Km * ds; t += ds / Km;   // t in m per unit inflow speed
    }
    const n = j.k.length - n0;
    if (n <= 8) { j.p.length = n0 * 3; j.k.length = j.s.length = j.z.length = j.t.length = n0; return; }
    for (let i = n0; i < n0 + n - 1; i++) j.seg.push(i);
    j.ends.push(j.t[n0 + n - 1]);
  }
  // trace seeds until the deadline; true when the sector is done
  function work(j: Job, until: number) {
    const t0 = performance.now();
    while (j.i < j.seeds.length && performance.now() < until) { trace(j, j.seeds[j.i], j.seeds[j.i + 1], j.seeds[j.i + 2]); j.i += 3; }
    j.ms += performance.now() - t0;
    return j.i >= j.seeds.length;
  }
  function finish(j: Job): Built {
    const nS = j.seg.length, pos = new Float32Array(nS * 6), aux = new Float32Array(nS * 4);
    const w = (K: number) => .55 + .65 * clamp(K / 1.3, 0, 1);                                         // width follows local speed
    for (let q = 0; q < nS; q++) {
      const i = j.seg[q];
      for (let c = 0; c < 6; c++) pos[6 * q + c] = j.p[3 * i + c];
      aux[4 * q] = j.t[i]; aux[4 * q + 1] = j.t[i + 1]; aux[4 * q + 2] = w(j.k[i]); aux[4 * q + 3] = w(j.k[i + 1]);
    }
    const geo = new LineSegmentsGeometry(); geo.setPositions(pos);
    const A = new THREE.InstancedInterleavedBuffer(aux, 4, 1);
    geo.setAttribute("instanceDistanceStart", new THREE.InterleavedBufferAttribute(A, 1, 0));
    geo.setAttribute("instanceDistanceEnd", new THREE.InterleavedBufferAttribute(A, 1, 1));
    geo.setAttribute("instanceWidthStart", new THREE.InterleavedBufferAttribute(A, 1, 2));
    geo.setAttribute("instanceWidthEnd", new THREE.InterleavedBufferAttribute(A, 1, 3));
    const col = new THREE.InstancedInterleavedBuffer(new Float32Array(nS * 6), 6, 1);
    geo.setAttribute("instanceColorStart", new THREE.InterleavedBufferAttribute(col, 3, 0));
    geo.setAttribute("instanceColorEnd", new THREE.InterleavedBufferAttribute(col, 3, 3));
    const ends = j.ends.slice().sort((x, y) => x - y), tMax = ends.length ? ends[Math.floor(ends.length * .9)] : 1;   // most lines finish drawing on time
    jobs.delete(j.sec);
    onBuilt?.({ lines: j.ends.length, segs: nS, ms: Math.round(j.ms) });
    return { sec: j.sec, geo, col, seg: Int32Array.from(j.seg), k: Float32Array.from(j.k), s: Float32Array.from(j.s), z: Float32Array.from(j.z),
      pc: new Float32Array(j.k.length * 3), tMax, painted: -1, lines: j.ends.length, ms: j.ms };
  }
  function paint(B: Built, i: number) {
    const gO = HRS[i].gust[2], rgb = [0, 0, 0], pc = B.pc, arr = B.col.array as Float32Array;
    for (let p = 0; p < B.k.length; p++) { ramp(gO * gustRatio(B.k[p], B.s[p], B.sec, B.z[p]), rgb); pc[3 * p] = rgb[0] ** 2.2; pc[3 * p + 1] = rgb[1] ** 2.2; pc[3 * p + 2] = rgb[2] ** 2.2; }
    for (let q = 0; q < B.seg.length; q++) { const s = 3 * B.seg[q]; for (let c = 0; c < 6; c++) arr[6 * q + c] = pc[s + c]; }
    B.col.needsUpdate = true; B.painted = i;
  }
  const order = W ? [...new Set(W.hour_sector)].map(k => W.sectors[k]) : [];
  // draw the field on along time of flight over `dur` seconds (0 = at once)
  const reveal = (dur: number) => { rv.t = 0; rv.dur = dur; rv.held = false; };

  return {
    // returns true when something drawn changed (for render-on-demand)
    update(i: number, on: boolean, dt: number, animate: boolean) {
      const s = secOf(i); let changed = false;
      if (s) {
        const now = performance.now();
        if (!cache.has(s)) { if (work(job(s), now + 7)) cache.set(s, finish(job(s))); }
        else { const next = order.find(x => !cache.has(x)); if (next && work(job(next), now + 2.5)) cache.set(next, finish(job(next))); }   // pre-trace the other sectors
        const B = cache.get(s);
        if (B && B !== cur) { cur = B; a.geometry = b.geometry = B.geo; if (shown++) reveal(animate ? 1.3 : 0); changed = true; }
      }
      if (!rv.held) rv.t += dt; else if ((rv.since += dt) > 6) rv.held = false;   // safety net: never stay hidden
      const vis = !!(on && s && cur);
      a.visible = vis; b.visible = vis && animate;
      if (!vis || !cur) return changed;
      if (cur.painted !== i) { paint(cur, i); changed = true; }
      const p = rv.held ? 0 : rv.dur > 0 ? clamp(rv.t / rv.dur, 0, 1) : 1;
      const R = p >= 1 ? 1e9 : ease(p) * cur.tMax, front = p >= 1 ? 0 : 1.8 * (1 - p ** 4);
      if (base.uniforms.uReveal.value !== R) changed = true;
      base.uniforms.uReveal.value = pulse.uniforms.uReveal.value = R; base.uniforms.uFront.value = pulse.uniforms.uFront.value = front;
      // pulses move at the local air speed: inflow speed (m/s at 10 m) x time, time-lapsed x4 so they read at a glance
      if (animate) pulse.dashOffset = (pulse.dashOffset - dt * HRS[i].sust * .447 * s!.exposure * 4) % (DASH + GAP);
      return changed || animate;
    },
    reveal,
    ready: () => !!cur,
    resize(w: number, h: number) { base.resolution.set(w, h); pulse.resolution.set(w, h); },
    dispose() { sc.remove(a, b); for (const B of cache.values()) B.geo.dispose(); a.geometry.dispose(); base.dispose(); pulse.dispose(); },
  };
}
