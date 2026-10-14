// Halftone storm: an ordered dot grid whose dot sizes come from a drifting, domain-warped cloud field. The brand backdrop.
// One WebGL point per grid cell; the cloud noise, rain dashes, pointer wake and lightning are all evaluated on the GPU, so a
// full-screen frame is one drawArrays call and a dozen uniform writes (no per-frame allocation).
//
// Usage: give the canvas a CSS size (class "halftone" in styles.css fills its positioned parent), then
//   const h = createHalftone(canvas, { seed: 3 }); h.setStorm(.4); ... h.dispose();
// The pointer is read from window, so the canvas can sit under UI with pointer-events: none.
// Reduced motion: no animation loop; a still frame is drawn on create, resize and every setStorm call. flash() is ignored.
// ponytail: no WebGL (or a shader fails) -> a no-op controller and a blank canvas; add a 2D fallback if that ever matters.

export type HalftoneOpts = { cell?: number; color?: string; accent?: string; seed?: number; interactive?: boolean };
export type Halftone = { setStorm(intensity: number): void; flash(): void; setPaused(p: boolean): void; dispose(): void };

const SEGS = 32;            // lightning segments (main channel first, then branches)
const TRAIL = 8;            // pointer wake points behind the live one

const VS = `
precision highp float;
attribute vec2 a_cell;
uniform vec2 u_res, u_off, u_seed, u_flash;
uniform float u_cell, u_dpr, u_t, u_storm, u_maxPt, u_ptrOn;
uniform vec3 u_ptr[${TRAIL + 1}];
uniform vec4 u_seg[${SEGS}];
uniform vec2 u_segm[${SEGS}];
uniform vec3 u_col, u_acc;
varying vec4 v_col;
varying vec3 v_shape;          // dash half-length, radius, sprite size (device px)
varying vec2 v_dir;

float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * .1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * f * (f * (f * 6. - 15.) + 10.);
  return mix(mix(hash(i), hash(i + vec2(1., 0.)), u.x), mix(hash(i + vec2(0., 1.)), hash(i + vec2(1., 1.)), u.x), u.y);
}
const mat2 ROT = mat2(.8, .6, -.6, .8);
float fbm(vec2 p) { float s = 0., a = .5; for (int k = 0; k < 5; k++) { s += a * noise(p); p = ROT * p * 2.03 + 11.7; a *= .5; } return s; }

void main() {
  float s = u_storm, c = u_cell;
  vec2 home = u_off + (a_cell + .5) * c;
  float h1 = hash(a_cell + u_seed), h2 = hash(a_cell.yx + u_seed + 7.1);

  // cloud field: domain-warped fbm, carried east by the wind, slowly re-forming
  vec2 p = home / 400. + u_seed - vec2(.032, .006) * (1. + 2.6 * s) * u_t;
  float tm = u_t * (.025 + .035 * s);
  vec2 q = vec2(fbm(p + tm), fbm(p + vec2(5.2, 1.3) - tm));
  vec2 r = vec2(fbm(p + 2. * q + vec2(1.7, 9.2) + tm), fbm(p + 2. * q + vec2(8.3, 2.8)));
  vec2 w = p + 1.5 * r;
  float d = fbm(w), dl = fbm(w + vec2(-.045, -.065));
  float y = home.y / u_res.y;
  float cov = mix(.53, .37, s) + (y - .45) * .1;                        // heavier toward the top of the sky
  float dens = smoothstep(cov - .04, cov + .17, d);
  float shade = clamp(.5 + (d - dl) * 10., 0., 1.);                    // lit from the upper left

  vec2 uv = home / u_res * 2. - 1.;
  float vig = 1. - .82 * smoothstep(.55, 1.45, length(uv));

  float rmax = .45 * c;
  float rad = (.11 + .89 * pow(dens, 1.35) * mix(.68, 1.06, shade)) * rmax;   // tone lives in dot size, as in print halftone
  float bri = mix(.14, .95, dens) * mix(1., mix(.62, 1.22, shade), dens);
  bri *= (.88 + .24 * h1) * (.94 + .06 * sin(u_t * .8 + h2 * 6.283));  // per-dot variance + a slow shimmer
  bri *= 1. - .24 * s * dens;                                          // storm cloud bodies sit heavier

  // rain: dots under the cloud turn into slanted dashes that hop down their column
  float len = 0.;
  vec2 dir = normalize(vec2(.2 + .25 * s, 1.));
  if (s > .3) {
    float up = fbm(p + vec2(0., -.12));
    float rm = smoothstep(.3, .9, s) * smoothstep(cov - .1, cov + .1, up) * (1. - smoothstep(.12, .4, dens));
    float hc = hash(vec2(a_cell.x, 3.7) + u_seed);
    float L = 130. + 120. * hash(vec2(a_cell.x, 9.1) + u_seed);
    float ph = fract((home.y - u_t * (380. + 280. * hc)) / L + hc * 7.);
    if (ph < .22 * rm) { len = .34 * c; rad = .11 * c; bri = .42 + .3 * h1 + .3 * ph / (.22 * rm); }   // brightest at the falling head
  }

  // pointer wake: dots part around the pointer and spring back (strength can dip negative = overshoot)
  vec2 pos = home;
  if (u_ptrOn > .5) {
    float hole = 0.;
    for (int k = 0; k < ${TRAIL + 1}; k++) {
      vec3 P = u_ptr[k];
      vec2 dv = home - P.xy;
      float g = exp(-dot(dv, dv) / 12100.) * P.z;
      pos += dv * g * .068;   // 20% of the original .34: a light touch
      hole += g;
    }
    rad *= clamp(1. - .13 * hole, .84, 1.06);
  }

  // lightning: the leader lights its path segment by segment, then the return stroke spreads light through the cloud
  float core = 0., glow = 0.;
  if (u_flash.y > .003) {
    float md = 1e4;
    for (int k = 0; k < ${SEGS}; k++) {
      vec4 sg = u_seg[k];
      vec2 m = u_segm[k], pa = home - sg.xy, ba = sg.zw - sg.xy;
      float dd = length(pa - ba * clamp(dot(pa, ba) / max(dot(ba, ba), 1.), 0., 1.));
      float on = smoothstep(m.x, m.x + .02, u_flash.x);
      core = max(core, on * (1. - smoothstep(m.y * .4, m.y, dd)));
      md = min(md, dd + (1. - on) * 1e4);
    }
    float reach = c * 4. + u_flash.x * 560.;
    glow = u_flash.y * exp(-md * md / (reach * reach)) * (.25 + .75 * dens);
    core = min(core * u_flash.y, 1.);
    bri = min(bri + glow * .45, .8);                                   // the cloud lights up, but stays below the bolt
  }
  rad = mix(rad, rmax * 1.14, core);
  len *= 1. - core;
  bri = mix(bri, 1., core);
  vec3 col = mix(mix(u_col, u_acc, min(glow * .35, .35)), vec3(1.), core);

  float a = clamp(bri, 0., 1.) * vig;
  float rd = rad * u_dpr, ld = len * u_dpr;
  if (rd < .7) { a *= rd * rd / .49; rd = .7; }                         // sub-pixel dot: keep it round, trade area for alpha
  float size = min(2. * (ld + rd) + 2., u_maxPt);
  v_col = vec4(col, a);
  v_shape = vec3(ld, rd, size);
  v_dir = dir;
  gl_PointSize = size;
  gl_Position = a < .004 ? vec4(2., 2., 2., 1.) : vec4(pos / u_res * vec2(2., -2.) + vec2(-1., 1.), 0., 1.);
}`;

const FS = `
precision mediump float;
varying vec4 v_col;
varying vec3 v_shape;
varying vec2 v_dir;
void main() {
  vec2 p = (gl_PointCoord - .5) * v_shape.z;
  float a = dot(p, v_dir), b = dot(p, vec2(-v_dir.y, v_dir.x));
  float k = clamp(.5 - (length(vec2(a - clamp(a, -v_shape.x, v_shape.x), b)) - v_shape.y), 0., 1.) * v_col.a;
  if (k < .002) discard;
  gl_FragColor = vec4(v_col.rgb * k, k);
}`;

const UNIFORMS = ["u_res", "u_off", "u_seed", "u_flash", "u_cell", "u_dpr", "u_t", "u_storm", "u_maxPt", "u_ptrOn", "u_ptr",
  "u_seg", "u_segm", "u_col", "u_acc"] as const;

const NOOP: Halftone = { setStorm() {}, flash() {}, setPaused() {}, dispose() {} };

// any CSS colour (not var()) -> 0..1 sRGB
function rgb(css: string): [number, number, number] {
  const x = document.createElement("canvas").getContext("2d");
  if (!x) return [1, 1, 1];
  x.fillStyle = css; x.fillRect(0, 0, 1, 1);
  const d = x.getImageData(0, 0, 1, 1).data;
  return [d[0] / 255, d[1] / 255, d[2] / 255];
}

export function createHalftone(canvas: HTMLCanvasElement, opts: HalftoneOpts = {}): Halftone {
  try { return build(canvas, opts)?.api ?? NOOP; }
  catch (e) { console.warn("halftone disabled:", e); return NOOP; }
}

function build(canvas: HTMLCanvasElement, o: HalftoneOpts) {
  const ctx = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
    powerPreference: "low-power" });
  if (!ctx) return null;
  const gl: WebGLRenderingContext = ctx;
  const cell = Math.max(4, o.cell ?? 10), interactive = o.interactive ?? true;
  const col = rgb(o.color ?? "#dfe5ea"), acc = rgb(o.accent ?? "#6fe3d3");
  const seed = o.seed ?? 1, seedXY = [(seed * .618034 % 1) * 97, (seed * .414214 % 1) * 89];
  const mq = matchMedia("(prefers-reduced-motion: reduce)");

  let prog: WebGLProgram | null = null, buf: WebGLBuffer | null = null, maxPt = 64, n = 0;
  const U = {} as Record<(typeof UNIFORMS)[number], WebGLUniformLocation | null>;
  let W = 0, H = 0, dpr = 1, t = 0, last = 0, raf = 0, paused = false, gone = false, still = mq.matches;
  let storm = 0, target = 0, ft = -1, T = 0, env = 0, nextAuto = 4;
  const seg = new Float32Array(SEGS * 4), segm = new Float32Array(SEGS * 2), ptr = new Float32Array((TRAIL + 1) * 3);
  // pointer: live point (smoothed position + spring strength) and a wake of released points (x, y, s0, age; age < 0 = free)
  let tx = 0, ty = 0, px = 0, py = 0, inside = false, ps = 0, pv = 0, dropX = 0, dropY = 0, slot = 0, ptrOn = false;
  const trail = new Float32Array(TRAIL * 4).fill(-1);

  function init() {
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!; gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader");
      return s;
    };
    prog = gl.createProgram()!;
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) || "link");
    gl.useProgram(prog);
    for (const k of UNIFORMS) U[k] = gl.getUniformLocation(prog, k);
    buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    const a = gl.getAttribLocation(prog, "a_cell"); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.clearColor(0, 0, 0, 0);
    maxPt = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    gl.uniform3fv(U.u_col, col); gl.uniform3fv(U.u_acc, acc); gl.uniform2fv(U.u_seed, seedXY);
    gl.uniform1f(U.u_maxPt, maxPt);
    for (let i = 0; i < SEGS; i++) { segm[i * 2] = 1e6; segm[i * 2 + 1] = 1; }
    gl.uniform2fv(U.u_segm, segm);
    W = H = 0; resize();
  }

  function resize() {
    if (gone || gl.isContextLost()) return;
    const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
    if (!w || !h) return;
    dpr = Math.min(devicePixelRatio || 1, 2);
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    const realloc = canvas.width !== bw || canvas.height !== bh;   // resizing clears the drawing buffer
    if (realloc) { canvas.width = bw; canvas.height = bh; }
    if (w !== W || h !== H) {
      W = w; H = h;
      const cols = Math.ceil(w / cell) + 1, rows = Math.ceil(h / cell) + 1, g = new Float32Array(cols * rows * 2);
      for (let r = 0, i = 0; r < rows; r++) for (let c = 0; c < cols; c++) { g[i++] = c; g[i++] = r; }
      n = cols * rows;
      gl.bufferData(gl.ARRAY_BUFFER, g, gl.STATIC_DRAW);
      gl.uniform2f(U.u_res, w, h); gl.uniform2f(U.u_off, (w - cols * cell) / 2, (h - rows * cell) / 2);
      gl.uniform1f(U.u_cell, cell);
    }
    gl.uniform1f(U.u_dpr, dpr);
    gl.viewport(0, 0, bw, bh);
    if (!raf || realloc) draw();   // observers run after this frame's rAF: redraw now or the cleared buffer gets painted
  }

  function draw() {
    if (!n || gl.isContextLost()) return;
    gl.uniform1f(U.u_t, t); gl.uniform1f(U.u_storm, storm);
    gl.uniform2f(U.u_flash, Math.max(ft, 0), env);
    gl.uniform1f(U.u_ptrOn, ptrOn ? 1 : 0);
    if (ptrOn) gl.uniform3fv(U.u_ptr, ptr);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, n);
  }

  // A bolt: a jittered main channel from high in the sky, then up to a few forks, as segments with leader arrival times.
  function strike() {
    if (!W) return;
    seg.fill(0);
    for (let i = 0; i < SEGS; i++) { segm[i * 2] = 1e6; segm[i * 2 + 1] = 1; }
    const R = Math.random, len0 = H * (.32 + .3 * R()), speed = len0 / .13;
    const todo: number[][] = [[W * (.15 + .7 * R()), H * (.02 + .18 * R()), (R() - .5) * .5, len0, cell * 1.3, 0, 0]];
    let k = 0;
    while (todo.length && k < SEGS) {
      let [x, y, ang, len, w, t0, depth] = todo.shift()!;
      const head = ang;
      while (len > 0 && k < SEGS) {
        const a = ang + (R() - .5) * .95, L = cell * (2 + R() * 2.6), nx = x + Math.sin(a) * L, ny = y + Math.cos(a) * L;
        seg.set([x, y, nx, ny], k * 4); segm[k * 2] = t0; segm[k * 2 + 1] = w; k++;
        t0 += L / speed;
        if (!depth) T = t0;
        if (depth < 2 && R() < .28) todo.push([nx, ny, a + (R() < .5 ? -1 : 1) * (.45 + R() * .6), len * .5, w * .62, t0, depth + 1]);
        x = nx; y = ny; len -= L; ang = a * .6 + head * .4;
      }
    }
    gl.uniform4fv(U.u_seg, seg); gl.uniform2fv(U.u_segm, segm);
    ft = 0;
    nextAuto = 3 + R() * 8 * (1.5 - storm);
  }

  function step(dt: number) {
    storm += (target - storm) * (1 - Math.exp(-dt / .9));
    if (ft >= 0) { ft += dt; if (ft > T + 1.3) ft = -1; }
    else if (storm > .55 && (nextAuto -= dt) <= 0) strike();
    // leader glow, return stroke, one re-strike; two pulses at most, well under the 3-per-second flash limit
    env = ft < 0 ? 0 : ft < T ? .4 : Math.exp(-(ft - T) / .16) + (ft > T + .11 ? .5 * Math.exp(-(ft - T - .11) / .14) : 0);

    if (!interactive) return;
    pv += (140 * ((inside ? 1 : 0) - ps) - 15 * pv) * dt; ps += pv * dt;          // underdamped spring: parts, then settles
    const f = 1 - Math.exp(-dt / .06); px += (tx - px) * f; py += (ty - py) * f;
    ptr[0] = px; ptr[1] = py; ptr[2] = ps;
    let any = Math.abs(ps) > .002;
    for (let i = 0; i < TRAIL; i++) {
      const j = i * 4, o = (i + 1) * 3;
      let s = 0;
      if (trail[j + 3] >= 0) {
        const age = (trail[j + 3] += dt);
        if (age > 1.3) trail[j + 3] = -1;
        else s = trail[j + 2] * Math.exp(-age * 4.5) * Math.cos(age * 10);
      }
      ptr[o] = trail[j]; ptr[o + 1] = trail[j + 1]; ptr[o + 2] = s;
      any ||= Math.abs(s) > .002;
    }
    ptrOn = any;
  }

  const running = () => !paused && !still && !gone && !document.hidden;
  const loop = (now: number) => {
    raf = requestAnimationFrame(loop);
    const dt = Math.max(0, Math.min((now - last) / 1000, .05)); last = now;
    t += dt; step(dt); draw();
  };
  const start = () => { if (!raf && running()) { last = performance.now(); raf = requestAnimationFrame(loop); } };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };

  const move = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    tx = e.clientX - r.left; ty = e.clientY - r.top;
    inside = tx >= 0 && ty >= 0 && tx <= r.width && ty <= r.height;
    if (ps < .02) { px = dropX = tx; py = dropY = ty; }                         // appear where the pointer is, don't sweep in
    if (Math.hypot(tx - dropX, ty - dropY) > 26) {                              // leave a wake point behind
      const j = slot * 4;                                                       // round robin: the oldest point is reused
      trail[j] = dropX; trail[j + 1] = dropY; trail[j + 2] = .7 * Math.max(ps, 0); trail[j + 3] = 0;
      slot = (slot + 1) % TRAIL; dropX = tx; dropY = ty;
    }
  };
  const out = (e: PointerEvent) => { if (!e.relatedTarget) inside = false; };            // left the window (touch: finger lifted)
  const blur = () => { inside = false; };
  const vis = () => document.hidden ? stop() : start();
  const motion = () => { still = mq.matches; if (still) { stop(); storm = target; ft = -1; env = 0; ptrOn = false; draw(); } else start(); };
  const lost = (e: Event) => { e.preventDefault(); stop(); };
  const restored = () => { init(); start(); };

  init();   // first: if it throws (shader failure, context lost at birth) nothing below is attached, so the caller's no-op handle leaks nothing
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  document.addEventListener("visibilitychange", vis);
  mq.addEventListener("change", motion);
  canvas.addEventListener("webglcontextlost", lost);
  canvas.addEventListener("webglcontextrestored", restored);
  if (interactive) {
    addEventListener("pointermove", move, { passive: true });
    addEventListener("pointerout", out);
    addEventListener("blur", blur);
  }
  start();

  const api: Halftone = {
    setStorm(x) {
      target = Math.min(1, Math.max(0, +x || 0));
      if (!raf) { storm = target; draw(); }
    },
    flash() { if (!still && !gone) { strike(); start(); } },
    setPaused(p) { paused = p; if (p) stop(); else start(); },
    dispose() {
      if (gone) return;
      gone = true; stop(); ro.disconnect();
      document.removeEventListener("visibilitychange", vis);
      mq.removeEventListener("change", motion);
      canvas.removeEventListener("webglcontextlost", lost);
      canvas.removeEventListener("webglcontextrestored", restored);
      removeEventListener("pointermove", move); removeEventListener("pointerout", out); removeEventListener("blur", blur);
      // No WEBGL_lose_context here: React StrictMode remounts on the same canvas, and a lost context can't be fetched again.
      gl.deleteBuffer(buf); gl.deleteProgram(prog);
      gl.clear(gl.COLOR_BUFFER_BIT);
    },
  };
  return { api, gl };
}

// Not wired into the UI. Draws one stormy frame on a detached canvas and checks that a sensible share of pixels got dots.
export function halftoneSelfTest(): true {
  const c = document.createElement("canvas");
  c.width = 240; c.height = 160;
  const h = build(c, { seed: 7, interactive: false });
  if (!h) throw new Error("halftone: WebGL unavailable");
  h.api.setPaused(true);
  h.api.setStorm(.8);                                   // paused -> draws synchronously
  const px = new Uint8Array(c.width * c.height * 4);
  h.gl.readPixels(0, 0, c.width, c.height, h.gl.RGBA, h.gl.UNSIGNED_BYTE, px);
  h.api.dispose();
  let lit = 0;
  for (let i = 3; i < px.length; i += 4) if (px[i] > 10) lit++;
  const f = lit / (c.width * c.height);
  if (!(f > .02 && f < .9)) throw new Error(`halftone: ${(f * 100).toFixed(1)}% of pixels lit`);
  return true;
}
