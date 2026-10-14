// 2D top-down view of the simulation: satellite image, CFD gust map at a chosen height, animated wind particles,
// footprints and LiDAR trees (risk-coloured). North up; wheel to zoom, drag to pan. Drawn in the report's dark
// "halftone storm" style: greyed imagery, hairlines, dotted crowns, mono labels; colour only for gust and risk.
import { clamp, css, gustRatio, ramp, type Model, type Tree, bldName } from "./model";
import type { Layers } from "./siteScene";

type St = { idx: number; layer: Layers; sliceH: number; active: boolean };

export function createPlan(M: Model, cv: HTMLCanvasElement, onHover: (t: Tree | null, x: number, y: number) => void) {
  const { D, HRS, HALF, W, terrH, secOf, wind } = M, HS = D.house;
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ctx = cv.getContext("2d")!, img = new Image(); img.src = D.scene.img;
  const trail = document.createElement("canvas"), tctx = trail.getContext("2d")!;
  const heat = document.createElement("canvas"), GN = 140; heat.width = heat.height = GN; const hctx = heat.getContext("2d")!;
  let st: St = { idx: M.peakIdx, layer: { flow: true, slice: true, risk: false, rain: false, lidar: false }, sliceH: 2, active: false };
  let dpr = 1, cw = 0, ch = 0, zoom = 1, panX = 0, panY = 0, raf = 0, last = performance.now();
  const EXT = Math.min(140, HALF);   // CFD domain half-width
  const scale = () => Math.min(cw, ch) / (2 * EXT) * zoom;
  const toS = (x: number, y: number): [number, number] => [cw / 2 + (x - panX) * scale(), ch / 2 - (y - panY) * scale()];
  const toW = (sx: number, sy: number): [number, number] => [panX + (sx - cw / 2) / scale(), panY - (sy - ch / 2) / scale()];

  function size() {
    const r = cv.getBoundingClientRect(); if (!r.width) return; dpr = Math.min(2, devicePixelRatio || 1);
    cw = cv.width = trail.width = Math.round(r.width * dpr); ch = cv.height = trail.height = Math.round(r.height * dpr);
  }

  // gust map at sliceH: same CFD sampling as the 3D sheet
  function paintHeat() {
    const sec = secOf(st.idx); if (!sec) return;
    const im = hctx.createImageData(GN, GN), rgb = [0, 0, 0], g0 = HRS[st.idx].gust[2];
    for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) {
      const x = -EXT + (i + .5) / GN * 2 * EXT, y = EXT - (j + .5) / GN * 2 * EXT, z = terrH(x, y) + st.sliceH, o = wind(sec, x, y, z), K = Math.hypot(o[0], o[1], o[2]), k = (j * GN + i) * 4;
      if (K < .02) { im.data[k + 3] = 200; continue; }   // inside a building / no flow
      ramp(g0 * gustRatio(K, o[3], sec, st.sliceH), rgb); im.data[k] = rgb[0] * 255; im.data[k + 1] = rgb[1] * 255; im.data[k + 2] = rgb[2] * 255; im.data[k + 3] = 255;
    }
    hctx.putImageData(im, 0, 0);
  }

  // particles advected through the CFD mean field at sliceH (time-lapse so motion reads at map scale)
  const NP = RM ? 0 : 2600, px = new Float32Array(NP), py = new Float32Array(NP), life = new Float32Array(NP);
  const spawn = (i: number) => { px[i] = (Math.random() * 2 - 1) * EXT; py[i] = (Math.random() * 2 - 1) * EXT; life[i] = 3 + Math.random() * 6; };
  for (let i = 0; i < NP; i++) spawn(i);
  function particles(dt: number) {
    const sec = secOf(st.idx); if (!sec) return;
    const Uref = HRS[st.idx].sust * .447 * sec.exposure, g0 = HRS[st.idx].gust[2], rgb = [0, 0, 0], s = scale(), step = dt * 7;
    tctx.globalCompositeOperation = "destination-out"; tctx.fillStyle = "rgba(0,0,0,.05)"; tctx.fillRect(0, 0, cw, ch); tctx.globalCompositeOperation = "source-over";
    tctx.lineWidth = Math.max(1, 1.1 * dpr * Math.sqrt(zoom)); tctx.lineCap = "round";
    for (let i = 0; i < NP; i++) {
      const x = px[i], y = py[i], z = terrH(x, y) + st.sliceH, o = wind(sec, x, y, z), K = Math.hypot(o[0], o[1]);
      const nx = x + o[0] * Uref * step, ny = y + o[1] * Uref * step; life[i] -= dt;
      if (life[i] < 0 || K < .03 || Math.abs(nx) > EXT || Math.abs(ny) > EXT) { spawn(i); continue; }
      ramp(g0 * gustRatio(Math.hypot(o[0], o[1], o[2]), o[3], sec, st.sliceH), rgb);
      tctx.strokeStyle = `rgb(${(.35 + .65 * rgb[0]) * 255 | 0},${(.35 + .65 * rgb[1]) * 255 | 0},${(.35 + .65 * rgb[2]) * 255 | 0})`;
      const [a, b] = toS(x, y); tctx.beginPath(); tctx.moveTo(a, b); tctx.lineTo(a + (nx - x) * s, b - (ny - y) * s); tctx.stroke();
      px[i] = nx; py[i] = ny;
    }
  }

  const rect = (b: { cx: number; cy: number; L: number; W: number; heading: number }) => {
    const a = (90 - b.heading) * Math.PI / 180, pts = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([u, v]) => { const lx = u * b.L / 2, lz = v * b.W / 2; return toS(b.cx + lx * Math.cos(a) + lz * Math.sin(a), b.cy + lx * Math.sin(a) - lz * Math.cos(a)); });
    ctx.beginPath(); pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath();
  };
  const riskCol = (p: number) => p < .002 ? null : p < .02 ? warn : bad;
  const [ink, muted, accent, night, warn, bad, MONO] = ["--ink", "--muted", "--accent", "--night", "--warn", "--bad", "--f-data"].map(css);   // tokens, read once

  // text on a glass pill; x is the centre, y the baseline
  function pill(text: string, x: number, y: number, font: string, color: string) {
    ctx.font = font; const w = ctx.measureText(text).width, p = 9 * dpr, h = 22 * dpr;
    ctx.beginPath(); ctx.roundRect(x - w / 2 - p, y - h + 6 * dpr, w + 2 * p, h, h / 2);
    ctx.fillStyle = "rgba(10,13,17,.72)"; ctx.fill(); ctx.lineWidth = dpr; ctx.strokeStyle = "rgba(255,255,255,.1)"; ctx.stroke();
    ctx.fillStyle = color; ctx.textAlign = "center"; ctx.fillText(text, x, y); ctx.textAlign = "left";
  }

  function draw() {
    if (!cw) return; const s = scale(), sec = secOf(st.idx), mono = (px: number) => `500 ${px * dpr}px ${MONO}`;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = night; ctx.fillRect(0, 0, cw, ch);
    const [ix, iy] = toS(-HALF, HALF);
    if (img.complete) { ctx.filter = "grayscale(1) contrast(1.05)"; ctx.drawImage(img, ix, iy, 2 * HALF * s, 2 * HALF * s); ctx.filter = "none"; ctx.fillStyle = "rgba(6,8,11,.42)"; ctx.fillRect(0, 0, cw, ch); }
    if (W && st.layer.slice && sec) { const [hx, hy] = toS(-EXT, EXT); ctx.globalAlpha = .5; ctx.imageSmoothingEnabled = true; ctx.drawImage(heat, hx, hy, 2 * EXT * s, 2 * EXT * s); ctx.globalAlpha = 1; }
    else { ctx.fillStyle = "rgba(6,8,11,.25)"; ctx.fillRect(0, 0, cw, ch); }
    // trees: crowns from LiDAR as dotted rings; the risk layer fills them by P(fail)
    ctx.lineCap = "round";
    for (const t of D.trees as Tree[]) {
      if (t.h < 8 && !st.layer.risk) continue; const [x, y] = toS(t.x, t.y), r = t.cw / 2 * s, c = st.layer.risk ? riskCol(t.p_fail || 0) : null;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.lineWidth = dpr;
      if (c) { ctx.fillStyle = c; ctx.globalAlpha = .32; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = c; ctx.stroke(); }
      else { ctx.setLineDash([0, 3.5 * dpr]); ctx.lineWidth = 1.6 * dpr; ctx.strokeStyle = "rgba(255,255,255,.3)"; ctx.stroke(); ctx.setLineDash([]); }
    }
    ctx.lineWidth = dpr; ctx.strokeStyle = "rgba(255,255,255,.38)"; for (const b of D.buildings) { rect(b); ctx.stroke(); }
    if (W && st.layer.flow) ctx.drawImage(trail, 0, 0);
    rect({ cx: 0, cy: 0, L: HS.L, W: HS.W, heading: HS.heading }); ctx.fillStyle = "rgba(111,227,211,.1)"; ctx.fill(); ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = accent; ctx.stroke();
    // labels: you, local gust at the house, wind direction, scale bar
    const h = HRS[st.idx], [hx, hy] = toS(0, 0), gHouse = M.localGust(0, -HS.W / 2 - 3, 2, st.idx);
    ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(hx, hy, 3 * dpr, 0, 7); ctx.fill();
    pill(`${bldName(HS)} · ${Math.round(gHouse)} mph gusts`, hx, hy - HS.W / 2 * s - 14 * dpr, mono(11), ink);
    // compass + scale bar take the 3D view's compass slot, right of the gust legend; on phones (no legend row) the right edge
    const R = 28 * dpr, r = cv.getBoundingClientRect(), lo = cv.parentElement?.querySelector(".lower")?.getBoundingClientRect();
    const row = !!lo?.width, ax = row ? (lo!.right - r.left + 10) * dpr + R : cw - 16 * dpr - R, ay = row ? (lo!.bottom - r.top) * dpr - R : ch * .4;
    const a = (h.dir + 180) * Math.PI / 180;   // arrow points where the wind blows to
    ctx.beginPath(); ctx.arc(ax, ay, R, 0, 7); ctx.fillStyle = "rgba(10,13,17,.72)"; ctx.fill(); ctx.lineWidth = dpr; ctx.strokeStyle = "rgba(255,255,255,.1)"; ctx.stroke();
    for (let k = 0; k < 16; k++) {   // 16 compass dots, north brightest
      const q = k / 16 * 6.2832; ctx.beginPath(); ctx.arc(ax + Math.sin(q) * (R - 6 * dpr), ay - Math.cos(q) * (R - 6 * dpr), (k ? 1 : 1.8) * dpr, 0, 7);
      ctx.fillStyle = !k ? ink : k % 4 ? "rgba(255,255,255,.2)" : "rgba(255,255,255,.45)"; ctx.fill();
    }
    ctx.save(); ctx.translate(ax, ay); ctx.rotate(a); ctx.fillStyle = ink; ctx.beginPath(); ctx.moveTo(0, -15 * dpr); ctx.lineTo(6 * dpr, 9 * dpr); ctx.lineTo(0, 5 * dpr); ctx.lineTo(-6 * dpr, 9 * dpr); ctx.closePath(); ctx.fill(); ctx.restore();
    // beside the compass on the legend row (below it would sit under the timeline), stacked under it on phones
    const wl = `WIND FROM ${Math.round(h.dir)}°`, tx = row ? ax + R + 16 * dpr : cw - 16 * dpr;
    ctx.font = mono(10); ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.fillText("N", ax, ay - R - 8 * dpr);
    ctx.textAlign = row ? "left" : "right"; ctx.fillText(wl, tx, row ? ay + 20 * dpr : ay + R + 18 * dpr); ctx.textAlign = "left";
    const ftPer = s * .3048, bar = [50, 100, 200, 500].find(f => f * ftPer > 70 * dpr) || 500, bx = row ? tx : cw - 16 * dpr - bar * ftPer, by = row ? ay - 4 * dpr : ay + R + 48 * dpr;
    ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + bar * ftPer, by); ctx.moveTo(bx, by - 4 * dpr); ctx.lineTo(bx, by + 4 * dpr); ctx.moveTo(bx + bar * ftPer, by - 4 * dpr); ctx.lineTo(bx + bar * ftPer, by + 4 * dpr); ctx.stroke();
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.fillText(`${bar} FT`, bx + bar * ftPer, by - 9 * dpr); ctx.textAlign = "left";
  }

  function loop(t: number) {
    raf = requestAnimationFrame(loop); const dt = Math.min(.05, (t - last) / 1000); last = t;
    if (!st.active) return;
    if (W && st.layer.flow && NP) particles(dt);
    draw();
  }
  raf = requestAnimationFrame(loop);

  // zoom / pan / hover
  let drag: [number, number] | null = null;
  const onWheel = (e: WheelEvent) => {
    e.preventDefault(); const r = cv.getBoundingClientRect(), [wx, wy] = toW((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    zoom = clamp(zoom * Math.exp(-e.deltaY * .0015), 1, 8); const [nx, ny] = toW((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    panX += wx - nx; panY += wy - ny; tctx.clearRect(0, 0, cw, ch);
  };
  const onDown = (e: PointerEvent) => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); };
  const onUp = () => { drag = null; };
  const onMove = (e: PointerEvent) => {
    const r = cv.getBoundingClientRect();
    if (drag) { panX -= (e.clientX - drag[0]) * dpr / scale(); panY += (e.clientY - drag[1]) * dpr / scale(); drag = [e.clientX, e.clientY]; tctx.clearRect(0, 0, cw, ch); onHover(null, 0, 0); return; }
    const [wx, wy] = toW((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    let best: Tree | null = null, bd = 1e9;
    for (const t of D.trees as Tree[]) { if (t.h < 8) continue; const d = Math.hypot(t.x - wx, t.y - wy); if (d < t.cw / 2 && d < bd) { bd = d; best = t; } }
    onHover(best, e.clientX - r.left, e.clientY - r.top);
  };
  const onLeave = () => onHover(null, 0, 0);
  cv.addEventListener("wheel", onWheel, { passive: false }); cv.addEventListener("pointerdown", onDown); cv.addEventListener("pointerup", onUp);
  cv.addEventListener("pointermove", onMove); cv.addEventListener("pointerleave", onLeave);
  const ro = new ResizeObserver(() => { size(); draw(); }); ro.observe(cv);
  img.onload = () => draw();

  return {
    update(next: Omit<St, "active">, active: boolean) {
      const repaint = next.idx !== st.idx || next.sliceH !== st.sliceH || !st.active;
      st = { ...next, active }; if (!active) return;
      if (!cw) size(); if (repaint) { paintHeat(); tctx.clearRect(0, 0, cw, ch); } draw();
    },
    dispose() {
      cancelAnimationFrame(raf); ro.disconnect();
      cv.removeEventListener("wheel", onWheel); cv.removeEventListener("pointerdown", onDown); cv.removeEventListener("pointerup", onUp);
      cv.removeEventListener("pointermove", onMove); cv.removeEventListener("pointerleave", onLeave);
    },
  };
}
