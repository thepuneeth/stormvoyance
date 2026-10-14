// Storm-track map: embedded dark basemap, 1000 ensemble tracks, official forecast, distance rings, animated storm glyph.
// Report style: white hairlines, dotted rings, the ensemble's positions at the hour as a dot cloud, mono labels, accent = you.
import { css, type Model } from "./model";

export function createStormMap(M: Model, cv: HTMLCanvasElement) {
  const { D, S, fmt } = M, MP = D.map, O = S.official;
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ctx = cv.getContext("2d")!, img = new Image();
  const MONO = css("--f-data") || "monospace", BODY = css("--f-body") || "sans-serif", ACC = css("--accent") || "#6fe3d3", INK = css("--ink") || "#f3f5f7";
  // label over a busy basemap: a dark halo under the glyphs keeps it legible
  const label = (t: string, x: number, y: number) => {
    ctx.save(); ctx.setLineDash([]); ctx.lineWidth = 3 * vw.dpr; ctx.lineJoin = "round"; ctx.strokeStyle = "rgba(6,8,11,.85)"; ctx.strokeText(t, x, y); ctx.fillText(t, x, y); ctx.restore();
  };
  let idx = M.peakIdx, active = false, raf = 0;
  const proj = (la: number, lo: number) => { const n = 256 * 2 ** MP.z; return [(lo + 180) / 360 * n - MP.x0, (1 - Math.asinh(Math.tan(la * Math.PI / 180)) / Math.PI) / 2 * n - MP.y0]; };
  let vw = { s: 1, ox: 0, oy: 0, dpr: 0 };
  function size() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1); if (!r.width) return;
    cv.width = r.width * dpr; cv.height = r.height * dpr; const s = Math.max(cv.width / MP.w, cv.height / MP.h);
    vw = { s, ox: (cv.width - MP.w * s) / 2, oy: (cv.height - MP.h * s) / 2, dpr };
  }
  const P = (la: number, lo: number) => { const [x, y] = proj(la, lo); return [vw.ox + x * vw.s, vw.oy + y * vw.s]; };
  function glyph(x: number, y: number, r: number, a: number, alpha: number) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(a); ctx.globalAlpha = alpha; ctx.strokeStyle = INK; ctx.lineWidth = 1.6 * vw.dpr; ctx.lineCap = "round"; ctx.fillStyle = INK;
    for (let k = 0; k < 2; k++) { ctx.rotate(Math.PI); ctx.beginPath(); ctx.arc(r * .55, 0, r * .9, Math.PI, Math.PI * 1.75); ctx.stroke(); }
    ctx.beginPath(); ctx.arc(0, 0, r * .38, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function draw(t: number) {
    if (!img.complete || !vw.dpr || !active) return; const dpr = vw.dpr, home = P(D.meta.lat, D.meta.lon);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(img, vw.ox, vw.oy, MP.w * vw.s, MP.h * vw.s);
    ctx.fillStyle = "rgba(6,8,11,.25)"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.lineWidth = dpr; ctx.strokeStyle = "rgba(255,255,255,.06)";
    for (const tr of S.tracks) { ctx.beginPath(); tr.lat.forEach((la: number, i: number) => { const [x, y] = P(la, tr.lon[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); }
    ctx.strokeStyle = "rgba(255,255,255,.32)"; ctx.lineWidth = 1.5 * dpr; ctx.lineCap = "round"; ctx.setLineDash([0, 5 * dpr]); ctx.font = `500 ${10 * dpr}px ${MONO}`; ctx.fillStyle = "rgba(255,255,255,.6)";
    for (const mi of [50, 100, 200]) {
      const [, y2] = P(D.meta.lat + mi * 1.609 / 110.54, D.meta.lon), r = home[1] - y2;
      ctx.beginPath(); ctx.arc(home[0], home[1], r, 0, 7); ctx.stroke(); label(mi + " mi", home[0] + 4 * dpr, home[1] - r - 5 * dpr);
    }
    ctx.setLineDash([6 * dpr, 5 * dpr]); ctx.lineCap = "butt"; ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.beginPath(); O.lat.forEach((la: number, i: number) => { const [x, y] = P(la, O.lon[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,.75)";
    for (const p of S.track_pts) if (p.lat != null) {
      const [x, y] = P(p.lat, p.lon); ctx.beginPath(); ctx.arc(x, y, 3 * dpr, 0, 7); ctx.fill();
      label(fmt(p.t, { weekday: "short", hour: "numeric" }) + " · " + Math.round(p.v * 1.151) + " mph", x + 8 * dpr, y + 4 * dpr);
    }
    const k = Math.min(Math.round(idx / 2), S.tracks[0].lat.length - 1); ctx.fillStyle = "rgba(255,255,255,.5)";   // the ensemble at this hour: a dot cloud
    for (const tr of S.tracks) { const [x, y] = P(tr.lat[k], tr.lon[k]); ctx.beginPath(); ctx.arc(x, y, 1.5 * dpr, 0, 7); ctx.fill(); }
    const v = O.vmax[idx], [sx, sy] = P(O.lat[idx], O.lon[idx]);
    ctx.strokeStyle = "rgba(255,255,255,.45)"; ctx.lineWidth = dpr; ctx.setLineDash([0, 4 * dpr]); ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(home[0], home[1]); ctx.lineTo(sx, sy); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = ACC; ctx.globalAlpha = .22; ctx.beginPath(); ctx.arc(home[0], home[1], 14 * dpr, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(home[0], home[1], 5 * dpr, 0, 7); ctx.fill();
    ctx.fillStyle = INK; ctx.font = `500 ${12 * dpr}px ${BODY}`; label("You", home[0] + 18 * dpr, home[1] + 4 * dpr);
    glyph(sx, sy, (14 + v * .35) * dpr, RM ? 0 : -t / 900, v > 0 ? 1 : .35);
    ctx.font = `500 ${11 * dpr}px ${MONO}`; ctx.fillStyle = INK;
    label(`${Math.round(O.dist_mi[idx])} mi · ${v > 0 ? Math.round(v * 1.151) + " mph max wind" : "dissipated"}`, (home[0] + sx) / 2 + 8 * dpr, (home[1] + sy) / 2);
  }
  img.onload = () => { size(); draw(0); }; img.src = MP.img;
  const ro = new ResizeObserver(() => { size(); draw(performance.now()); }); ro.observe(cv);
  const loop = (t: number) => { raf = requestAnimationFrame(loop); if (active && !RM) draw(t); };
  raf = requestAnimationFrame(loop);
  return {
    update(i: number, isActive: boolean) { idx = i; if (isActive && !active) size(); active = isActive; draw(performance.now()); },
    dispose() { cancelAnimationFrame(raf); ro.disconnect(); },
  };
}
