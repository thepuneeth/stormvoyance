import { useEffect, useMemo, useRef, useState } from "react";
import { bearing, bldNoun, clamp, compass, ft, makeModel, monotonePath, pct, quantileAt, RAMP, riskiestTree, type Bundle, type Hour, type Model, type Tree } from "./model";
import { createSiteScene, type FocusTarget as Focus, type Layers, type View } from "./siteScene";
import { createStormMap } from "./stormMap";
import { createPlan } from "./plan2d";
import type { EarthStatus, HouseMode } from "./earth";
import { Icon, type IconName } from "../icons";
import { condition, fmtHour, sentence } from "../weather";
import { createHalftone, type Halftone } from "../halftone";
import { Brand } from "../Logo";
import "./report.css";

type Scene = ReturnType<typeof createSiteScene>;
type Tab = "risks" | "plan" | "method";

const sev = (p: number) => p < .02 ? "ok" : p < .15 ? "warn" : "bad";
const riskCol = (p: number) => `var(--${sev(p)})`;
const TREE_RISKS = ["tree_hit", "limb"], HOUSE_RISKS = ["cosmetic", "minor", "moderate"];
const VIEWS: [View, string, string][] = [["site", "3D", "3D"], ["plan", "2D map", "Map"], ["house", "House loads", "House"], ["storm", "Storm track", "Storm"]];
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const STORM_CLS: Record<string, string> = { HU: "Hurricane", TS: "Tropical Storm", TD: "Tropical Depression", PTC: "Post-tropical cyclone", PT: "Post-tropical cyclone" };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const stormLine = (st: any) => `${STORM_CLS[st.cls] || st.cls} ${st.name} · ${Math.round(st.kt * 1.151)} mph · ${st.mb} mb · advisory ${String(st.adv).replace(/^0+(?=\d)/, "")}`;
// The geocoder returns addresses in capitals; only those get title case, so "McKinney" typed by hand survives.
const tidy = (a: string) => a === a.toUpperCase() ? a.toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase()) : a;

// Eases a displayed number toward its new value (numerals count instead of jumping). Still under reduced motion.
function useTween(v: number, ms = 320) {
  const [x, setX] = useState(v), cur = useRef(v);
  useEffect(() => {
    const a = cur.current;
    if (a === v || reduced()) { cur.current = v; setX(v); return; }
    let raf = 0; const t0 = performance.now();
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms); cur.current = a + (v - a) * (1 - (1 - k) ** 3); setX(cur.current);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [v, ms]);
  return x;
}

export default function Report({ data, onBack, standaloneUrl }: { data: Bundle; onBack: () => void; standaloneUrl: string }) {
  const M = useMemo(() => makeModel(data), [data]);
  const { W, NT } = M;
  const [idx, setIdx] = useState(M.peakIdx);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>("site");
  const [layer, setLayer] = useState<Layers>({ flow: !!W, slice: false, risk: false, rain: true, lidar: false });
  const sliceH = 2;
  const [sheet, setSheet] = useState(false), [tab, setTab] = useState<Tab>("risks"), [openRisk, setOpenRisk] = useState<string | null>(null);
  const [layersOpen, setLayersOpen] = useState(() => !matchMedia("(max-width: 720px)").matches);
  const [cards, setCards] = useState(false), [hint, setHint] = useState(false);
  const [tip, setTip] = useState<{ t: Tree; x: number; y: number } | null>(null);
  const houseMode: HouseMode = "auto";
  const [key, setKey] = useState<string | null>(null), [gmode, setGmode] = useState<"tiles" | "map3d" | "">("");
  const [earth, setEarth] = useState<EarthStatus | null>(null);
  const stage = useRef<HTMLCanvasElement>(null), mapCv = useRef<HTMLCanvasElement>(null), planCv = useRef<HTMLCanvasElement>(null), overlay = useRef<HTMLDivElement>(null), gmap = useRef<HTMLDivElement>(null), needle = useRef<SVGGElement>(null), credits = useRef<HTMLDivElement>(null);
  const scene = useRef<Scene>(), smap = useRef<ReturnType<typeof createStormMap>>(), plan = useRef<ReturnType<typeof createPlan>>();
  const pending = useRef<Focus | null>(null);
  const hasStorm = !!(M.D.map && M.S.official);
  const riskyTree = useMemo(() => riskiestTree(M.D.trees), [M]);

  useEffect(() => { fetch("/api/config").then(r => r.json()).then(c => { setGmode(c.google_mode || ""); setKey(c.google_key || ""); }, () => setKey("")); }, []);
  useEffect(() => {
    if (key == null) return;
    const hover = (t: Tree | null, x: number, y: number) => setTip(t ? { t, x, y } : null);
    const s: Scene = createSiteScene(M, { canvas: stage.current!, overlay: overlay.current!, needle: needle.current!, onHover: hover,
      googleKey: key || undefined, googleMode: gmode, onEarth: setEarth, credits: credits.current!, gmapHost: gmap.current! }, { idx, view, layer, sliceH, houseMode });
    scene.current = s;
    plan.current = createPlan(M, planCv.current!, hover);
    if (hasStorm) smap.current = createStormMap(M, mapCv.current!);
    s.intro?.();                                 // once per scene instance; the scene makes it idempotent
    return () => { scene.current?.dispose(); smap.current?.dispose(); plan.current?.dispose(); scene.current = smap.current = plan.current = undefined; };
  }, [M, key]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    scene.current?.update({ idx, view, layer, sliceH, houseMode }); smap.current?.update(idx, view === "storm"); plan.current?.update({ idx, layer, sliceH }, view === "plan");
    if (pending.current) { scene.current?.focus?.(pending.current); pending.current = null; }   // after update(), so it wins over the view's own camera move
  }, [idx, view, layer, sliceH, houseMode, key]);

  useEffect(() => { if (!playing) return; const iv = setInterval(() => setIdx(i => Math.min(i + 1, NT - 1)), 700); return () => clearInterval(iv); }, [playing, NT]);
  useEffect(() => { if (playing && idx >= NT - 1) setPlaying(false); }, [idx, playing, NT]);
  const togglePlay = () => { if (!playing && idx >= NT - 1) setIdx(0); setPlaying(p => !p); };
  const scrub = (i: number) => { setPlaying(false); setIdx(clamp(i, 0, NT - 1)); };

  const go = (v: View) => { setView(v); if (v === "plan" && W) setLayer(l => ({ ...l, slice: true })); };
  const flyTo = (f: Focus, v: View) => { if (v === view) scene.current?.focus?.(f); else { pending.current = f; go(v); } };
  const showTree = () => { setLayer(l => ({ ...l, risk: true })); if (riskyTree >= 0) flyTo({ kind: "tree", i: riskyTree }, "site"); };
  const details = (k: string | null, t: Tab = "risks") => { setTab(t); setOpenRisk(k); setSheet(true); };
  const closeSheet = () => { setSheet(false); setOpenRisk(null); };   // the sheet stays mounted: clear it so picking the same risk again re-opens it
  const pickRisk = (k: string) => {
    setCards(false);
    if (TREE_RISKS.includes(k) && riskyTree >= 0) showTree();
    else if (HOUSE_RISKS.includes(k)) flyTo({ kind: "house" }, "house");
    else details(k);
  };
  const toggle = (k: keyof Layers) => setLayer(l => ({ ...l, [k]: !l[k] }));

  // Keyboard: Left/Right hour, Space play, 1-4 views, D details, ? shortcuts, Esc closes the top-most thing.
  const onKey = useRef<(e: KeyboardEvent) => void>();
  onKey.current = e => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
    const t = e.target instanceof Element ? e.target : null;
    if (t?.closest("input, textarea, select, [contenteditable='true']")) return;
    const k = e.key;
    if (k === "Escape") { if (hint) setHint(false); else if (sheet) setSheet(false); else setCards(false); }
    else if (k === "ArrowLeft" || k === "ArrowRight") { if (t?.closest("[role=tablist]")) return; e.preventDefault(); scrub(idx + (k === "ArrowRight" ? 1 : -1)); }
    else if (k === " ") { if (t?.closest("button, a, summary")) return; e.preventDefault(); togglePlay(); }
    else if (/^[1-4]$/.test(k)) { const v = VIEWS[+k - 1][0]; if (v !== "storm" || hasStorm) go(v); }
    else if (k === "d" || k === "D") setSheet(s => !s);
    else if (k === "?") setHint(h => !h);
  };
  useEffect(() => { const k = (e: KeyboardEvent) => onKey.current?.(e); addEventListener("keydown", k); return () => removeEventListener("keydown", k); }, []);

  return (
    <div className="rv" data-view={view} data-g3d={earth?.phase === "map3d" || undefined}>
      <div className="stagebox">
        <div ref={gmap} className="gmap" hidden />
        <canvas ref={stage} className="stage" aria-label="3D model of the house, neighbouring buildings, LiDAR-measured trees and the simulated wind" />
        <canvas ref={planCv} className="plan" aria-label="Top-down map of the lot with the simulated wind" />
        <canvas ref={mapCv} className="map" aria-label="Storm track map" />
        <div ref={overlay} className="overlay" />

        <Hero M={M} idx={idx} onBack={onBack} />

        <div ref={credits} className="credits" hidden={!earth || earth.phase === "error" || earth.phase === "map3d" || view === "storm" || view === "plan"} />

        <aside className="side" data-cards={cards} aria-label="Views, layers and findings">
          <div className="nav">
            <div className="seg glass" role="group" aria-label="View">
              {VIEWS.filter(([v]) => v !== "storm" || hasStorm).map(([v, l, s]) => v === "house" && M.D.house.cls && M.D.house.cls !== "WSF" ? [v, "Wind loads", "Loads"] as const : [v, l, s] as const).map(([v, l, s], i) =>
                <button key={v} aria-pressed={view === v} onClick={() => go(v)} title={`${l} (${i + 1})`} aria-label={l}><span className="l">{l}</span><span className="s">{s}</span></button>)}
            </div>
            <button className="round glass lay" aria-pressed={layersOpen} aria-controls="rv-layers" onClick={() => setLayersOpen(o => !o)} aria-label="Layers" title="Layers">
              <Icon name="layers" size={18} /></button>
            <button className="round glass cardsbtn" aria-expanded={cards} onClick={() => setCards(c => !c)} aria-label="Wind and risks" title="Wind and risks">
              <Icon name={cards ? "close" : "info"} size={18} /></button>
          </div>
          {layersOpen && <div className="chips glass" id="rv-layers" role="group" aria-label="Layers">
            {W && <Chip on={layer.flow} icon="wind" onClick={() => toggle("flow")}>Airflow</Chip>}
            {view !== "plan" && <Chip on={layer.rain} icon="rain" onClick={() => toggle("rain")}>Rain</Chip>}
          </div>}
          <div className="cards">
            <WindCard M={M} idx={idx} />
            <RiskCard M={M} onPick={pickRisk} onAll={() => details(null)} />
          </div>
        </aside>

        <div className="lower">
          <Legend />
          <div className="compass glass" aria-hidden="true">
            <svg width="40" height="40" viewBox="-20 -20 40 40"><g ref={needle}><path d="M0-15 4.5 2 0-1-4.5 2z" className="needle" />
              <text y="13" x="-3.2" fontSize="8" className="n" transform="rotate(180 0 10)">N</text></g></svg>
          </div>
        </div>

        <Timeline M={M} idx={idx} onScrub={scrub} playing={playing} onPlay={togglePlay} onHint={() => setHint(h => !h)} />
        {hint && <Shortcuts hasStorm={hasStorm} onClose={() => setHint(false)} />}
        {tip && view !== "storm" && <TreeTip M={M} t={tip.t} x={tip.x} y={tip.y} idx={idx} />}
        <Sheet M={M} open={sheet} tab={tab} setTab={setTab} openRisk={openRisk} onClose={closeSheet} standaloneUrl={standaloneUrl}
          onRisk={(key, open) => {
            if (key === "tree_hit" || key === "limb") { setLayer(l => ({ ...l, risk: open })); if (open && riskyTree >= 0) flyTo({ kind: "tree", i: riskyTree }, "site"); }
            if ((key === "cosmetic" || key === "minor") && open) flyTo({ kind: "house" }, "house");
          }} />
      </div>
    </div>
  );
}

function Chip({ on, icon, onClick, title, children }: { on: boolean; icon: IconName | "dots"; onClick: () => void; title?: string; children: React.ReactNode }) {
  return (
    <button className="chip" aria-pressed={on} onClick={onClick} title={title}>
      {icon === "dots"   // LiDAR returns: a little point cloud
        ? <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">{[[5, 6], [12, 4], [19, 7], [8, 12], [15, 11], [4, 18], [11, 19], [18, 17]].map(([x, y]) => <circle key={x * 31 + y} cx={x} cy={y} r="1.9" />)}</svg>
        : <Icon name={icon} size={14} />}
      {children}
    </button>
  );
}

function Hero({ M, idx, onBack }: { M: Model; idx: number; onBack: () => void }) {
  const h = M.HRS[idx], c = condition(h), g = useTween(h.gust[2]), m = M.D.meta;
  const [street, ...rest] = tidy(m.address || "").split(", ");
  const bits = [h.roof && `${Math.round(h.roof[1])} mph at your roof`, `from ${compass(h.dir)}`].filter(Boolean);
  return (
    <header className="hero">
      <div className="top">
        <button className="round glass" onClick={onBack} aria-label="New address" title="New address"><Icon name="arrow-left" size={18} /></button>
        <Brand />
      </div>
      {m.storm && <p className="lbl storm"><i />{stormLine(m.storm)}</p>}
      <p className="cond"><Icon name={c.icon} size={20} /><span>{c.label}</span><span className="when">{M.hourLbl(h.t)}</span></p>
      <div className="big num">{Math.round(g)}<small>mph</small></div>
      <p className="line">{bits.join(" · ")}<Arrow dir={h.dir} />{h.rain[0] >= .01 && <> · rain {h.rain[0].toFixed(2)} in/hr</>}</p>
      <p className="say">{sentence(M.HRS, idx, m.tz)}</p>
      {street && <p className="addr"><b>{street}</b>{rest.length ? `, ${rest.join(", ")}` : ""}<span>Generated {m.generated}</span></p>}
    </header>
  );
}

// Points where the wind is going (it comes *from* dir), north up.
const Arrow = ({ dir }: { dir: number }) => (
  <svg className="arrow" width="13" height="13" viewBox="-6.5 -6.5 13 13" aria-hidden="true" style={{ transform: `rotate(${dir + 180}deg)` }}>
    <path d="M0 5.5V-5.5M-3.6-1.9 0-5.5l3.6 3.6" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

function WindCard({ M, idx }: { M: Model; idx: number }) {
  const { HRS, NT } = M, h = HRS[idx], g = useTween(h.gust[2]);
  const n = Math.min(24, NT), a = clamp(idx - n + 1, 0, NT - n), win = HRS.slice(a, a + n);
  const gmax = useMemo(() => Math.max(10, ...HRS.map((x: Hour) => x.gust[2])), [HRS]);
  const xmax = useMemo(() => Math.max(30, Math.ceil(Math.max(...HRS.map((x: Hour) => x.gust[4])) / 10) * 10), [HRS]);
  // 24 h dot matrix: one column per hour, dots filled up to the median gust; the line is the NWS sustained wind
  const VW = 288, R = 6, DY = 8.5, VH = R * DY + 4, cw = VW / n, rr = Math.min(cw * .3, 3);
  const lvl = (v: number) => v / gmax * R, yAt = (r: number) => VH - 5 - r * DY;
  const sus = win.every((x: Hour) => x.sust != null) ? monotonePath(win.map((x: Hour, i: number) => [cw * (i + .5), Math.min(VH - 1, yAt(lvl(x.sust) - 1))])) : "";
  // spread of the simulated storms at this hour: 19 evenly spaced quantiles (5%..95%), stacked where they land on the same spot
  const D = 7, X = (v: number) => 6 + v / xmax * (VW - 12), stack: Record<number, number> = {};
  const dots = Array.from({ length: 19 }, (_, k) => { const b = Math.round(X(quantileAt(h.gust, .05 + .05 * k)) / D); stack[b] = (stack[b] || 0) + 1; return [b * D, stack[b]]; });
  const SH = Math.max(30, Math.max(...Object.values(stack)) * D + 4);
  const mem = M.D.meta.members ? M.D.meta.members.toLocaleString() : "";
  return (
    <section className="card glass wind" aria-label="Wind">
      <header><span className="lbl">Wind · {M.hourLbl(h.t)}</span><span className="dir"><Arrow dir={h.dir} />from {compass(h.dir)}</span></header>
      <div className="now">
        <div className="num">{Math.round(g)}<small>mph</small></div>
        <dl>
          {h.sust != null && <div><dt>{sus && <i className="kl" />}Sustained</dt><dd>{Math.round(h.sust)} mph</dd></div>}
          <div><dt>Range 5–95%</dt><dd>{Math.round(h.gust[0])}–{Math.round(h.gust[4])}</dd></div>
        </dl>
      </div>
      <svg className="matrix" viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={`Median gust for ${n} hours from ${M.hourLbl(win[0].t)}`}>
        {win.map((x: Hour, i: number) => {
          const on = Math.max(1, Math.round(lvl(x.gust[2]))), sel = a + i === idx;
          return Array.from({ length: R }, (_, r) => <circle key={i * R + r} cx={cw * (i + .5)} cy={yAt(r)} r={rr} className={r >= on ? "off" : sel ? "sel" : "on"} />);
        })}
        {sus && <path d={sus} className="sus" />}
      </svg>
      <div className="axis">{[0, Math.floor((n - 1) / 2), n - 1].map(i => <span key={i}>{fmtHour(win[i].t, M.D.meta.tz)}</span>)}</div>
      <div className="spread">
        <span className="lbl">{mem ? `${mem} storms` : "Simulated storms"}, middle 90%</span>
        <svg viewBox={`0 0 ${VW} ${SH}`} role="img" aria-label={`Gusts from ${Math.round(h.gust[0])} to ${Math.round(h.gust[4])} mph, median ${Math.round(h.gust[2])}`}>
          <line x1="6" x2={VW - 6} y1={SH - .5} y2={SH - .5} className="base" />
          {dots.map(([x, s], k) => <circle key={k} r="2.8" className={k === 9 ? "med" : ""} style={{ transform: `translate(${x}px, ${SH - 4 - (s - .5) * D}px)` }} />)}
        </svg>
        <div className="axis"><span>0</span><span>{xmax / 2}</span><span>{xmax} mph</span></div>
      </div>
    </section>
  );
}

function RiskCard({ M, onPick, onAll }: { M: Model; onPick: (k: string) => void; onAll: () => void }) {
  const top = useMemo(() => [...M.D.risks].sort((a, b) => (b.p || 0) - (a.p || 0)).slice(0, 4), [M]);
  return (
    <section className="card glass risks" aria-label="Top risks">
      <header><span className="lbl">Risks</span><span className="lbl">next {M.NT} h</span></header>
      <ul>{top.map((r: { key: string; title: string; p: number }) => {
        const tree = TREE_RISKS.includes(r.key), house = HOUSE_RISKS.includes(r.key);
        return <li key={r.key}><button onClick={() => onPick(r.key)} title={tree ? "Show the riskiest tree" : house ? `Show the loads on the ${bldNoun(M.D.house)}` : "Open the details"}>
          <i className="sev" style={{ background: riskCol(r.p || 0) }} /><span className="t">{r.title}</span><span className="p">{pct(r.p)}</span>
          <Icon name={tree ? "tree" : house ? "house" : "arrow-right"} size={15} /></button></li>;
      })}</ul>
      <button className="all" onClick={onAll}>All details<kbd>D</kbd></button>
    </section>
  );
}

function Legend() {
  const mx = 70, bg = `linear-gradient(90deg,${RAMP.filter(r => r[0] <= mx).map(([v, c]) => `rgb(${c}) ${v / mx * 100}%`).join(",")})`;
  return (
    <div className="legend glass"><span className="lbl">Gust speed, mph</span><div className="ramp" style={{ background: bg }} />
      <div className="ticks">{[0, 10, 20, 30, 40, 50, 60, 70].map(v => <span key={v}>{v}</span>)}</div></div>
  );
}

function Shortcuts({ hasStorm, onClose }: { hasStorm: boolean; onClose: () => void }) {
  const rows: [string, string][] = [["← →", "Previous or next hour"], ["Space", "Play or pause"], [hasStorm ? "1–4" : "1–3", "Switch view"], ["D", "All details"], ["Esc", "Close"], ["?", "Show or hide this"]];
  const el = useRef<HTMLDivElement>(null), close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {   // any press outside closes it (the ? button toggles it itself)
    const f = (e: PointerEvent) => { const t = e.target as Element; if (!el.current?.contains(t) && !t.closest?.(".qk")) close.current(); };
    addEventListener("pointerdown", f, true); return () => removeEventListener("pointerdown", f, true);
  }, []);
  return (
    <div ref={el} className="hint glass" role="dialog" aria-label="Keyboard shortcuts">
      <header><span className="lbl">Keyboard</span><button className="x" onClick={onClose} aria-label="Close"><Icon name="close" size={14} /></button></header>
      <dl>{rows.map(([k, v]) => <div key={v}><dt><kbd>{k}</kbd></dt><dd>{v}</dd></div>)}</dl>
    </div>
  );
}

function TreeTip({ M, t, x, y, idx }: { M: Model; t: Tree; x: number; y: number; idx: number }) {
  const HR = Math.sqrt(M.D.house.L * M.D.house.W / Math.PI), box = document.querySelector(".rv .stagebox")!.getBoundingClientRect();
  const rows: [string, string][] = [
    [`Distance to ${bldNoun(M.D.house)}`, `${ft(Math.max(0, Math.hypot(t.x, t.y) - HR))} ft`],
    ["Trunk (from LiDAR size)", t.dbh_cm ? `≈ ${Math.round(t.dbh_cm / 2.54)} in` : "n/a"],
    ["Gust at its crown, this hour", `${Math.round(M.localGust(t.x, t.y, t.h + 1, idx))} mph`],
    ["Storm peak (95%)", t.gust95 ? `${Math.round(t.gust95)} mph` : "n/a"],
    [`Gust to ${t.mode === "uproot" ? "uproot" : "snap"} it`, t.crit_mph ? `≈ ${t.crit_mph} mph` : "n/a"],
    ["Chance it fails", pct(t.p_fail)], [`Chance it hits the ${bldNoun(M.D.house)}`, pct(t.p_hit)],
  ];
  return (
    <div className="tip glass" style={{ left: clamp(x + 16, 8, box.width - 268), top: clamp(y + 16, 8, box.height - 230) }}>
      <b><Icon name="tree" size={15} />Tree, {ft(t.h)} ft tall</b>
      <table><tbody>{rows.map(([a, b]) => <tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody></table>
    </div>
  );
}

// Timeline after a week row: the median gust as a smooth line threading glass nodes, the 5-95% range as a halftone band.
const TH = 134, CT = 54, CH = 58, RB = 130, PAD = 18;
function Timeline({ M, idx, onScrub, playing, onPlay, onHint }: { M: Model; idx: number; onScrub: (i: number) => void; playing: boolean; onPlay: () => void; onHint: () => void }) {
  const { HRS, NT, fmt } = M, box = useRef<HTMLDivElement>(null), [w, setW] = useState(0), [hov, setHov] = useState<number | null>(null);
  useEffect(() => { const el = box.current!, ro = new ResizeObserver(() => setW(el.clientWidth)); ro.observe(el); return () => ro.disconnect(); }, []);
  const G = useMemo(() => {
    if (w < 60) return null;
    const X = (i: number) => PAD + (NT > 1 ? i / (NT - 1) : .5) * (w - 2 * PAD);
    const ymax = Math.max(30, ...HRS.map((h: Hour) => h.gust[4])) * 1.06, Y = (v: number) => CT + (1 - clamp(v / ymax, 0, 1)) * CH;
    const pts = (f: (h: Hour) => number) => HRS.map((h: Hour, i: number) => [X(i), Y(f(h))] as [number, number]);
    const lh = HRS.map((h: Hour) => +fmt(h.t, { hour: "numeric", hourCycle: "h23" }) % 24);
    const per = (w - 2 * PAD) / Math.max(1, NT - 1), step = [3, 6, 12].find(s => per * s >= 44) ?? 24;
    const nodes = HRS.map((_: Hour, i: number) => i).filter((i: number) => lh[i] % step === 0);
    return {
      X, Y, nodes, lh, rmax: Math.max(.05, ...HRS.map((h: Hour) => h.rain[0])),
      grid: [10, 20, 30, 40, 50, 60, 70, 80, 100].filter(v => v < ymax && (ymax < 50 || v % 20 === 0)),
      mid: monotonePath(pts(h => h.gust[2])), band: monotonePath(pts(h => h.gust[4])) + "L" + monotonePath(pts(h => h.gust[0]).reverse()).slice(1) + "Z",
      roof: HRS[0].roof ? monotonePath(pts(h => h.roof[1])) : "",
    };
  }, [w, M]);   // eslint-disable-line react-hooks/exhaustive-deps
  const pick = (e: React.PointerEvent) => { const r = box.current!.getBoundingClientRect(); return clamp(Math.round((e.clientX - r.left - PAD) / (r.width - 2 * PAD) * (NT - 1)), 0, NT - 1); };
  const h = HRS[idx], c = condition(h);
  const label = (i: number) => G!.lh[i] === 0 ? fmt(HRS[i].t, { weekday: "short" }) : fmtHour(HRS[i].t, M.D.meta.tz);
  return (
    <div className="tl">
      <button className="round glass play" onClick={onPlay} aria-label={playing ? "Pause" : "Play the forecast"} title={playing ? "Pause (Space)" : "Play (Space)"}>
        <Icon name={playing ? "pause" : "play"} size={18} /></button>
      <div ref={box} className="track" role="slider" tabIndex={0} aria-label="Forecast hour" aria-valuemin={0} aria-valuemax={NT - 1} aria-valuenow={idx}
        aria-valuetext={`${M.hourLbl(h.t)}, ${c.label.toLowerCase()}, gusts ${Math.round(h.gust[2])} mph`}
        onPointerDown={e => { box.current!.setPointerCapture(e.pointerId); onScrub(pick(e)); }}
        onPointerMove={e => { const i = pick(e); if (e.buttons) onScrub(i); setHov(i); }} onPointerLeave={() => setHov(null)}
        onKeyDown={e => { if (e.key === "Home" || e.key === "End") { e.preventDefault(); onScrub(e.key === "Home" ? 0 : NT - 1); } }}>
        {G && <svg width={w} height={TH} aria-hidden="true">
          <defs>
            <pattern id="rv-ht" width="4" height="4" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r=".85" fill="#fff" /></pattern>
            <radialGradient id="rv-glow"><stop offset="0" style={{ stopColor: "var(--accent)", stopOpacity: .5 }} /><stop offset="1" style={{ stopColor: "var(--accent)", stopOpacity: 0 }} /></radialGradient>
          </defs>
          {G.grid.map(v => <g key={v} className="grid"><line x1={PAD} x2={w - PAD} y1={G.Y(v)} y2={G.Y(v)} /><text x={w - 2} y={G.Y(v) + 3}>{v}</text></g>)}
          {G.nodes.filter(i => i > 0 && G.lh[i] === 0).map(i => <line key={i} className="day" x1={G.X(i)} x2={G.X(i)} y1={CT - 10} y2={RB + 2} />)}
          <path className="band" d={G.band} />
          {G.roof && <path className="roof" d={G.roof} />}
          <path className="mid" d={G.mid} pathLength={1} />
          {HRS.map((x: Hour, i: number) => {
            const k = x.rain[0] < .005 ? 0 : clamp(Math.ceil(x.rain[0] / G.rmax * 4), 1, 4);
            return Array.from({ length: k }, (_, j) => <circle key={i * 4 + j} className="rain" cx={G.X(i)} cy={RB - j * 4.5} r={1.5} />);
          })}
          {hov != null && hov !== idx && <g className="ghost"><line x1={G.X(hov)} x2={G.X(hov)} y1={CT - 10} y2={RB} /><circle cx={G.X(hov)} cy={G.Y(HRS[hov].gust[2])} r={3} /></g>}
          <line className="hair" x1={0} x2={0} y1={CT - 12} y2={RB + 2} style={{ transform: `translateX(${G.X(idx)}px)` }} />
          {G.nodes.map((i, n) => (
            <g key={i}>
              <g className="lab" data-on={i === idx || undefined} style={{ transform: `translateX(${G.X(i)}px)` }}>
                <text className="hr" y={11}>{label(i)}</text><text className="v" y={36}>{Math.round(HRS[i].gust[2])}</text>
              </g>
              <g className="nd" data-on={i === idx || undefined} style={{ transform: `translate(${G.X(i)}px, ${G.Y(HRS[i].gust[2])}px)` }}>
                <g className="node" style={{ animationDelay: `${300 + n * 35}ms` }}>
                  <circle r={13} /><g transform="translate(-8 -8)"><Icon name={condition(HRS[i]).icon} size={16} /></g>
                </g>
              </g>
            </g>
          ))}
          <g className="sel" style={{ transform: `translate(${G.X(idx)}px, ${G.Y(h.gust[2])}px)` }}>
            <circle r={28} fill="url(#rv-glow)" /><circle r={14} className="ring" /><g transform="translate(-8 -8)"><Icon name={c.icon} size={16} /></g>
          </g>
        </svg>}
      </div>
      <div className="tl-key">
        <span><i className="k mid" />Median gust</span><span><i className="k band" />5–95%</span>
        {G?.roof && <span><i className="k roof" />At your roof</span>}
        <span><i className="k rain" />Rain</span>
        <button className="qk" onClick={onHint} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">?</button>
      </div>
    </div>
  );
}

function Sheet({ M, open, tab, setTab, openRisk, onClose, onRisk, standaloneUrl }: { M: Model; open: boolean; tab: Tab; setTab: (t: Tab) => void; openRisk: string | null;
  onClose: () => void; onRisk: (key: string, open: boolean) => void; standaloneUrl: string }) {
  const el = useRef<HTMLElement>(null);
  useEffect(() => {
    const s = el.current!; s.inert = !open;
    if (open) s.querySelector<HTMLElement>("[role=tab][aria-selected=true]")?.focus({ preventScroll: true });
  }, [open]);
  return (
    <aside ref={el} className="sheet glass" id="rv-sheet" aria-label="Findings" data-open={open} aria-hidden={!open}>
      <Summary M={M} open={open} />
      <div className="tabs" role="tablist">
        {([["risks", "Risks"], ["plan", "What to do"], ["method", "How it works"]] as const).map(([k, l]) =>
          <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>)}
        <button className="round x" aria-label="Close" title="Close (Esc)" onClick={onClose}><Icon name="close" size={16} /></button>
      </div>
      <div className="pane" role="tabpanel">
        {tab === "risks" && <Risks M={M} onRisk={onRisk} focus={openRisk} />}
        {tab === "plan" && <Plan M={M} />}
        {tab === "method" && <Method M={M} standaloneUrl={standaloneUrl} />}
      </div>
    </aside>
  );
}

function Summary({ M, open }: { M: Model; open: boolean }) {
  const { D, S, HRS, fmt } = M, m = D.meta, st = m.storm, P = S.probs, pk = S.peak;
  // like with like: both peaks are the highest hourly median (open terrain vs. the CFD roof ring); no CFD, no roof figure
  const argmax = (v: number[]) => v.reduce((a, x, i) => x > v[a] ? i : a, 0);
  const openS = HRS.map(h => h.gust[2]), roofS = HRS[0].roof ? HRS.map(h => h.roof[1]) : null;
  const openPk = Math.max(...openS), roofPk = roofS ? Math.max(...roofS) : null, pkI = argmax(roofS || openS);
  const nn = bldNoun(D.house);
  // the headline names what drives it: [chance, wording when possible, wording when likely]
  const drivers: [number, string, string][] = [
    [P.limb, "Branches may come down. Clear the yard tonight.", "Branches likely down. Move cars and clear the yard."],
    [P.power, "Power cuts possible. Charge up tonight.", "Power cuts likely. Charge up and plan for an outage."],
    [P.tree_hit * 4, `A tree could reach the ${nn}. Prepare tonight.`, `A tree may hit the ${nn}. Prepare now.`],
    [P.minor * 6, "Some damage possible. Prepare tonight.", "Damage likely. Prepare now."],
    [Math.max(P.flash, P.creek || 0), "Flooding possible nearby. Prepare tonight.", "Flooding likely nearby. Prepare now."]];
  const [worst, maybe, likely] = drivers.reduce((a, d) => d[0] > a[0] ? d : a);
  const word = openPk >= 25 ? "Windy" : openPk >= 12 ? "Breezy" : "Calm";   // the same bands as weather.ts condition()
  const verdict = worst < .05 ? (word === "Windy" ? `Windy, but low risk to the ${nn}.` : `${word}. Low risk to the ${nn}.`)
    : worst < .3 ? maybe : worst < .6 || P.minor < .1 ? likely : "High impact expected.";
  const top = [...D.risks].sort((a, b) => b.p - a.p)[0];
  let close = null;
  if (S.official) {
    const d: number[] = S.official.dist_mi, i = d.indexOf(Math.min(...d));
    close = <>{st.name}'s center passes about <b>{Math.round(d[i])} mi {compass(bearing(m.lat, m.lon, S.official.lat[i], S.official.lon[i]))}</b> of you around {fmt(HRS[i].t, { weekday: "short", hour: "numeric" })}. </>;
  }
  const figs: [string | number, string, string][] = [[Math.round(openPk), "mph", "peak gust, open"], ...(roofPk != null ? [[Math.round(roofPk), "mph", "at your roof"] as [number, string, string]] : []),
    [S.rain_total.p50.toFixed(2), "in", "rain"], [pct(P.power), "", "lose power"]];
  // halftone sky behind the verdict, as heavy as the findings say; lightning only if thunder is actually forecast
  const cv = useRef<HTMLCanvasElement>(null), ht = useRef<Halftone>();
  const thunder = HRS.some(h => (h.thunder || 0) >= 30), storm = clamp(Math.max(worst / .6, (pk.p50 - 15) / 45), .06, thunder ? 1 : .5);
  useEffect(() => { const h = createHalftone(cv.current!, { cell: 7, seed: 5 }); h.setStorm(storm); ht.current = h; return () => { h.dispose(); ht.current = undefined; }; }, [storm]);
  useEffect(() => { ht.current?.setPaused(!open); if (open && thunder) ht.current?.flash(); }, [open, thunder]);
  return (
    <section className="hud">
      <canvas ref={cv} className="halftone" aria-hidden="true" />
      <div className="storm-chip"><i /><span>{st ? stormLine(st) : "No tropical cyclone nearby · NWS forecast"}</span></div>
      <h1>{verdict}</h1>
      <div className="figs">{figs.map(([v, u, l]) => <div className="fig" key={l}><div className="v num">{v}{u && <small>{u}</small>}</div><div className="lbl">{l}</div></div>)}</div>
      <p className="summary">{close}Gusts {roofPk != null ? "at your roofline" : "in the open"} peak near <b>{Math.round(roofPk ?? openPk)} mph</b> around {fmt(HRS[pkI].t, { weekday: "short", hour: "numeric" })}. Biggest concern: <b>{top.title.toLowerCase()}</b> ({pct(top.p)}).</p>
    </section>
  );
}

function Risks({ M, onRisk, focus }: { M: Model; onRisk: (key: string, open: boolean) => void; focus: string | null }) {
  const order = useMemo(() => [...M.D.risks].sort((a, b) => b.p - a.p), [M]);
  const [open, setOpen] = useState<string | null>(focus);
  useEffect(() => { if (focus) setOpen(focus); }, [focus]);
  return (
    <>
      <p className="lbl" style={{ margin: "14px 0 4px" }}>Chance in the next {M.NT} hours · {M.D.meta.members.toLocaleString()} simulated storms</p>
      {order.map(r => (
        <div className="risk" key={r.key}>
          <button aria-expanded={open === r.key} onClick={() => { const o = open !== r.key; setOpen(o ? r.key : null); onRisk(r.key, o); }}>
            <span className="t"><i className="sev" style={{ background: riskCol(r.p) }} />{r.title}</span><span className="p">{pct(r.p)}</span>
            <span className="bar"><i style={{ width: `${Math.max(1.5, Math.min(100, r.p * 100))}%`, background: riskCol(r.p) }} /></span>
          </button>
          {open === r.key && <div className="more"><p>{r.plain}</p>
            <table className="calc"><tbody>{(r.calc || []).map(([a, b]: string[]) => <tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody></table><p>{r.how}</p></div>}
        </div>
      ))}
      <p className="note" style={{ marginTop: 12 }}>Open a risk to see how it was calculated. Hover a tree in the scene for its own numbers.</p>
    </>
  );
}

function Plan({ M }: { M: Model }) {
  const { D, S, HRS, fmt, hourLbl, peakIdx } = M, P = S.probs, pk = S.peak, HR = Math.sqrt(D.house.L * D.house.W / Math.PI);
  const ids = HRS.map((_, i) => i).filter(i => HRS[i].gust[2] >= pk.p50 * .85);
  const win = ids.length ? `${hourLbl(HRS[ids[0]].t)} to ${fmt(HRS[ids[ids.length - 1]].t, { hour: "numeric" })}` : hourLbl(pk.hour);
  const upwind = compass(HRS[peakIdx].dir), near = D.trees.filter((t: Tree) => t.h >= 8 && Math.hypot(t.x, t.y) < 25 + HR).length;
  const before: string[] = [], during: string[] = [], after: string[] = [];
  if (pk.p50 > 25) before.push("Bring in or tie down patio furniture, trash bins and anything that can blow around.");
  if (P.power > .02) before.push(`Charge phones and battery packs. Chance of losing power: ${pct(P.power)}.`);
  if (S.rain_total.p50 > .3 || P.ponding > .05) before.push("Clear gutters and downspouts, and point them away from the foundation.");
  if (near) before.push(`Park cars away from the ${near} trees taller than 26 ft within 80 ft of the ${bldNoun(D.house)}.`);
  if (P.tree_hit > .01) before.push(`Sleep in a room away from the tall trees on the ${upwind} side.`);
  before.push("Turn on Wireless Emergency Alerts on your phone.");
  during.push(`Strongest gusts: ${win}, from the ${upwind}.`);
  if (HRS.some(h => (h.thunder || 0) > 20)) during.push("Rain bands can spin up brief tornadoes with little warning. Know your interior room on the lowest floor.");
  during.push("Stay away from windows on the windward side during the strongest gusts.");
  if (P.flash > .02) during.push("Never drive through water over a road. Six inches can float a car.");
  after.push("Soaked ground keeps trees vulnerable for a day or two. Look up before walking under them.");
  after.push("Treat every downed line as live and report it to your utility.");
  if (P.power > .02) after.push("Run any generator outdoors, at least 20 ft from doors and windows.");
  after.push(bldNoun(D.house) === "house" ? "Walk around the house: check roof corners and the ridge for lifted shingles, and look for loose soffit or gutters."
    : "Walk around the building: check roof edges and corners, flashing, cladding panels and windows on the windward side.");
  if (D.site.river) after.push(`Check ${D.site.river.name}. Creeks crest hours after the rain ends.`);
  const sec = (t: string, L: string[]) => <><h3 className="sub">{t}</h3><ul>{L.map(x => <li key={x}>{x}</li>)}</ul></>;
  return (
    <>
      {sec("Before the wind picks up", before)}{sec(`During, ${win}`, during)}{sec("After it passes", after)}
      <h3 className="sub">Official alerts</h3>
      {D.alerts.length ? D.alerts.map((a: { event: string; headline: string }, i: number) => <div className="alert" key={i}><b>{a.event}</b><br />{a.headline}</div>)
        : <p className="note">No National Weather Service alerts for this point right now.</p>}
      <p className="note">This is a model, not an official forecast. Follow NWS {D.meta.office} and local officials.</p>
    </>
  );
}

function Method({ M, standaloneUrl }: { M: Model; standaloneUrl: string }) {
  const { D, W } = M, s = D.site, H = D.house, L = s.lidar, k = D.knobs, sec0 = W && W.sectors[0];
  const steps: [string, string][] = [
    ["Finds your roof", `Building outline (${H.src}): ${ft(H.L)} × ${ft(H.W)} ft, ${H.stories} storeys, eaves ${ft(H.eave_m)} ft, ridge ${ft(H.ridge_m)} ft${H.roof_src ? (/^USGS/.test(H.roof_src) ? ` (measured, ${H.roof_src})` : ` (${H.roof_src.replace(/^assumed \((.*)\)$/, "assumed: $1; the LiDAR survey predates the building or misses it")})`) : ""}.`],
    ["Rebuilds the ground", `${s.dem} bare-earth elevation, ${D.scene.terrain.n * 2} m square around the house. Lot elevation ${s.elev_ft ? Math.round(s.elev_ft) + " ft" : "n/a"}; relief within a mile ${s.terrain_relief ? ft(s.terrain_relief) + " ft" : "n/a"}, checked for hill speed-up with the ASCE 7 topographic factor.`],
    ["Measures every tree", L ? `${L.points.toLocaleString()} airborne LiDAR returns from USGS survey ${L.survey.replace(/_/g, " ")}. Treetops are found with a height-dependent search window (Popescu & Wynne 2004). Trees cut down since ${L.year} are removed by checking current imagery. ${D.trees.filter((t: Tree) => t.h >= 8).length} trees over 26 ft are modelled one by one.` : "No LiDAR survey covers this point, so tree heights come from imagery and a typical height."],
    ["Simulates the wind around your lot", sec0 ? `3D lattice-Boltzmann large-eddy simulation (D3Q19, regularized BGK collision, Smagorinsky subgrid model, half-way bounce-back walls; GPU), the method behind fast urban-wind tools. Grid ${sec0.lattice.join(" × ")} cells of ${sec0.dx} m, ${W!.steps.toLocaleString()} time steps for each forecast wind direction (${W!.sectors.map(x => compass(x.dir)).join(", ")}). Buildings are solid walls; the ground carries a log-law surface-stress wall model; tree crowns are porous drag (leaf area index ${s.lai ?? k.tree_LAI}${s.leaf ? `: NASA MODIS shows the canopy at ${Math.round(s.leaf.factor * 100)}% of its summer leaf area on ${s.leaf.date}` : ""}). Every building's shape and height come from its footprint and the USGS LiDAR roof surface.${H.roof_cp?.some((x: unknown) => x != null) ? ` A second, nested simulation around the building${H.cfd_grid ? ` (${H.cfd_grid.dx} m cells, ${H.cfd_grid.cells.join(" × ")})` : ""}, driven by the coarse run's wind at its edge, gives the pressure on each wall and the roof against the free-stream static pressure and the approach-flow dynamic pressure at mean roof height (ASCE 7 / AIJ convention). Loads shown are net of the building's internal pressure, taken as the mean of its wall pressures (air leaks in through every wall).` : ""} The incoming wind has a log-law profile and synthetic turbulence (Smirnov et al. 2001) for upwind roughness z₀ ≈ ${W!.sectors.map(x => x.z0.toFixed(2)).join(" / ")} m, measured from building and tree density (Macdonald et al. 1998).` : "The wind simulation was not run for this report."],
    ["Turns flow into gusts", "Local 3-second gust = mean wind + 3 × turbulence (Hunt et al. 1976; Lawson's gust-equivalent mean), both taken from the simulation. The result is expressed relative to the same formula over open terrain, so the NWS gust forecast sets the absolute level and the simulation only redistributes it around your lot. One run per direction scales to any forecast speed, because flow around sharp-edged buildings does not depend on Reynolds number at storm speeds."],
    ["Calibrates the forecast", (() => {
      const C = D.calib; if (!C) return "Calibration data was unavailable, so NWS and Google gusts are blended with equal weight.";
      const w = Object.entries(C.weights["0-24 h"] as Record<string, number>).filter(([n]) => !/bias|cape/.test(n)).sort((a, b) => b[1] - a[1]);
      const v = C.verify?.leads?.["0-24 h"];
      const on = w.filter(([, x]) => x >= .01), off = w.filter(([, x]) => x < .01).map(([n]) => n);
      return `${w.length} weather models are blended by EMOS, the statistical post-processing used by NWS and ECMWF: weights, bias and spread are refit every run to the peak gusts measured over the last ${C.train_days} days at ${C.stations.join(", ")}, and storm energy (CAPE) widens the upper tail. Current weights: ${on.map(([n, x]) => `${n} ${x.toFixed(2)}`).join(", ")}${off.length ? ` (${off.join(", ")} added nothing here lately)` : ""}.`
        + (v ? ` Tested the honest way (each day predicted only from the days before it) over the last 60 days: gust error ${(v.skill * 100).toFixed(0)}% lower than an equal-weight blend, and the 90% range held ${(v.cover90_new * 100).toFixed(0)}% of measured gusts (old method: ${(v.cover90_old * 100).toFixed(0)}%).` : "");
    })()],
    ["Runs the storm 1,000 times", `NHC-style Monte Carlo (DeMaria et al. 2009): track and intensity errors drawn from NHC's 5-year verification with hour-to-hour correlation, a Holland wind profile and Kaplan–DeMaria inland decay, anchored to the NWS ${D.meta.office} hourly gust forecast.${D.sim.dir_cover == null ? "" : ` Each run also has its own wind direction: the forecast direction plus its error, turned further when that run's storm passes on a different side of the house. The wind simulation is repeated for every direction that carries a real share of the damaging wind${W ? ` (${W.sectors.length} here, holding ${Math.round(((W as any).cover ?? 1) * 100)}% of it within half a sector)` : ""}, and each run reads the flow for its own direction.`}`],
    ...(D.sim.dir_cover == null ? [] : [["Checks for tornadoes", D.sim.tornado ? `SPC's tornado outlook gives the chance of a tornado within 25 miles (${D.sim.tornado.days.map((d: any) => `day ${d.day}: ${Math.round(d.p25 * 100)}%`).join(", ")}). Thousands of path scenarios (rating, length, width, forward speed) are drawn from US climatology, shifted toward strong tornadoes by SPC's conditional-intensity level, and each passes the house as a moving Rankine vortex. The share of the 25-mile circle where the lot would feel each wind level is integrated exactly, then scored with this building's damage curves and added to the damage odds.` : "SPC's tornado outlook was unavailable, so tornadoes are not included."] as [string, string]]),
    ["Scores each hazard", `Trees: wind bending moment at each tree's crown vs. trunk strength (green-wood MOR ${k.tree_MOR / 1e6} MPa) and root anchorage, scaled so a typical mature hardwood fails near ${k.tree_ref[2]} mph (post-hurricane tree surveys), weaker in wet soil. ${H.cls ? `Building (${H.cls_label}): the gust at mean roof height over the ASCE 7 design gust (${k.design_mph ?? 110} mph basic speed), squared, on lognormal damage curves for its construction class${H.cls === "WSF" ? " (Hazus-style wood-frame curves)" : ""}, with the ASCE 7 topographic factor` : "House: Hazus wood-frame fragility curves with the ASCE 7 topographic factor"}. Wall pressures: ${H.roof_cp?.some((x: unknown) => x != null) ? `net pressure coefficients from the nested ${bldNoun(H)} simulation, applied to the 3-second gust (quasi-steady)` : "ASCE 7 low-rise coefficients"}. Power: logistic outage curve. Water: SCS curve number and NWS river forecasts.`],
  ];
  const src: [string, string | undefined][] = [["NHC / WPC advisory", D.meta.storm?.url], ["NWS gridded forecast (api.weather.gov)", "https://www.weather.gov/documentation/services-web-api"], ["SPC convective outlooks", "https://www.spc.noaa.gov/products/outlook/"],
    ["Open-Meteo archived model forecasts", "https://open-meteo.com/en/docs/previous-runs-api"], ["Iowa Environmental Mesonet ASOS 1-minute archive", "https://mesonet.agron.iastate.edu/request/asos/1min.phtml"],
    ["Gneiting et al. 2005, EMOS", "https://doi.org/10.1175/MWR2904.1"],
    ["USGS 3DEP LiDAR point clouds (AWS open data)", "https://registry.opendata.aws/usgs-lidar/"], ["USGS 3D Elevation Program", "https://www.usgs.gov/3d-elevation-program"],
    ["Esri World Imagery and World Geocoder", "https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9"], ["Building footprints: OpenStreetMap, Microsoft, FEMA/ORNL", "https://github.com/microsoft/GlobalMLBuildingFootprints"],
    ["FEMA National Flood Hazard Layer", "https://hazards.fema.gov/"], ["NRCS Soil Data Access", "https://sdmdataaccess.sc.egov.usda.gov/"], ["NOAA National Water Prediction Service", "https://water.noaa.gov/"],
    ["DeMaria et al. 2009, wind speed probabilities", "https://www.nhc.noaa.gov/pdf/2009waf_wsp.pdf"], ["FEMA Hazus hurricane model", "https://www.fema.gov/flood-maps/products-tools/hazus"]];
  return (
    <>
      <div style={{ marginTop: 6 }}>{steps.map(([t, d], i) => <div className="step" key={t}><span className="n">{String(i + 1).padStart(2, "0")}</span><div><b>{t}</b>{d}</div></div>)}</div>
      <h3 className="sub">Known limits</h3>
      <ul className="note">
        <li>The LiDAR is from {L ? L.year : "an older survey"}. Trees planted since then are modelled as small 6 m crowns.</li>
        <li>Validated on the standard cube-in-boundary-layer test (AIJ / COST 732; run it with <code>--validate-cfd</code>), referenced like the experiments to the undisturbed approach flow: at 10 and 16 cells per building height the wake reattaches 1.40 and 1.38 heights downstream (wind tunnels 1.4–1.6) and the windward-face pressure coefficient is 0.85 and 0.89 (tunnels 0.7–0.8, so about 10 % high, on the safe side). On an isolated low-rise box the area-averaged windward wall comes out near +0.6, matching tunnel data, but side-wall and roof suctions read 30–50 % weaker than tunnel data: the separated shear layer at the roof edge needs a finer grid than ~14 cells per height to resolve fully.{H.cfd_grid ? ` This building got ${Math.round(H.ridge_m / H.cfd_grid.dx)} cells over its height${H.ridge_m / H.cfd_grid.dx < 10 ? ", coarser than the benchmark, so its pressures are less certain than that" : ""}.` : ""}</li>
        <li>The forecast calibration learns from the last {D.calib?.train_days ?? 40} days. Strong storms are rare in any 40-day window, so the far upper tail still leans on the physics and the storm-energy term.</li>
        <li>Species, decay and root damage are unknown. A hollow or leaning tree can fail far below its modelled gust.</li>
      </ul>
      <h3 className="sub">Data sources</h3>
      <ul className="note">{src.filter(x => x[1]).map(([t, u]) => <li key={t}><a href={u} target="_blank" rel="noopener">{t}</a></li>)}</ul>
      <p className="note">Generated {D.meta.generated} · NWS {D.meta.office} · {D.meta.lat.toFixed(5)}, {D.meta.lon.toFixed(5)} · <a href={standaloneUrl} target="_blank" rel="noopener">standalone report</a></p>
    </>
  );
}
