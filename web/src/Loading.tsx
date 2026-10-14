import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Brand } from "./Logo";
import { Icon, type IconName } from "./icons";
import { createHalftone, type Halftone } from "./halftone";
import { createDotMorph, MODELS, type DotMorph } from "./dotMorph";
import "./loading.css";

// Engine progress lines -> stages. secs = typical first-run duration (drives the progress estimate, never a promise).
// short + icon label the step track; dotMorph.ts draws one scene per stage, in this order.
const STAGES: { re: RegExp; title: string; short: string; icon: IconName; secs: number; facts: string[] }[] = [
  { re: /^(Starting|Waiting)/, title: "Warming up the simulator", short: "Start", icon: "spark", secs: 4, facts: [
    "Everything you're about to see is computed live for your address. Nothing is canned."] },
  { re: /\(-?\d+\.\d+, -?\d+\.\d+\)$/, title: "Found your address", short: "Address", icon: "pin", secs: 4, facts: [
    "Your address is pinned with Esri's geocoder, with the US Census geocoder as a backup."] },
  { re: /^(blended|calibrated|nowcast)/, title: "Weighing seven weather models", short: "Models", icon: "cloud", secs: 20, facts: [
    "Seven forecast models are graded on how well they predicted gusts at your nearest airports over the last 40 days.",
    "The better a model has done near you, the more weight it gets in your forecast.",
    "Observed gusts come from 1-minute airport wind records, not hourly summaries, so short peaks aren't missed."] },
  { re: /^site data/, title: "Flood zone, elevation and soil", short: "Ground", icon: "flood", secs: 10, facts: [
    "FEMA flood maps, USGS elevation and NRCS soil surveys decide how fast rain turns into standing water.",
    "Clay-heavy soil sheds most rain as runoff and holds tree roots less firmly when soaked."] },
  { re: /^locating/, title: "Finding your house", short: "House", icon: "house", secs: 10, facts: [
    "Your house is picked out of public building-footprint maps and checked against aerial imagery to get the exact shape of its walls."] },
  { re: /^satellite/, title: "Satellite imagery and the neighbours", short: "Imagery", icon: "search", secs: 15, facts: [
    "Neighbouring houses matter: they shelter your walls from some directions and funnel wind from others."] },
  { re: /^terrain/, title: "Measuring your trees with USGS LiDAR", short: "LiDAR", icon: "tree", secs: 45, facts: [
    "USGS LiDAR maps the ground at two or more laser points per square metre, accurate to about 10 cm in height.",
    "Every tree near your house gets a measured height and crown width, not a guess.",
    "Satellites check whether the trees are in leaf. Bare branches catch much less wind."] },
  { re: /^wind simulation/, title: "Simulating the wind around your house", short: "Wind", icon: "wind", secs: 300, facts: [
    "This is the big one: a lattice-Boltzmann CFD solver, the same family used for city wind studies.",
    "Wind piles up against the wall facing it and pulls at the roof and the far side.",
    "Roof edges are where shingles lift first. Suction there can be more than twice the roof average.",
    "A second, finer 1-metre grid zooms in on your house for wall-by-wall pressures.",
    "First runs at a new address take a few minutes here. Later runs reuse the result instantly."] },
  { re: /^Monte Carlo/, title: "Playing out 1,000 versions of the storm", short: "Storms", icon: "thunder", secs: 15, facts: [
    "Each simulated storm gets its own track, timing and gust surprises.",
    "A 30% chance means about 300 of the 1,000 simulated storms caused it."] },
  { re: /^wrote/, title: "Building your 3D report", short: "3D", icon: "layers", secs: 5, facts: [
    "Packing your house, trees and wind field into an interactive 3D model."] },
];

const stageOf = (log: string[]) => log.reduce((s, l) => Math.max(s, STAGES.findIndex(st => st.re.test(l))), 0);

// What the engine has learned so far, read straight off its log lines. Later lines of a kind replace earlier ones.
type Tile = { k: string; label: string; v: string; unit?: string; sub?: string; mono?: boolean };
function findings(log: string[]): Tile[] {
  const out = new Map<string, Tile>();
  for (const l of log) {
    let m;
    if ((m = l.match(/\((-?\d+\.\d+), (-?\d+\.\d+)\)$/))) out.set("at", { k: "at", label: "Pinned at", mono: true,
      v: `${Math.abs(+m[1]).toFixed(3)}° ${+m[1] >= 0 ? "N" : "S"}`, sub: `${Math.abs(+m[2]).toFixed(3)}° ${+m[2] >= 0 ? "E" : "W"}` });
    if ((m = l.match(/model disagreement (\d+)%/))) out.set("spread", { k: "spread", label: "Forecasts disagree", v: m[1], unit: "%" });
    if ((m = l.match(/at ((?:K\w+, )*K\w+)\)/))) {
      const st = m[1].split(", ");
      out.set("air", { k: "air", label: "Graded at", v: String(st.length), unit: st.length === 1 ? "airport" : "airports", sub: st.join(" ") });
    }
    if ((m = l.match(/top weights (\w+)(?: (\d*\.\d+))?/))) out.set("top", { k: "top", label: "Most trusted here", v: m[1], sub: m[2] && `weight ${m[2]}` });
    if ((m = l.match(/gust now (\d+) mph/))) out.set("gust", { k: "gust", label: "Gust right now", v: m[1], unit: "mph" });
    if ((m = l.match(/CFD, (\d+) wind directions?\)/))) out.set("dirs", { k: "dirs", label: "Wind directions", v: m[1], sub: "lattice-Boltzmann CFD" });
    if ((m = l.match(/^Monte Carlo \((\d+) members\)/))) out.set("mc", { k: "mc", label: "Storms simulated", v: (+m[1]).toLocaleString("en-US") });
  }
  return [...out.values()];
}

const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export default function Loading({ address, log, state, started, onReset }:
  { address: string; log: string[]; state: "running" | "error"; started: number; onReset: () => void }) {
  const seen = useRef<number[]>([]);                       // when each stage first appeared (this tab's clock)
  const peak = useRef(0), shown = useRef(0), run = useRef(started), prevStage = useRef(-1);
  if (run.current !== started) { run.current = started; seen.current = []; peak.current = 0; shown.current = 0; }   // a new run
  const [now, setNow] = useState(Date.now());
  const [notify, setNotify] = useState(() => "Notification" in window ? Notification.permission : "denied");
  const sky = useRef<HTMLCanvasElement>(null), dots = useRef<HTMLCanvasElement>(null), box = useRef<HTMLDivElement>(null);
  const num = useRef<HTMLSpanElement>(null), fx = useRef<{ sky: Halftone; dots: DotMorph } | null>(null), logEl = useRef<HTMLOListElement>(null);

  useEffect(() => { if (state === "error") return; const iv = setInterval(() => setNow(Date.now()), 250); return () => clearInterval(iv); }, [state]);   // a failed run's clock stops

  const failed = state === "error", stage = stageOf(log);
  if (seen.current[stage] === undefined) seen.current[stage] = Date.now();

  const inStage = now - seen.current[stage];
  const total = STAGES.reduce((t, s) => t + s.secs, 0);
  const done = STAGES.slice(0, stage).reduce((t, s) => t + s.secs, 0);
  const pct = Math.min(99, Math.round(100 * (done + STAGES[stage].secs * (1 - Math.exp(-inStage / 1000 / STAGES[stage].secs))) / total));
  const shownPct = peak.current = Math.max(peak.current, pct);   // a reconnect replays the log from stage 0; never count backwards
  useEffect(() => { document.title = failed ? "Failed · StormVoyance" : `${shownPct}% · StormVoyance`; }, [shownPct, failed]);
  useEffect(() => () => { document.title = "StormVoyance"; }, []);

  const facts = STAGES[stage].facts, fi = Math.floor(inStage / 9000) % facts.length;
  const secs = Math.max(0, Math.round((now - started) / 1000)), clock = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
  const found = findings(log), top = found.find(f => f.k === "top")?.v;

  // the sky and the dots live for the whole screen; stage, progress and the trusted model are pushed in
  useEffect(() => {
    const s = createHalftone(sky.current!, { seed: 11, cell: 11 }), d = createDotMorph(dots.current!, box.current!);
    fx.current = { sky: s, dots: d };
    return () => { s.dispose(); d.dispose(); fx.current = null; prevStage.current = -1; };
  }, []);
  useEffect(() => {
    const f = fx.current;
    if (!f) return;
    f.dots.setStage(failed ? -1 : stage);
    if (!failed && prevStage.current >= 0 && stage > prevStage.current) f.sky.flash();
    prevStage.current = stage;
  }, [stage, failed]);
  useEffect(() => { fx.current?.sky.setStorm(failed ? .06 : .08 + .8 * shownPct / 100); }, [shownPct, failed]);
  useEffect(() => { fx.current?.dots.setMark(top ? MODELS.indexOf(top) : -1); }, [top]);
  useEffect(() => { const el = logEl.current; if (el) el.scrollTop = el.scrollHeight; }, [failed, log.length]);   // the failing line in view

  // the big number eases toward the estimate instead of ticking
  useEffect(() => {
    const el = num.current;
    if (!el) return;
    if (reduced()) { shown.current = shownPct; el.textContent = String(shownPct); return; }
    let raf = 0, last = performance.now();
    const tick = (t: number) => {
      const s = shown.current, dt = Math.min(.1, (t - last) / 1000), n = Math.abs(shownPct - s) < .5 ? shownPct : s + (shownPct - s) * (1 - Math.exp(-dt * 3.5));
      last = t; shown.current = n; el.textContent = String(Math.round(n));
      if (n !== shownPct) raf = requestAnimationFrame(tick);
    };
    el.textContent = String(Math.round(shown.current));
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [shownPct, failed]);

  return (
    <div className={`lo${failed ? " lo-failed" : ""}`}>
      <div className="lo-bg" aria-hidden="true">
        <canvas ref={sky} className="halftone lo-sky" />
        <div className="lo-scrim" />
        <canvas ref={dots} className="halftone" />
      </div>

      <header className="lo-top">
        <Brand />
        <div className="lo-meta">
          {!failed && notify !== "denied" && (notify === "default"
            ? <button className="ghost lo-bell" aria-label="Notify me when it's ready" onClick={() => Notification.requestPermission().then(setNotify)}>
                <Icon name="bell" size={16} /><span>Notify me when it's ready</span></button>
            : <span className="lo-hint" title="Feel free to switch tabs."><Icon name="check" size={14} /><span>We'll notify you when it's ready</span></span>)}
          <span className="lo-clock" title="Time elapsed"><Icon name="clock" size={14} />{clock}</span>
        </div>
      </header>

      <main className="lo-main">
        <div className="lo-scene" ref={box} />
        <section className="lo-copy">
          <h1 className="lo-addr" title={address || undefined}><Icon name="pin" size={16} /><span>{address || "Your house"}</span></h1>
          {failed ? <>
            <p className="lbl lo-step lo-bad">Simulation stopped at step {stage + 1} of {STAGES.length}</p>
            <h2 className="lo-title">Something went wrong</h2>
            <p className="lo-fact">It stopped at “{STAGES[stage].title}”. Here is everything the engine reported.</p>
            <ol className="lo-log glass" ref={logEl}>
              {log.length ? log.map((l, i) => <li key={i} className={i === log.length - 1 ? "bad" : ""}>
                <Icon name={i === log.length - 1 ? "close" : "check"} size={14} /><span>{l}</span></li>)
                : <li className="bad"><Icon name="close" size={14} /><span>No progress was received from the engine.</span></li>}
            </ol>
            <div><button className="primary" onClick={onReset}>Try again</button></div>
          </> : <>
            <div className="lo-pct num" role="progressbar" aria-label="Progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={shownPct}>
              <span ref={num} /><small>%</small>
            </div>
            <div className="lo-now">
              <p className="lbl lo-step">Step {stage + 1} of {STAGES.length}</p>
              <div aria-live="polite"><h2 className="lo-title" key={stage}>{STAGES[stage].title}</h2></div>
              <p className="lo-fact" key={`${stage}-${fi}`}>{facts[fi]}</p>
            </div>
            <section className="lo-found" aria-label="Found so far">
              {found.length > 0 && <p className="lbl">Found so far</p>}
              <ul>{found.map(f => <li key={f.k} className="lo-tile glass">
                <span className="lbl">{f.label}</span>
                <b className={f.mono ? "lo-tv lo-mono" : `lo-tv num${String(f.v).length > 4 ? " lo-long" : ""}`}>{f.v}{f.unit && <small>{f.unit}</small>}</b>
                {f.sub && <span className="lo-sub">{f.sub}</span>}
              </li>)}</ul>
            </section>
          </>}
        </section>
      </main>

      <Track stage={stage} failed={failed} />
    </div>
  );
}

// Shot-1 style week row: a smooth curve threading one glass node per stage. Solid behind you, dotted ahead.
const WAVE = STAGES.map((_, i) => Math.sin(i * 1.1 + .4));

function Track({ stage, failed }: { stage: number; failed: boolean }) {
  const ref = useRef<HTMLOListElement>(null), [geo, setGeo] = useState({ w: 0, h: 0, amp: 0, y: 0 });
  useEffect(() => {
    const el = ref.current!;
    const ro = new ResizeObserver(() => {
      const cs = getComputedStyle(el);
      setGeo({ w: el.clientWidth, h: el.clientHeight, amp: parseFloat(cs.getPropertyValue("--amp")) || 0, y: parseFloat(cs.getPropertyValue("--y")) || 0 });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const n = STAGES.length, { w, h, amp } = geo, cy = h * geo.y / 100;
  const at = (i: number) => [(i + .5) * w / n, cy + Math.sin(i * 1.1 + .4) * amp];
  const P = [at(-.5), ...STAGES.map((_, i) => at(i)), at(n - .5)];     // phantom ends run the curve to both edges
  const d = P.slice(1).reduce((s, c, i) => {                            // Catmull-Rom through the nodes, as cubic Beziers
    const a = P[i - 1] ?? P[i], b = P[i], e = P[i + 2] ?? c;
    return s + `C${b[0] + (c[0] - a[0]) / 6} ${b[1] + (c[1] - a[1]) / 6} ${c[0] - (e[0] - b[0]) / 6} ${c[1] - (e[1] - b[1]) / 6} ${c[0]} ${c[1]}`;
  }, `M${P[0][0]} ${P[0][1]}`);
  const x = at(stage)[0];
  return (
    <ol className="lo-track" ref={ref} aria-label="Steps">
      {w > 0 && <svg width={w} height={h} aria-hidden="true">
        <defs>
          <clipPath id="lo-done"><rect className="lo-clip" width={w} height={h} style={{ transform: `scaleX(${x / w})` }} /></clipPath>
          <clipPath id="lo-todo"><rect x={x} width={Math.max(0, w - x)} height={h} /></clipPath>
        </defs>
        <path className="lo-curve todo" d={d} clipPath="url(#lo-todo)" />
        <path className="lo-curve done" d={d} clipPath="url(#lo-done)" />
      </svg>}
      {STAGES.map((s, i) => {
        const cls = i < stage ? "done" : i > stage ? "" : failed ? "bad" : "cur";
        return <li key={i} className={`lo-node ${cls}`} title={s.title} aria-current={i === stage ? "step" : undefined}
          style={{ left: `${(i + .5) * 100 / n}%`, "--dy": WAVE[i] } as CSSProperties}>
          <Icon name={s.icon} size={16} />
          <span className="lo-nl">{s.short}</span>
          <span className="sr-only">{i < stage ? ", done" : i === stage ? (failed ? ", failed" : ", in progress") : ""}</span>
        </li>;
      })}
    </ol>
  );
}
