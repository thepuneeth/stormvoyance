import { useEffect, useRef, useState, type CSSProperties, type FormEvent, type PointerEvent } from "react";
import { Brand } from "./Logo";
import { Icon } from "./icons";
import { condition, fmtHour, type Cond } from "./weather";
import { createHalftone, type Halftone } from "./halftone";
import { pct } from "./report/model";   // the same probability format as the report (never "100%")
import "./picker.css";

export type Site = { address: string };
type Wx = { gust: number[]; rain: number[]; pop?: number; thunder?: number };
// GET /api/runs (server.py summary()); v2 fields are optional so an older server still renders a card.
export type Recent = {
  id: string; address: string; generated: string; top: string; p: number;
  peak_p50?: number; peak_p95?: number; peak_hour?: string; tz?: string; cond_hour?: Wx; spark?: number[];
};

const REST = .12;                                                   // background storm at rest; typing and hovering a run raise it
const HEAD = "See what the next storm does to your house.".split(" ");
const sev = (p: number) => p < .02 ? "ok" : p < .15 ? "warn" : "bad";         // same cut-offs as the report's risk list
const street = (a: string) => a.split(",")[0].toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase()).replace(/\b(Ne|Nw|Se|Sw)\b/g, m => m.toUpperCase());
const ranAt = (g: string) => { const m = /^\w+ (\w+ \d+), (\d+:\d+ [AP]M)/.exec(g); return m ? `${m[1]}, ${m[2]}` : g; };  // "Mon Oct 12, 3:59 PM CDT" -> "Oct 12, 3:59 PM": an old run must not pass for today's
const wkday = (t: string, tz?: string) => { try { return new Intl.DateTimeFormat("en-US", { weekday: "short", timeZone: tz }).format(new Date(t)); } catch { return ""; } };
const hoverable = typeof matchMedia === "function" && matchMedia("(hover: hover)").matches;   // no autofocus (keyboard pop-up) on phones

// The sky behind the page acts out a run: wind sets the drift, rain the dashes, and lightning only when the forecast had thunder.
// (createHalftone strikes on its own above .55, so a dry or rain-only run stays below that.)
function stormOf(r: Recent, c: Cond | null) {
  const g = Math.min(.3, Math.max(.06, ((r.peak_p50 ?? 0) - 8) / 60));
  return c?.key === "thunder" ? .78 : c?.key === "rain" ? .5 : c?.key === "showers" ? .4 : g;
}

// 24 hourly gust medians (a fixed 24 h axis) as a dot-matrix bar chart: a column of 5 dots per hour, lit up to the gust; the windiest hour in teal.
const ROWS = 5;
function Dots({ v }: { v: number[] }) {
  const top = Math.max(30, ...v), pk = v.indexOf(Math.max(...v));
  return (
    <svg className="pk-dots" viewBox={`0 0 144 ${ROWS * 5}`} aria-hidden="true">
      {v.slice(0, 24).flatMap((g, c) => {
        const on = Math.max(1, Math.round(g / top * ROWS));
        return Array.from({ length: ROWS }, (_, r) => (
          <circle key={c * ROWS + r} cx={c * 6 + 3} cy={(ROWS - r) * 5 - 2.5} r={1.35} style={{ "--c": c } as CSSProperties}
            className={r >= on ? undefined : c === pk ? "pk-hot" : "pk-on"} />
        ));
      })}
    </svg>
  );
}

function Card({ r, i, onOpen, onSky }: { r: Recent; i: number; onOpen: (id: string) => void; onSky: (r: Recent | null, c: Cond | null) => void }) {
  // label the gust the card shows (median storm's peak), so a 22 mph card never says "Calm"; rain and thunder stay the peak hour's
  const c = r.cond_hour ? condition({ ...r.cond_hour, gust: [0, 0, r.peak_p50 ?? r.cond_hour.gust[2]] }) : null, s = street(r.address), g = r.peak_p50 != null ? Math.round(r.peak_p50) : null;
  const at = r.peak_hour ? `${wkday(r.peak_hour, r.tz)} ${fmtHour(r.peak_hour, r.tz)}`.trim() : "";
  const past = !!r.peak_hour && Date.parse(r.peak_hour) < Date.now() - 3.6e6;   // that storm has been and gone
  // card spotlight follows the pointer
  const spot = (e: PointerEvent<HTMLButtonElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty("--mx", `${e.clientX - b.left}px`); e.currentTarget.style.setProperty("--my", `${e.clientY - b.top}px`);
  };
  return (
    <li style={{ "--i": i } as CSSProperties}>
      <button className="pk-card glass" onClick={() => onOpen(r.id)} onPointerMove={spot}
        onPointerEnter={() => onSky(r, c)} onPointerLeave={() => onSky(null, null)} onFocus={() => onSky(r, c)} onBlur={() => onSky(null, null)}
        aria-label={`Open ${s}${g != null ? `, peak gust ${g} mph${at && ` around ${at}`}` : ""}, ${r.top} ${pct(r.p)}, run ${ranAt(r.generated)}`}>
        <span className="pk-c-top">
          {c && <><Icon name={c.icon} size={18} /><span>{c.label}</span></>}
          <time className="pk-c-when">{ranAt(r.generated)}</time>
        </span>
        {g != null && <span className="pk-c-gust">
          <span className="num">{g}<small>mph</small></span>
          {r.peak_p95 != null && <span className="pk-c-p95"><span>p95</span><b>{Math.round(r.peak_p95)}</b></span>}
        </span>}
        <span className="pk-c-sub">{past ? `Peaked${at && ` ${at}`}` : `Peak gust${at && ` · ${at}`}`}</span>
        {!!r.spark?.length && <Dots v={r.spark} />}
        <span className="pk-c-street">{s}</span>
        <span className="pk-c-risk"><i data-sev={sev(r.p)} /><span>{r.top}</span><b>{pct(r.p)}</b></span>
      </button>
    </li>
  );
}

export default function Picker({ onRun, onOpen }: { onRun: (s: Site) => Promise<void>; onOpen: (id: string) => void }) {
  const cv = useRef<HTMLCanvasElement>(null), inp = useRef<HTMLInputElement>(null), ht = useRef<Halftone | null>(null);
  const idle = useRef(0), lastFlash = useRef(0);
  const [q, setQ] = useState(""), [busy, setBusy] = useState(false), [err, setErr] = useState("");
  const [recent, setRecent] = useState<Recent[]>([]), [sky, setSky] = useState<{ r: Recent; c: Cond | null } | null>(null);
  const rest = () => REST + .1 * Math.min(1, q.trim().length / 30);

  useEffect(() => {
    const h = createHalftone(cv.current!, { seed: 1 + Math.floor(Math.random() * 97) });
    h.setStorm(REST); ht.current = h;
    return () => { clearTimeout(idle.current); ht.current = null; h.dispose(); };
  }, []);

  useEffect(() => { fetch("/api/runs").then(r => r.ok ? r.json() : []).then(v => setRecent(Array.isArray(v) ? v : []), () => {}); }, []);

  // "/" jumps to the search field from anywhere on the page
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey || t?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t?.tagName ?? "")) return;
      e.preventDefault(); inp.current?.focus();
    };
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  }, []);

  // typing whips the sky up a little; it settles a second after the last key
  function type(v: string) {
    setQ(v); setErr("");
    const k = Math.min(1, v.trim().length / 30);
    ht.current?.setStorm(.2 + .3 * k);
    clearTimeout(idle.current);
    idle.current = window.setTimeout(() => ht.current?.setStorm(REST + .1 * k), 1000);
  }

  function preview(r: Recent | null, c: Cond | null) {
    if (busy) return;
    clearTimeout(idle.current);
    setSky(r && { r, c });
    ht.current?.setStorm(r ? stormOf(r, c) : rest());
    if (c?.key === "thunder" && performance.now() - lastFlash.current > 2000) { lastFlash.current = performance.now(); ht.current?.flash(); }
  }

  // The engine finds the house itself (building footprints + LiDAR at the geocoded rooftop), so an address is all it needs.
  // Geocoding here first just turns a typo into an instant error instead of a failed run.
  async function find(e: FormEvent) {
    e.preventDefault();
    const a = q.trim();
    if (!a) return inp.current?.focus();
    if (busy) return;
    setBusy(true); setErr(""); setSky(null); clearTimeout(idle.current);
    ht.current?.setStorm(.72); ht.current?.flash();
    try {
      const r = await fetch(`/api/geocode?q=${encodeURIComponent(a)}`).catch(() => { throw new Error("Can't reach the StormVoyance server. Is it running?"); });
      if (!r.ok) throw new Error("We couldn't find that address. US addresses only.");
      await onRun({ address: (await r.json()).address });
    } catch (x) {
      setErr((x as Error).message); setBusy(false); ht.current?.setStorm(rest());
      inp.current?.focus();
    }
  }

  return (
    <main className="pk" data-busy={busy || undefined}>
      <div className="pk-bg" aria-hidden="true"><canvas ref={cv} className="halftone" /></div>

      <header className="pk-top"><Brand /></header>

      <div className="pk-main">
      <section className="pk-hero">
        <h1 className="pk-title">{HEAD.flatMap((w, i) => [<span key={i} className="pk-w" style={{ "--i": i } as CSSProperties}>{w}</span>, " "])}</h1>
        <form className="pk-field glass" onSubmit={find} role="search">
          <Icon name="search" size={20} />
          <input ref={inp} className="pk-in" value={q} onChange={e => type(e.target.value)} placeholder="Street address, city, state"
            aria-label="Address" autoComplete="street-address" enterKeyHint="go" spellCheck={false} autoFocus={hoverable} readOnly={busy}
            aria-invalid={!!err || undefined} aria-describedby={err ? "pk-err" : undefined}
            onKeyDown={e => { if (e.key === "Escape") e.currentTarget.blur(); }} />
          <kbd className="pk-kbd" aria-hidden="true">/</kbd>
          <button className="primary pk-go" disabled={busy} aria-busy={busy} aria-label={busy ? "Starting" : "Simulate"}>
            <span className="pk-go-t">{busy ? "Starting" : "Simulate"}</span>
            {busy ? <span className="pk-spin" aria-hidden="true"><i /><i /><i /></span> : <Icon name="arrow-right" size={18} />}
          </button>
        </form>
        {err && <p id="pk-err" className="err pk-err" role="alert">{err}</p>}
      </section>

      {recent.length > 0 && <section className="pk-recent" aria-labelledby="pk-recent-h">
        <div className="pk-recent-h">
          <h2 id="pk-recent-h" className="lbl">Recent runs</h2>
          <span className="pk-sky" data-on={sky ? "" : undefined}>
            {sky && `Sky: ${street(sky.r.address)}${sky.c ? ` · ${sky.c.label}` : ""}${sky.r.peak_p50 != null ? ` · ${Math.round(sky.r.peak_p50)} mph` : ""}`}
          </span>
        </div>
        <ul className="pk-cards">{recent.map((r, i) => <Card key={r.id} r={r} i={i} onOpen={onOpen} onSky={preview} />)}</ul>
      </section>}
      </div>
    </main>
  );
}
