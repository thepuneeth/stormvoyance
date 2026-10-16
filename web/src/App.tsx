import { Component, useEffect, useRef, useState, type ReactNode } from "react";
import Picker, { type Site } from "./Picker";
import Loading from "./Loading";
import Report from "./report/Report";
import type { Bundle } from "./report/model";

type Run = { id: string; address: string };

// a crash while drawing a report must not leave a blank page: say so and offer the way back
class Guard extends Component<{ children: ReactNode; onBack: () => void }, { err: Error | null }> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) { return { err }; }
  componentDidCatch(err: Error) { console.error("report crashed:", err); }
  render() {
    if (!this.state.err) return this.props.children;
    return <main className="crash"><h1>This report could not be drawn.</h1><p>{this.state.err.message}</p>
      <div className="crash-acts"><button className="primary" onClick={() => location.reload()}>Reload</button>
        <button className="ghost" onClick={this.props.onBack}>Back to start</button></div></main>;
  }
}
const runFromUrl = () => new URLSearchParams(location.search).get("run");

export default function App() {
  const [run, setRun_] = useState<Run | null>(() => { const id = runFromUrl(); return id ? { id, address: "" } : null; });
  const [log, setLog] = useState<string[]>([]);
  const [state, setState] = useState<"running" | "done" | "error">("running");
  const t0 = useRef(Date.now());
  const [data, setData] = useState<Bundle | null>(null);
  // switching runs clears the previous run's state in the same render; otherwise a stale "done" makes the new run fetch
  // its not-yet-written report, 404 and show as failed
  const setRun = (r: Run | null) => { setState("running"); setData(null); setLog([]); setRun_(r); };

  // back/forward to the run already on screen keeps it (resetting it would never reconnect: the stream effect keys on the id)
  const runRef = useRef(run); runRef.current = run;
  useEffect(() => { const f = () => { const id = runFromUrl(); if (id !== (runRef.current?.id ?? null)) setRun(id ? { id, address: "" } : null); }; addEventListener("popstate", f); return () => removeEventListener("popstate", f); }, []);

  useEffect(() => {
    if (!run) return;
    setLog([]); setState("running"); setData(null); t0.current = Date.now();
    // A dropped stream is not a failed run (long silent CFD steps, sleep, flaky Wi-Fi): check whether the report
    // exists, else reconnect (the server replays the whole log). Give up only after several dead reconnects.
    let es: EventSource, retry: number | undefined, dead = 0, gone = false;
    const connect = () => {
      if (gone) return;
      es = new EventSource(`/api/runs/${run.id}/events`);
      es.onmessage = e => {
        const m = JSON.parse(e.data); dead = 0;
        if (m.type === "meta") { setLog([]); t0.current = m.started * 1000; if (m.address) setRun_(r => r && { ...r, address: m.address }); }
        else if (m.type === "log") setLog(l => [...l, m.text]);
        else { setState(m.type); es.close(); }
      };
      es.onerror = () => {
        es.close();
        retry = window.setTimeout(async () => {
          if (gone) return;
          const ok = (await fetch(`/api/runs/${run.id}/data`).catch(() => null))?.ok;
          if (gone) return;   // the user switched runs while the probe was in flight
          if (ok) return setState(s => s === "running" ? "done" : s);
          if (++dead > 5) return setState(s => s === "running" ? "error" : s);
          connect();
        }, 2000);
      };
    };
    connect();
    return () => { gone = true; es.close(); clearTimeout(retry); };
  }, [run?.id]);

  async function start(site: Site) {
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(site) });
    if (!r.ok) throw new Error((await r.json()).detail?.[0]?.msg ?? "Could not start the simulation");
    const { id } = await r.json();
    history.pushState(null, "", `?run=${id}`);
    setRun({ id, address: site.address });
  }
  const reset = () => { history.pushState(null, "", location.pathname); setRun(null); };
  const open = (id: string) => { history.pushState(null, "", `?run=${id}`); setRun({ id, address: "" }); };

  useEffect(() => {
    if (state !== "done" || !run) return;
    let alive = true;   // a slower fetch for a run the user already left must not land in the new one
    fetch(`/api/runs/${run.id}/data`).then(r => r.ok ? r.json() : Promise.reject()).then(d => alive && setData(d), () => alive && setState("error"));
    if (document.hidden && "Notification" in window && Notification.permission === "granted")
      try { new Notification("Your StormVoyance report is ready", { body: run.address || undefined, icon: (document.querySelector("link[rel=icon]") as HTMLLinkElement)?.href }); }
      catch { /* Android Chrome only allows service-worker notifications; the tab title still says it's ready */ }
    return () => { alive = false; };
  }, [state, run?.id]);

  if (!run) return <Picker onRun={start} onOpen={open} />;

  if (state === "done" && data) return <Guard key={run.id} onBack={reset}><Report data={data} onBack={reset} standaloneUrl={`/api/runs/${run.id}/report`} /></Guard>;
  return <Loading address={run.address} log={log} state={state === "error" ? "error" : "running"} started={t0.current} onReset={reset} />;
}
