"""StormVoyance web API.

    .venv/bin/uvicorn server:app --port 8000          (serves web/dist too, after `npm run build` in web/)

POST /api/runs starts the engine (stormvoyance.py) as a subprocess; progress lines stream over SSE.
"""
import asyncio, json, os, re, sys, time, uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import stormvoyance as eng

HERE = Path(__file__).resolve().parent
RUNS = eng.CACHE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="StormVoyance")


@app.middleware("http")
async def fresh_page(request, call_next):
    """Hashed /assets/ files never change; everything else (index.html) must be revalidated, or a browser keeps
    running the previous build after a rebuild."""
    r = await call_next(request)
    if not request.url.path.startswith("/assets/"):
        r.headers.setdefault("Cache-Control", "no-cache")
    return r
jobs: dict[str, dict] = {}
gate = asyncio.Semaphore(1)   # ponytail: one engine run at a time (CFD saturates the CPU); use a worker queue for multi-user


def env(name: str) -> str:
    """Process env first, then the .env next to this file (re-read each call so adding a key needs no restart)."""
    if v := os.environ.get(name):
        return v
    p = HERE / ".env"
    for line in p.read_text().splitlines() if p.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip("\"'")
    return ""


_tiles_ok: dict[str, bool] = {}


def tiles_ok(key: str) -> bool:
    """Does this key reach Photorealistic 3D Tiles? (Needs billing; the free Demo Key gets 403.) Probed once per key."""
    if key not in _tiles_ok:
        try:
            eng.get("https://tile.googleapis.com/v1/3dtiles/root.json?key=" + key, timeout=15)
            _tiles_ok[key] = True
        except Exception:
            _tiles_ok[key] = False
    return _tiles_ok[key]


@app.get("/api/config")
def config():
    # The key is a browser key by design (Google's own JS clients ship it); restrict it by HTTP referrer.
    # mode: "tiles" = raw 3D Tiles in our renderer, "map3d" = Google's 3D map component under our canvas, "" = none
    key = env("GOOGLE_MAPS_KEY")
    return dict(google_key=key, google_mode=("tiles" if tiles_ok(key) else "map3d") if key else "")


@app.get("/api/geocode")
def geocode(q: str):
    try:
        lat, lon, addr = eng.geocode(q)
    except SystemExit as e:          # engine CLI exits on a miss
        raise HTTPException(404, str(e))
    return dict(lat=lat, lon=lon, address=addr)


class RunIn(BaseModel):
    address: str = Field(max_length=200)
    outline: list[tuple[float, float]] | None = Field(None, min_length=3, max_length=200)   # [[lat, lon], ...]; default: engine finds the house
    stories: int = Field(2, ge=1, le=4)
    roof_age: float | None = Field(None, ge=0, le=100)

    @field_validator("outline")
    @classmethod
    def in_range(cls, v):
        if v is not None and not all(-90 <= a <= 90 and -180 <= b <= 180 for a, b in v):
            raise ValueError("outline points must be lat, lon")
        return v


async def run(jid: str, body: RunIn):
    job, out = jobs[jid], RUNS / jid
    out.mkdir(exist_ok=True)
    cmd = [sys.executable, "-I", str(HERE / "stormvoyance.py"), body.address, "-o", str(out / "report.html"), "--stories", str(body.stories)]
    if body.outline:
        cmd += ["--outline", json.dumps(body.outline)]
    if body.roof_age is not None:
        cmd += ["--roof-age", str(body.roof_age)]
    job["log"].append("Waiting for the simulator…" if gate.locked() else "Starting…")
    async with gate:
        p = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, cwd=out,
                                              env={**os.environ, "GOOGLE_MAPS_KEY": env("GOOGLE_MAPS_KEY")})
        async for line in p.stderr:
            t = line.decode(errors="replace").strip()
            if t.startswith(("•", "✓")):
                job["log"].append(t.lstrip("•✓ "))
            elif t:
                job["tail"] = (job["tail"] + [t])[-6:]
        rc = await p.wait()
    job["state"] = "done" if rc == 0 and (out / "report.html").exists() else "error"
    if job["state"] == "error":
        job["log"].append(job["tail"][-1] if job["tail"] else f"engine exited with code {rc}")


@app.post("/api/runs")
async def start(body: RunIn):
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = dict(state="running", log=[], tail=[], address=body.address, started=time.time())
    asyncio.create_task(run(jid, body))
    return dict(id=jid)


@app.get("/api/runs/{jid}/events")
async def events(jid: str):
    if jid not in jobs:
        if (RUNS / jid / "report.html").exists():
            jobs[jid] = dict(state="done", log=[], tail=[], address="", started=time.time())
        else:
            raise HTTPException(404)
    async def stream():
        sent, quiet = 0, 0
        yield f"data: {json.dumps(dict(type='meta', address=jobs[jid]['address'], started=jobs[jid]['started']))}\n\n"
        while True:
            job = jobs[jid]
            for msg in job["log"][sent:]:
                yield f"data: {json.dumps(dict(type='log', text=msg))}\n\n"
            quiet = 0 if len(job["log"]) > sent else quiet + 1
            sent = len(job["log"])
            if job["state"] != "running":
                yield f"data: {json.dumps(dict(type=job['state']))}\n\n"
                return
            if quiet >= 30:   # SSE comment every 15 s of silence keeps proxies / browsers from dropping the stream
                yield ": ping\n\n"; quiet = 0
            await asyncio.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream")


def _bundle(f: Path) -> str:
    line = next(l for l in f.read_text().splitlines() if l.startswith("const D = "))
    return line[len("const D = "):].rstrip(";")


SUMMARY_V = 4   # bump when summary() gains fields; older summary.json files are rebuilt from their report


def summary(d: Path) -> dict:
    """What the start page's run cards need from one report: headline risk, peak gust, the peak hour's weather, a 24 h gust line."""
    D = json.loads(_bundle(d / "report.html"))
    S, top = D["sim"], max(D["risks"], key=lambda r: r["p"] or 0)
    # the report's own peak: the highest hourly median gust (and that hour's 95th percentile), so card and report agree
    at = max(S["hours"], key=lambda h: h["gust"][2])
    return dict(v=SUMMARY_V, engine=D["meta"].get("engine", 0), id=d.name, address=D["meta"]["address"], generated=D["meta"]["generated"], top=top["title"], p=top["p"] or 0,
                peak_p50=round(at["gust"][2], 1), peak_p95=round(at["gust"][4], 1), peak_hour=at["t"], tz=D["meta"]["tz"],
                cond_hour={k: at.get(k) for k in ("gust", "rain", "pop", "thunder")},
                spark=[round(h["gust"][2], 1) for h in S["hours"][:24]])


@app.get("/api/runs")
def recent(n: int = 6):
    """Newest finished run per address, with its headline risk. summary.json caches what the list needs from each report."""
    out, seen = [], set()
    for d in sorted((p for p in RUNS.iterdir() if (p / "report.html").exists()), key=lambda p: -(p / "report.html").stat().st_mtime):
        sf = d / "summary.json"
        try:
            s = json.loads(sf.read_text())
        except (OSError, ValueError):
            s = {}
        if s.get("v") != SUMMARY_V:
            try:
                s = summary(d)
            except (StopIteration, ValueError, KeyError, IndexError, TypeError):
                continue
            sf.write_text(json.dumps(s))
        if s.get("engine") != eng.ENGINE_VERSION:   # older engines' reports stay reachable by link, not listed
            continue
        if s["address"] not in seen:
            seen.add(s["address"]); out.append(s)
        if len(out) >= n:
            break
    return out


@app.get("/api/runs/{jid}/report")
def report(jid: str):
    f = RUNS / jid / "report.html"
    if not re.fullmatch(r"[0-9a-f]{12}", jid) or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="text/html")


@app.get("/api/runs/{jid}/data")
def data(jid: str):
    """The report's data bundle. The standalone report embeds it on one line as `const D = {...};`."""
    f = RUNS / jid / "report.html"
    if not re.fullmatch(r"[0-9a-f]{12}", jid) or not f.exists():
        raise HTTPException(404)
    return Response(_bundle(f), media_type="application/json")


if (HERE / "web" / "dist").exists():
    app.mount("/", StaticFiles(directory=HERE / "web" / "dist", html=True), name="web")
