#!/usr/bin/env python3
"""StormVoyance engine: storm impact simulator for any US address.

    python3 stormvoyance.py "3764 Elvis Presley Blvd, Memphis, TN 38116"
    python3 stormvoyance.py "35.0477,-90.0260" --storm al092026 --trees-height 18 --roof-age 3

Pulls live NHC/WPC storm advisories, NWS gridded forecast, FEMA flood zone, USGS
terrain, NRCS soil, NOAA river gauges, building footprints and satellite imagery,
runs an NHC-style 1000-member Monte Carlo (track/intensity error + Holland vortex
+ Kaplan-DeMaria decay, anchored to the NWS point forecast), scores Hazus-style
fragilities for the house, trees, power and flooding, and writes ONE self-contained
HTML report (3D scene + airflow simulation run in the browser).

Deps: numpy, scipy, Pillow.  Self-check: python3 stormvoyance.py --selftest
"""
import argparse, base64, concurrent.futures as cf, csv, datetime as dt, gzip, hashlib, io, json, math
import os, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules under `python -I` (cwd stays off the path)
import calib
import tornado
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))  # cfd.py next to this file (also under python -I)
from scipy import ndimage as nd
from scipy.special import ndtr  # standard normal CDF

UA = {"User-Agent": "StormVoyance/0.2 (storm impact report)"}  # api.weather.gov requires a UA
CACHE = Path(os.environ.get("STORMVOYANCE_CACHE", Path.home() / ".cache" / "stormvoyance"))
KT2MPH, MM2IN = 1.15078, 1 / 25.4
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- calibration knobs
# Every number here is a published-ish central value; tune per region / new science.
ENGINE_VERSION = 5   # bump when reports change meaning; the start page lists only runs from the current engine
K = dict(
    members=1000,
    # NHC official Atlantic 5-yr mean track error (n mi) by lead hour, ~2020-2024 verification
    track_err_nm={0: 10, 12: 23, 24: 35, 36: 46, 48: 58, 72: 85, 96: 118, 120: 155},
    # NHC official intensity mean absolute error (kt) by lead hour
    int_err_kt={0: 2, 12: 5, 24: 7.5, 36: 9, 48: 10, 72: 12, 96: 13, 120: 14},
    err_autocorr_hourly=0.97,          # serial correlation of forecast errors (DeMaria et al. 2009)
    holland_B=(1.0, 1.6),              # uniform range; weak / post-tropical storms are broad
    rmax_lognorm_sigma=0.35,
    land_factor=0.85,                  # open-terrain 1-min wind -> suburban 10 m mean
    gust_factor=1.49,                  # WMO (Harper et al. 2010) 3-s / 1-min, "in-land" exposure
    nws_gust_sigma=0.18,               # lognormal spread of NWS gust forecast error (event scale)
    nws_gust_sigma_hourly=0.08,
    convective_gust_mult=1.35,         # median thunderstorm-outflow gust / grid gust (calibration knob; outflows run 1.2-1.6x)
    blend_C_mph=10.0,                  # regularizer for vortex/NWS ratio when storm is far
    dir_sigma=15.0, dir_sigma_hourly=8.0,  # NWS wind-direction forecast error (deg), event scale and hourly, at gusty winds
    inflow_deg=20.0,                   # surface inflow angle of a tropical cyclone's wind toward its centre
    rain_sigma=0.45,                   # lognormal spread of QPF error (WPC QPF verification order)
    kd_Vb=26.7, kd_R=0.9, kd_alpha=0.095,  # Kaplan & DeMaria (1995) inland decay
    # fragility medians (3-s gust, mph) + lognormal beta. Hazus-like, wood-frame 1-2 story, shingle roof
    frag_cosmetic=(70, .25), frag_minor=(95, .20), frag_moderate=(120, .20), frag_severe=(145, .20),
    # Building damage by construction class. Loads are judged against what the building was designed for:
    # rho = (3-s gust at mean roof height / ASCE 7 design gust at that height)^2, the ratio of actual to design wind
    # pressure, so one scale serves a house, a warehouse and a tower. design_mph: ASCE 7-22 Risk Category II basic speed
    # (3-s gust, 33 ft, exposure C); 110 mph is typical of the interior US (override with --design-mph near the coast).
    # Per class: (median rho, beta on gust speed) for cosmetic / minor / moderate / severe. WSF reproduces the wood-frame
    # curves above at 110 mph (70/95/120/145 mph open-terrain gust); the other classes are engineering estimates of how
    # much later their weakest parts (membrane edges, glazing, metal panels, cladding) give way, after the ordering of the
    # Hazus building classes' damage functions (FEMA Hazus Hurricane Technical Manual, 2021). Calibrate against claims data.
    design_mph=110.0,
    dmg_class=dict(WSF=[(.405, .25), (.746, .20), (1.19, .20), (1.74, .20)], WMF=[(.44, .25), (.80, .20), (1.27, .20), (1.85, .20)],
                   MLR=[(.50, .25), (.90, .22), (1.45, .22), (2.10, .22)], MBI=[(.45, .25), (.80, .22), (1.25, .22), (1.85, .22)],
                   MRB=[(.60, .25), (1.05, .22), (1.70, .22), (2.50, .22)]),
    frag_limb=(55, .30), frag_tree=(80, .30), wet_soil_tree_reduction=0.20,
    outage_g50=60.0, outage_scale=7.0,  # logistic per-customer outage vs peak gust (Guikema-style form)
    tree_height_m=15.0, fall_kappa=2.0,  # von Mises concentration of fall direction around downwind
    # tree mechanics (HWIND / ForestGALES family). Green-wood MOR for eastern hardwoods (USDA Wood Handbook ~45-60 MPa)
    tree_MOR=50e6, tree_fknot=0.85, tree_green_density=900.0, tree_Creg=260.0, tree_overhang=1.14,
    tree_res_cov=0.35,
    # The static moment model over-predicts load on small, flexible crowns, so resistance is scaled once so that
    # a typical mature open-grown hardwood (18 m tall, 12 m crown) has a median critical gust of 95 mph on dry
    # soil, in line with post-hurricane urban tree surveys (Duryea et al. 2007). Relative differences between
    # trees (height, slenderness, crown size, local wind) still come from the mechanics.
    tree_ref=(18.0, 12.0, 95.0), tree_min_h=8.0,                 # lognormal scatter of tree resistance (species, defects, root health)
    tree_LAI=3.5, tree_Cd=0.2,         # canopy drag for the CFD: peak-summer leaf area index, element Cd
    wood_frac=0.15,                    # bare-branch share of the summer canopy area (wood area index / LAI)
    leafoff_drag=0.35,                 # crown drag of a bare deciduous tree vs. in full leaf (Vollsinger et al. 2005)
    leaf=1.0,                          # current leaf-on fraction; set per run from MODIS
    house_dx=1.0, house_half=45.0, house_steps=3600,   # nested fine CFD around the house (wall / roof pressures)
    stories=2, story_h=3.0, roof_pitch=6 / 12,
    scene_half_m=165, maps_half_m=220,
)

# ---------------------------------------------------------------- tiny utilities
def get(url, data=None, headers=None, timeout=45, raw=False, cache=False):
    p = CACHE / "http" / hashlib.sha1((url + str(data)).encode()).hexdigest()
    if cache and p.exists():
        b = p.read_bytes()
    else:
        req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read()
        if cache:
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    return b if raw else json.loads(b)

def soft(fn, *a, **k):
    """Optional data source: a failure is logged and the report carries on without it."""
    try:
        return fn(*a, **k)
    except Exception as e:
        print(f"  ! {fn.__name__} unavailable: {e}", file=sys.stderr)
        return None

def enu(lat0, lon0):
    """lat/lon -> local east/north metres (equirectangular; fine at < 5 km)."""
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540
    return (lambda lat, lon: ((np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky))

def hav_km(lat1, lon1, lat2, lon2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))

def merc_px(lat, lon, z):
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n

def interp_table(tab, h):
    ks = sorted(tab); return float(np.interp(h, ks, [tab[k] for k in ks]))

def data_uri(img, fmt="JPEG", q=82):
    b = io.BytesIO(); img.save(b, fmt, quality=q) if fmt == "JPEG" else img.save(b, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()

def mosaic(url_tpl, lat, lon, z, half_px_x, half_px_y=None):
    """Stitch XYZ tiles into one image centred on lat/lon. Returns (PIL image, top-left global px)."""
    half_px_y = half_px_y or half_px_x
    cx, cy = merc_px(lat, lon, z)
    x0, y0, x1, y1 = cx - half_px_x, cy - half_px_y, cx + half_px_x, cy + half_px_y
    tiles = [(tx, ty) for tx in range(int(x0 // 256), int(x1 // 256) + 1) for ty in range(int(y0 // 256), int(y1 // 256) + 1)]
    with cf.ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(lambda t: get(url_tpl.format(z=z, x=t[0], y=t[1]), raw=True, cache=True), tiles))
    tx0, ty0 = tiles[0]
    big = Image.new("RGB", ((tiles[-1][0] - tx0 + 1) * 256, (tiles[-1][1] - ty0 + 1) * 256))
    for (tx, ty), b in zip(tiles, imgs):
        big.paste(Image.open(io.BytesIO(b)).convert("RGB"), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = x0 - tx0 * 256, y0 - ty0 * 256
    return big.crop((round(ox), round(oy), round(ox + 2 * half_px_x), round(oy + 2 * half_px_y))), (x0, y0)

ESRI_SAT = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_DARK = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
ESRI_DARK_REF = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# ---------------------------------------------------------------- data sources
def geocode(q):
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", q)
    if m:
        return float(m[1]), float(m[2]), q
    # 1) Esri World Geocoder: rooftop/parcel "PointAddress" matches (Census only interpolates along the street)
    e = soft(get, "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?" +
             urllib.parse.urlencode({"f": "json", "SingleLine": q, "maxLocations": 1, "outFields": "Addr_type,Score", "countryCode": "USA"}))
    for c in (e or {}).get("candidates", []):
        if c["attributes"]["Addr_type"] in ("PointAddress", "Subaddress") and c["score"] >= 90:
            return c["location"]["y"], c["location"]["x"], c["address"].upper()
    # 2) Census street-range interpolation (can land on the street or a neighbour; use --house-at to correct)
    d = get("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?" +
            urllib.parse.urlencode({"address": q, "benchmark": "Public_AR_Current", "format": "json"}))
    hits = d["result"]["addressMatches"]
    if not hits:
        sys.exit(f"Address not found: {q!r}. US addresses only; or pass 'lat,lon'.")
    h = hits[0]
    return h["coordinates"]["y"], h["coordinates"]["x"], h["matchedAddress"]

DUR = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?")

def hourly(layer, conv=lambda v: v, spread=False):
    """NWS gridpoint layer (ISO interval values) -> {utc hour: value}. spread=True splits totals."""
    out = {}
    for v in layer.get("values", []):
        t, d = v["validTime"].split("/")
        m = DUR.fullmatch(d); n = max(1, int(m[1] or 0) * 24 + int(m[2] or 0))
        t0 = dt.datetime.fromisoformat(t).astimezone(dt.timezone.utc)
        for i in range(n):
            out[t0 + dt.timedelta(hours=i)] = None if v["value"] is None else conv(v["value"]) / (n if spread else 1)
    return out

def nws(lat, lon):
    p = get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")["properties"]
    g = get(p["forecastGridData"])["properties"]
    kmh = lambda v: v / 1.609344
    al = soft(get, f"https://api.weather.gov/alerts/active?point={lat:.4f},{lon:.4f}") or {"features": []}
    return dict(
        tz=p["timeZone"], office=p["gridId"], place=p["relativeLocation"]["properties"],
        sust=hourly(g["windSpeed"], kmh), gust=hourly(g["windGust"], kmh), dir=hourly(g["windDirection"]),
        qpf=hourly(g["quantitativePrecipitation"], lambda v: v * MM2IN, spread=True),
        pop=hourly(g["probabilityOfPrecipitation"]), thunder=hourly(g.get("probabilityOfThunder", {"values": []})),
        alerts=[dict(event=f["properties"]["event"], headline=f["properties"]["headline"],
                     severity=f["properties"]["severity"], ends=f["properties"].get("ends") or f["properties"].get("expires"),
                     desc=(f["properties"].get("description") or "")[:900]) for f in al["features"]],
    )

def google_weather(lat, lon, hours=96):
    """Google Maps Platform Weather API hourly forecast (independent of NWS). Needs GOOGLE_MAPS_KEY (Demo Key works)."""
    key = os.environ.get("GOOGLE_MAPS_KEY")
    if not key:
        return None
    out = dict(gust={}, sust={}, dir={}, qpf={}, thunder={}); tok = ""
    while len(out["gust"]) < hours:
        d = get("https://weather.googleapis.com/v1/forecast/hours:lookup?" + urllib.parse.urlencode(dict(
            key=key, **{"location.latitude": lat, "location.longitude": lon}, hours=hours, pageSize=24, pageToken=tok)))
        for h in d["forecastHours"]:
            t = dt.datetime.fromisoformat(h["interval"]["startTime"].replace("Z", "+00:00")); wd = h["wind"]
            mph = lambda v: v["value"] / 1.609344 if v.get("unit", "KILOMETERS_PER_HOUR").startswith("KILO") else v["value"]
            out["gust"][t] = mph(wd["gust"]); out["sust"][t] = mph(wd["speed"]); out["dir"][t] = wd["direction"]["degrees"]
            out["qpf"][t] = h["precipitation"]["qpf"]["quantity"] * MM2IN; out["thunder"][t] = h.get("thunderstormProbability")
        tok = d.get("nextPageToken")
        if not tok:
            break
    return out

def google_now(lat, lon):
    """Google current conditions + last 24 h (hourly history). Gust now feeds the nowcast; history QPF is the rain
    that already fell (NWS grids only carry forecasts, so past rain from them is mostly missing)."""
    key = os.environ.get("GOOGLE_MAPS_KEY")
    if not key:
        return None
    q = {"key": key, "location.latitude": lat, "location.longitude": lon}
    c = get("https://weather.googleapis.com/v1/currentConditions:lookup?" + urllib.parse.urlencode(q))
    h = get("https://weather.googleapis.com/v1/history/hours:lookup?" + urllib.parse.urlencode(dict(q, hours=24, pageSize=24)))
    kmh = lambda v: (v or {}).get("value", 0) / 1.609344
    return dict(t=dt.datetime.fromisoformat(c["currentTime"].replace("Z", "+00:00")), gust=kmh(c["wind"].get("gust")),
                sust=kmh(c["wind"].get("speed")), past24_in=sum(x["precipitation"]["qpf"]["quantity"] for x in h["historyHours"]) * MM2IN)

def fp_google(lat, lon, r=None):
    """Google's building outline under the point (Geocoding API v4 SearchDestinations, building filter)."""
    key = os.environ.get("GOOGLE_MAPS_KEY")
    if not key:
        return []
    req = urllib.request.Request("https://geocode.googleapis.com/v4/geocode/destinations", method="POST",
                                 data=json.dumps({"locationQuery": {"location": {"latitude": lat, "longitude": lon},
                                                                     "placeFilter": {"structureType": "BUILDING"}}}).encode(),
                                 headers={"Content-Type": "application/json", "X-Goog-Api-Key": key,
                                          "X-Goog-FieldMask": "destinations.primary.displayPolygon,destinations.primary.structureType"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    out = []
    for x in d.get("destinations", []):
        poly = x["primary"].get("displayPolygon") or {}
        if x["primary"].get("structureType") == "BUILDING" and poly.get("type") == "Polygon":
            out.append(dict(ring=[(p[1], p[0]) for p in poly["coordinates"][0]], height=None, src="Google building outline"))
    return out

def archive_forecast(lat, lon, times, w, g):
    """Keep every NWS / Google forecast we fetch. Neither service archives its own, so this is what lets them join the
    calibrated blend later. ponytail: one JSONL per site; move to a DB if many sites run hourly."""
    try:
        p = CACHE / "archive" / f"{lat:.3f},{lon:.3f}.jsonl"; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(dict(issued=dt.datetime.now(dt.timezone.utc).isoformat(), t=[t.isoformat() for t in times],
                                    nws=[w["models"]["nws"][i] for i in range(len(times))],
                                    google=g and [g["gust"].get(t) for t in times])) + "\n")
    except OSError as e:
        print(f"  ! forecast archive: {e}", file=sys.stderr)

def nowcast(w, now_obs, times, efold_h=6.0):
    """Scale the blended gust by observed / forecast at the current hour, relaxing back to the forecast with an
    e-folding time of efold_h (standard nowcast-to-NWP blending). Ratio clipped to [0.5, 2]."""
    if not now_obs or not times or not w["gust"].get(times[0]) or now_obs["gust"] <= 0:
        return 1.0
    r = min(2.0, max(0.5, now_obs["gust"] / w["gust"][times[0]]))
    for i, t in enumerate(times):
        if w["gust"].get(t):
            w["gust"][t] *= r ** math.exp(-i / efold_h)
    return r

def blend(w, g, times):
    """Two-model blend: geometric-mean gust (both models anchored equally; ponytail: equal weights, verify against ASOS
    obs and weight by recent skill once forecasts are archived). Model disagreement widens the Monte Carlo spread:
    sigma_event^2 = sigma_nws^2 + (mean |ln(g_nws / g_google)| / 2)^2."""
    if g:
        for t in times:
            a, b = w.setdefault("thunder", {}).get(t), g.get("thunder", {}).get(t)
            if b is not None:
                w["thunder"][t] = max(a or 0, b)   # either model calling for storms is enough for the gust tail
    w["models"] = dict(nws=[w["gust"].get(t) for t in times], google=[g["gust"].get(t) for t in times] if g else None)
    if not g:
        return 0.0
    r = []
    for t in times:
        a, b = w["gust"].get(t), g["gust"].get(t)
        if a and b:
            w["gust"][t] = math.sqrt(a * b); r.append(abs(math.log(a / b)))
        elif b and not a:
            w["gust"][t] = b
    return float(np.mean(r)) / 2 if r else 0.0

def pick_storm(lat, lon, want):
    ss = get("https://www.nhc.noaa.gov/CurrentStorms.json")["activeStorms"]
    if want:
        ss = [s for s in ss if want.lower() in (s["id"].lower(), s["name"].lower())]
        if not ss:
            sys.exit(f"No active storm matches {want!r}.")
    if not ss:
        return None
    s = min(ss, key=lambda s: hav_km(lat, lon, s["latitudeNumeric"], s["longitudeNumeric"]))
    d = float(hav_km(lat, lon, s["latitudeNumeric"], s["longitudeNumeric"]))
    if not want and d > 1500:
        print(f"  nearest active storm {s['name']} is {d:.0f} km away; running weather-only mode", file=sys.stderr)
        return None
    return s

MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
LL = r"(\d+\.\d)([NS])\s+(\d+\.\d)([EW])"

def parse_tcm(txt):
    """NHC/WPC forecast advisory (TCM) -> issue time + [(utc, lat, lon, vmax_kt)]; vmax 0 = dissipated."""
    m = re.search(r"(\d{4}) UTC \w{3} (\w{3}) (\d+) (\d{4})", txt)
    issue = dt.datetime(int(m[4]), MON[m[2]], int(m[3]), int(m[1][:2]), tzinfo=dt.timezone.utc)
    def when(day, hh):
        t = issue.replace(day=1, hour=0) + dt.timedelta(days=int(day) - 1, hours=int(hh))
        return t if t >= issue - dt.timedelta(days=2) else (t + dt.timedelta(days=32)).replace(day=int(day))
    ll = lambda g: (float(g[0]) * (1 if g[1] == "N" else -1), float(g[2]) * (-1 if g[3] == "W" else 1))
    c = re.search(r"CENTER LOCATED NEAR " + LL + r" AT (\d\d)/(\d\d)00Z", txt)
    v0 = int(re.search(r"MAX SUSTAINED WINDS\s+(\d+) KT", txt)[1])
    pts = [(when(c[5], c[6]), *ll(c.groups()[:4]), v0)]
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        f = re.search(r"(?:FORECAST|OUTLOOK) VALID (\d\d)/(\d\d)00Z\s*(.*)", line)
        if not f:
            continue
        t = when(f[1], f[2])
        if "DISSIPATED" in f[3]:
            pts.append((t, None, None, 0)); break
        p = re.search(LL, f[3])
        w = re.search(r"MAX WIND\s+(\d+) KT", " ".join(lines[i:i + 3]))
        pts.append((t, *ll(p.groups()), int(w[1]) if w else pts[-1][3]))
    return issue, pts

def storm_track(s):
    html = get(s["forecastAdvisory"]["url"], raw=True).decode("latin1")
    txt = re.sub(r"<[^>]+>", "", re.search(r"<pre[^>]*>(.*?)</pre>", html, re.S)[1])
    return parse_tcm(txt)

def flood_zone(lat, lon):
    d = get("https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query?" + urllib.parse.urlencode(dict(
        geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", inSR=4326, spatialRel="esriSpatialRelIntersects",
        outFields="FLD_ZONE,ZONE_SUBTY,SFHA_TF", returnGeometry="false", f="json")))
    f = d.get("features") or []
    return f[0]["attributes"] if f else {"FLD_ZONE": "unmapped"}

def elevation_ft(lat, lon):
    return get(f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Feet&wkid=4326")["value"]

def soil(lat, lon):
    q = (f"SELECT TOP 1 mu.muname, c.hydgrp FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point({lon} {lat})') k "
         "JOIN mapunit mu ON mu.mukey=k.mukey JOIN component c ON c.mukey=mu.mukey WHERE c.majcompflag='Yes' ORDER BY c.comppct_r DESC")
    t = get("https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest", data=json.dumps({"format": "JSON", "query": q}).encode(),
            headers={"Content-Type": "application/json"})["Table"][0]
    return {"name": t[0], "group": (t[1] or "C")[0]}

def river(lat, lon):
    b = 0.2
    gs = get(f"https://api.water.noaa.gov/nwps/v1/gauges?bbox.xmin={lon-b}&bbox.ymin={lat-b}&bbox.xmax={lon+b}&bbox.ymax={lat+b}&srid=EPSG_4326")["gauges"]
    if not gs:
        return None
    g = min(gs, key=lambda g: hav_km(lat, lon, g["latitude"], g["longitude"]))
    meta = get(f"https://api.water.noaa.gov/nwps/v1/gauges/{g['lid']}")
    sf = get(f"https://api.water.noaa.gov/nwps/v1/gauges/{g['lid']}/stageflow")
    fc = [p["primary"] for p in (sf.get("forecast") or {}).get("data", []) if p["primary"] > -900]
    obs = [p["primary"] for p in sf["observed"]["data"] if p["primary"] > -900]
    cats = meta.get("flood", {}).get("categories", {})
    lvl = lambda k: (lambda v: v if v is not None and v > -900 else None)((cats.get(k) or {}).get("stage"))   # -9999 = not defined
    return dict(lid=g["lid"], name=g["name"], dist_mi=round(float(hav_km(lat, lon, g["latitude"], g["longitude"])) / 1.609, 1),
                lat=g["latitude"], lon=g["longitude"], stage=obs[-1] if obs else None, fcst_max=max(fc) if fc else None,
                minor=lvl("minor"), action=lvl("action"),
                units=sf["observed"].get("primaryUnits", "ft"))

# ---------------------------------------------------------------- buildings
def quadkey(lat, lon, z):
    x, y = (int(v // 256) for v in merc_px(lat, lon, z)); s = ""
    for i in range(z, 0, -1):
        m = 1 << (i - 1); s += str((1 if x & m else 0) + (2 if y & m else 0))
    return s

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

def fp_osm(lat, lon, r):
    q = f"[out:json][timeout:25];way[building](around:{r},{lat},{lon});out tags geom;"
    cp = CACHE / "osm" / (hashlib.sha1(q.encode()).hexdigest() + ".json")    # outlines change slowly: reuse for 30 days
    if cp.exists() and time.time() - cp.stat().st_mtime < 30 * 86400:
        d = json.loads(cp.read_text())
    else:
        # public Overpass servers time out under load: every mirror, twice, with a pause between rounds
        for k, url in enumerate(OVERPASS * 2):
            try:
                d = get(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=40)
                break
            except Exception:
                if k == 2 * len(OVERPASS) - 1:
                    raise
                if k == len(OVERPASS) - 1:
                    time.sleep(5)
        cp.parent.mkdir(parents=True, exist_ok=True); tmp = cp.with_suffix(".tmp"); tmp.write_text(json.dumps(d)); tmp.replace(cp)
    out = []
    for e in d["elements"]:
        t = e.get("tags", {})
        h = float(re.sub(r"[^\d.]", "", t["height"]) or 0) if "height" in t else (float(t["building:levels"]) * 3 + 1.5 if "building:levels" in t else None)
        out.append(dict(ring=[(p["lat"], p["lon"]) for p in e["geometry"]], height=h, src="OpenStreetMap",
                        tags={k: t[k] for k in ("building", "building:levels", "roof:shape", "building:material", "start_date", "name") if k in t}))
    return out

def fp_microsoft(lat, lon, r):
    q = quadkey(lat, lon, 9)
    links = get("https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv", raw=True, cache=True).decode()
    urls = [row["Url"] for row in csv.DictReader(io.StringIO(links)) if row["QuadKey"] == q and row["Location"] == "UnitedStates"]
    dlat, dlon = r / 110540, r / (111320 * math.cos(math.radians(lat))); out = []
    for u in urls:
        for line in gzip.open(io.BytesIO(get(u, raw=True, cache=True, timeout=300)), "rt"):
            c = line[line.find("[[[") + 3:line.find("]", line.find("[[[")) + 1]
            x, y = map(float, c.strip("[]").split(","))
            if abs(y - lat) < dlat and abs(x - lon) < dlon:
                g = json.loads(line)
                h = g["properties"].get("height")
                out.append(dict(ring=[(p[1], p[0]) for p in g["geometry"]["coordinates"][0]],
                                height=h if h and h > 0 else None, src="Microsoft ML footprints"))
    return out

def fp_usa_structures(lat, lon, r):
    d = get("https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0/query?" +
            urllib.parse.urlencode(dict(geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", inSR=4326, distance=r,
                                        units="esriSRUnit_Meter", outFields="HEIGHT,PROP_ADDR", outSR=4326, f="json")))
    return [dict(ring=[(p[1], p[0]) for p in f["geometry"]["rings"][0]], height=f["attributes"]["HEIGHT"],
                 src="FEMA/ORNL USA Structures") for f in d.get("features", [])]

def min_rect(pts):
    """Minimum-area bounding rectangle of 2-D points -> (cx, cy, length, width, heading_deg of long axis, area)."""
    pts = np.asarray(pts, float); best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        c, s = np.cos(a), np.sin(a); q = pts @ np.array([[c, -s], [s, c]])
        w, h = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        if best is None or w * h < best[0]:
            best = (w * h, a, w, h, (q.min(0) + q.max(0)) / 2)
    A, a, w, h, cq = best; c, s = np.cos(a), np.sin(a)
    cx, cy = np.array([[c, -s], [s, c]]) @ cq
    ang = math.degrees(a) + (0 if w >= h else 90)
    return float(cx), float(cy), float(max(w, h)), float(min(w, h)), (90 - ang) % 180, float(A)

# ---------------------------------------------------------------- buildings
SRC_RANK = {"Google building outline": 0, "OpenStreetMap": 1, "Microsoft ML footprints": 2, "FEMA/ORNL USA Structures": 3}

def poly_area(ring):
    x, y = np.asarray(ring, float).T
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)

def _cover(a, b):
    """Share of outline a's area inside outline b (grid sample over a's bounding box)."""
    lo, hi = a["ring"].min(0), a["ring"].max(0); st = max(.25, min(1.0, math.sqrt(a["area"]) / 12))
    X, Y = np.meshgrid(np.arange(lo[0] + st / 2, hi[0], st), np.arange(lo[1] + st / 2, hi[1], st))
    m = in_poly(X, Y, a["ring"])
    return float((m & in_poly(X, Y, b["ring"])).sum() / max(m.sum(), 1))

def merge_footprints(fps, to_m):
    """Every source's outlines -> one building list in local metres, with no size limits (a house, a warehouse and a
    railway station are all buildings). One building found by several datasets is kept once, from the best source
    (Google, OSM, Microsoft, FEMA/ORNL); the others fill in its height and tags. A lower-ranked outline covering several
    kept buildings is a merged block or parcel and is dropped."""
    items = []
    for f in fps:
        xs, ys = to_m(np.array([p[0] for p in f["ring"]]), np.array([p[1] for p in f["ring"]]))
        ring = np.c_[xs, ys]
        if len(ring) > 3 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        A = poly_area(ring) if len(ring) >= 3 else 0
        if not 12 <= A <= 2e5:
            continue
        cx, cy, L, W, hd, _ = min_rect(ring)
        items.append(dict(cx=cx, cy=cy, L=L, W=W, heading=hd, area=A, height=f.get("height") or None, src=f["src"],
                          tags=dict(f.get("tags") or {}), ring=ring, rank=SRC_RANK.get(f["src"], 4), lo=ring.min(0), hi=ring.max(0)))
    items.sort(key=lambda b: (b["rank"], b["area"]))
    kept, wraps = [], {}
    for b in items:
        hits = [k for k in kept if (b["lo"] < k["hi"]).all() and (k["lo"] < b["hi"]).all() and max(_cover(b, k), _cover(k, b)) > .5]
        if not hits:
            kept.append(b)
        elif len(hits) == 1 and 1 / 2.5 < b["area"] / hits[0]["area"] < 2.5:   # the same building in another dataset
            k = hits[0]
            k["height"] = k["height"] or b["height"]
            for t, v in b["tags"].items():
                k["tags"].setdefault(t, v)
        elif len(hits) == 1 and b["area"] > hits[0]["area"] and _cover(hits[0], b) > .8:
            wraps.setdefault(id(hits[0]), []).append(b)          # a bigger outline around a kept one
    # consensus: when two other datasets agree (areas within 1.5x) on a bigger outline around a kept one, the kept
    # outline is a building part (a wing, a tower or a roof section mapped on its own) and the agreed building wins
    for i, k in enumerate(kept):
        w = wraps.get(id(k), [])
        for c in w:
            agree = {o["src"] for o in w if max(o["area"], c["area"]) / min(o["area"], c["area"]) < 1.5}
            if len(agree) >= 2:
                best = min((o for o in w if o["src"] in agree), key=lambda o: o["rank"])
                for o in w:
                    if o["src"] in agree:
                        best["height"] = best["height"] or o["height"]
                        for t, v in o["tags"].items():
                            best["tags"].setdefault(t, v)
                kept[i] = best
                break
    return kept

def pick_target(blds):
    """The building the address point is in (best source, then smallest: a wing rather than the whole campus); failing
    that the nearest outline within 25 m (rooftop geocodes can land on a drive or a courtyard)."""
    on = [b for b in blds if in_poly(0.0, 0.0, b["ring"])]
    if on:
        return min(on, key=lambda b: (b["rank"], b["area"]))
    near = [(float(seg_dist(np.zeros(1), np.zeros(1), b["ring"])[0]), b) for b in blds]
    near = [nb for nb in near if nb[0] < 25]
    return min(near, key=lambda nb: nb[0])[1] if near else None

# Construction classes after the Hazus hurricane model's building classes (FEMA 2021, ch. 5), grouped by the parts that
# fail first in wind: WSF wood single-family; WMF low-rise multi-unit residential; MLR low-rise masonry / steel
# commercial and institutional; MBI pre-engineered metal and industrial; MRB engineered mid/high-rise (concrete, steel)
CLASSES = {"WSF": "Wood-frame house", "WMF": "Low-rise apartments / townhouses", "MLR": "Low-rise commercial (masonry/steel)",
           "MBI": "Metal / industrial building", "MRB": "Engineered mid/high-rise"}
_RES1 = {"house", "detached", "semidetached_house", "bungalow", "cabin", "farm", "terrace", "static_caravan", "hut", "shed",
         "garage", "garages", "carport", "houseboat"}
_RESM = {"apartments", "residential", "dormitory"}
_IND = {"industrial", "warehouse", "manufacture", "hangar", "factory", "barn", "farm_auxiliary", "greenhouse", "service",
        "storage_tank", "silo", "cowshed", "stable", "transportation", "depot"}

DMG_TEXT = {   # (cosmetic title, cosmetic note, minor title, minor note) per construction class
    "WSF": ("A few shingles, gutters or siding", "Loose edges and ridge caps go first.",
            "Real roof damage (needs a roofer)", "Hazus 'minor' damage: some roof cover lost, a window or door."),
    "WMF": ("A few shingles, gutters or siding", "Roof edges, ridge caps and siding go first.",
            "Roof cover or windows damaged", "Some roof covering lost, a window or door broken."),
    "MLR": ("Flashing, coping or signage damage", "Roof edge metal, parapet coping and signs go first.",
            "Roof membrane or windows damaged", "Membrane peeled at edges and corners, some glazing broken."),
    "MBI": ("Trim, gutters or panel fasteners", "Trim, gutters and loose panel fasteners go first.",
            "Roof or wall panels or doors damaged", "Panels torn at edges and corners, large doors pushed in."),
    "MRB": ("Loose cladding, flashing or rooftop gear", "Rooftop equipment, flashing and cladding edges go first.",
            "Glazing or cladding panels damaged", "Some curtain-wall glass or cladding panels lost, mostly at corners."),
}

def classify(b):
    """Construction class, storeys and roof form from OSM tags, measured height and footprint size."""
    t = b["tags"].get("building", "yes"); A = b["area"]
    h = b.get("eave") or b.get("h") or 0          # storeys from the main roof edge, not a tower or spire above it
    try:
        lv = float(b["tags"].get("building:levels", ""))
    except ValueError:
        lv = None
    stories = int(lv) if lv else 2 if str(b.get("h_src", "")).startswith("assumed") else max(1, round((h - 0.5) / 3.4)) if h else None
    if (stories or 0) >= 5 or h > 17:
        cls = "MRB"
    elif t in _IND or (t not in _RES1 | _RESM and A > 2500 and b.get("flat")):
        cls = "MBI"
    elif t in _RES1 or (t == "yes" and A < 450 and (stories or 2) <= 3):
        cls = "WSF"
    elif t in _RESM:
        cls = "WMF"
    else:
        cls = "MLR"
    if cls == "MBI" and not lv:
        stories = 1                                # high-bay industrial: one tall storey, not eave / 3.4 m
    shape = b["tags"].get("roof:shape")
    flat = b["flat"] if b.get("flat") is not None else (shape == "flat" if shape else cls not in ("WSF", "WMF"))
    return dict(cls=cls, label=CLASSES[cls], stories=stories, flat=bool(flat), osm_type=t if t != "yes" else None)

def measure_buildings(blds, P, ground, R):
    """Roof heights from LiDAR for every building: first returns that are neither ground, water nor noise (class 6
    building and class 1 unclassified -- many surveys classify buildings sparsely or not at all) gridded to 1 m, minus
    the 3DEP ground, inside each outline. A building counts as measured when returns cover 40 % of its outline (newer
    buildings show bare ground). Sets h (roof top, 95th pct: robust to overhanging crowns), eave (low roof edge, 10th
    pct) and flat (top-to-edge spread < 1.5 m). A building whose outline is mostly ground in the survey (built later) stays
    unmeasured. Returns the 1 m height-above-ground grid (NaN = no return)."""
    n = 2 * R; x, y, z, c = P.T
    m = np.isin(c, (1, 6)) & (np.abs(x) < R) & (np.abs(y) < R)
    i = np.clip((R - y[m]).astype(int), 0, n - 1); j = np.clip((x[m] + R).astype(int), 0, n - 1)
    dsm = np.full((n, n), -np.inf); np.maximum.at(dsm, (i, j), z[m])
    ndsm = np.where(np.isfinite(dsm), dsm - ground, np.nan)
    for b in blds:
        cells = _cells(b, R)
        if cells is None:
            continue
        # 1 m in from the outline: footprints and LiDAR disagree by a metre or so, and edge cells catch ground, shrubs
        # and gutters that would drag the eave down (small buildings keep their full outline)
        inner = np.zeros((n, n), bool); inner[cells] = True; inner = nd.binary_erosion(inner)
        core = np.nonzero(inner) if inner.sum() >= max(8, .3 * cells[0].size) else cells
        v = ndsm[core]; v = v[np.isfinite(v) & (v < 600)]
        # a survey flown before the building stood (or mid-construction) shows ground and scaffolding, not a roof
        if len(v) < max(8, .4 * core[0].size) or (v > 2.5).mean() < .6:
            continue
        v = v[v > 1.5]
        if len(v) >= max(8, .4 * core[0].size):
            p10, p90, p95 = np.percentile(v, [10, 90, 95])
            # flat or low-slope: a 1/4:12 (2 %) drainage slope still rises ~1 % of the span, metres on a big warehouse
            b.update(h=float(p95), eave=float(p10), flat=bool(p90 - p10 < max(1.5, .012 * b["W"])), h_src="LiDAR")
    return ndsm

def _cells(b, R):
    """Grid indices (rows, cols) of the 1 m site cells inside a building outline, or None."""
    n = 2 * R; lo, hi = b["ring"].min(0), b["ring"].max(0)
    c0, c1 = max(0, int(lo[0] + R)), min(n, int(math.ceil(hi[0] + R))); r0, r1 = max(0, int(R - hi[1])), min(n, int(math.ceil(R - lo[1])))
    if c0 >= c1 or r0 >= r1:
        return None
    rr, cc = np.mgrid[r0:r1, c0:c1]
    m = in_poly(cc - R + .5, R - rr - .5, b["ring"])
    return (rr[m], cc[m]) if m.any() else None

def building_field(blds, R, ndsm=None):
    """1 m roof-height-above-ground grid for the CFD and the building mask. Measured buildings use their LiDAR roof
    (holes filled with the building's median); the rest get a flat roof at h or a hip roof (eave + pitch x distance to
    the outline, capped at h: the straight-skeleton roof, so L-shapes get valleys)."""
    n = 2 * R; Hb = np.zeros((n, n), np.float32); pitch = K["roof_pitch"]
    for b in blds:
        cells = _cells(b, R)
        if cells is None:
            continue
        if ndsm is not None and b.get("h_src") == "LiDAR":
            v = ndsm[cells]; med = np.nanmedian(np.where(v > 1.5, v, np.nan))
            v = np.clip(np.where(np.isfinite(v) & (v > 1.5), v, med), max(2.0, .5 * b["eave"]), b["h"])
        elif b.get("flat"):
            v = np.full(cells[0].size, b["h"])
        else:
            X, Y = cells[1] - R + .5, R - cells[0] - .5
            v = np.minimum(b["eave"] + pitch * seg_dist(X, Y, b["ring"]), b["h"])
        Hb[cells] = np.maximum(Hb[cells], v)
    return Hb

def detect_roofs(img, mpp):
    """Find dark, neutral (shingle-like) roofs in RGB imagery, adaptive to the image's colour cast.
    ponytail: colour heuristic; misses white/red roofs. Upgrade: a segmentation model (SAM / footprint net)."""
    im = nd.uniform_filter(np.asarray(img, float), (3, 3, 1))
    r, g, b = im[..., 0], im[..., 1], im[..., 2]; v = im.max(2); bg = b / (g + 1)
    # vegetation reflects little blue relative to green; shingles and pavement do not
    tex = np.sqrt(np.maximum(nd.uniform_filter(v * v, 7) - nd.uniform_filter(v, 7) ** 2, 0))
    roof = (bg > max(0.68, np.percentile(bg, 55))) & (v < np.percentile(v, 62))
    shade = (v < np.percentile(v, 12)) & (bg > 0.55)
    roof = nd.binary_fill_holes(nd.binary_closing(nd.binary_opening(roof, iterations=2), iterations=3))
    lab, n = nd.label(roof); out = []
    for i, sl in enumerate(nd.find_objects(lab), 1):
        m = lab[sl] == i; a = m.sum() * mpp * mpp
        if not 50 < a < 1500:
            continue
        # merge the roof's own shaded slope: adjacent very dark pixels within ~4 m
        pad = int(4 / mpp); y0, x0 = max(0, sl[0].start - pad), max(0, sl[1].start - pad)
        big = np.zeros_like(roof[y0:sl[0].stop + pad, x0:sl[1].stop + pad])
        big[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = m
        grown = big | (nd.binary_dilation(big, iterations=pad) & shade[y0:y0 + big.shape[0], x0:x0 + big.shape[1]])
        grown = nd.binary_fill_holes(nd.binary_closing(grown, iterations=2))
        ys, xs = np.nonzero(grown)
        cx, cy, L, W, hd, A = min_rect(np.c_[xs, -ys] * mpp)
        if grown.sum() * mpp * mpp / A < .55 or L / W > 2.6 or not 60 < A < 900 \
                or tex[y0:y0 + big.shape[0], x0:x0 + big.shape[1]][big].mean() > np.percentile(tex, 75):
            continue
        out.append(dict(px=(cx / mpp + x0, -cy / mpp + y0), L=L, W=W, heading=hd, area=grown.sum() * mpp * mpp))
    return out

def detect_trees(img):
    """Canopy mask: low blue/green (chlorophyll), darker and bumpier than lawn."""
    im = nd.uniform_filter(np.asarray(img, float), (3, 3, 1))
    r, g, b = im[..., 0], im[..., 1], im[..., 2]; v = im.max(2); bg = b / (g + 1)
    tex = np.sqrt(np.maximum(nd.uniform_filter(v * v, 9) - nd.uniform_filter(v, 9) ** 2, 0))
    m = (bg < min(0.6, np.percentile(bg, 40))) & (g >= r) & (v < np.percentile(v, 45)) & ((tex > np.percentile(tex, 45)) | (v < np.percentile(v, 22)))
    return nd.binary_closing(nd.binary_opening(m, iterations=1), iterations=4)

def rect_ring(cx, cy, L, W, heading):
    a = math.radians(90 - heading); u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    return [list(map(float, c + sx * L / 2 * u + sy * W / 2 * n)) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

# ---------------------------------------------------------------- terrain
def dem_grid(lat, lon, half_m, n, z):
    """Terrarium elevation tiles -> n x n grid (m) centred on lat/lon spanning +-half_m."""
    mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z
    hp = half_m / mpp + 2
    img, _ = mosaic(TERRARIUM, lat, lon, z, hp)
    a = np.asarray(img, float); e = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    s = np.linspace(2, e.shape[0] - 3, n)
    return nd.map_coordinates(e, np.meshgrid(s, s, indexing="ij"), order=1)  # row 0 = north

def merc_m(lat, lon):
    R = 6378137.0
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

def dem_3dep(lat, lon, half_m, n):
    """USGS 3DEP seamless DEM (1 m where lidar exists) -> n x n grid (m), row 0 = north, pixel-centred."""
    X, Y = merc_m(lat, lon); h = half_m / math.cos(math.radians(lat))
    b = get("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?" + urllib.parse.urlencode(dict(
        bbox=f"{X-h},{Y-h},{X+h},{Y+h}", bboxSR=3857, imageSR=3857, size=f"{n},{n}", format="tiff", pixelType="F32",
        noData=-9999, interpolation="RSP_BilinearInterpolation", f="image")), raw=True, cache=True, timeout=90)
    a = np.asarray(Image.open(io.BytesIO(b)), float)
    if a.shape != (n, n) or (a < -1000).any():
        raise ValueError("3DEP returned gaps")
    return a

EPT_INDEX = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"

def _in_poly(geom, x, y):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    inside = False
    for poly in polys:
        for ring in poly:
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
                if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                    inside = not inside
    return inside

def lidar_points(lat, lon, half):
    """USGS 3DEP airborne LiDAR (Entwine Point Tiles, AWS open data) within +-half m.
    Returns local (x east, y north, z, class) array and the survey name. Needs laspy[lazrs]."""
    import laspy
    idx = get(EPT_INDEX, cache=True, timeout=120)
    hits = [f["properties"] for f in idx["features"] if _in_poly(f["geometry"], lon, lat)]
    if not hits:
        raise LookupError("no 3DEP lidar survey covers this point")
    year = lambda h: max([int(y) for y in re.findall(r"(?:19|20)\d\d", h["name"])] or [0])
    survey = max(hits, key=year)                                   # newest survey
    base = survey["url"].rsplit("/", 1)[0] + "/"
    ept = get(base + "ept.json", cache=True)
    X, Y = merc_m(lat, lon); c = math.cos(math.radians(lat)); h = half / c
    q = (X - h, Y - h, X + h, Y + h); b = ept["bounds"]; size = b[3] - b[0]
    hier = {}; keys = []; stack = ["0-0-0-0"]
    hier.update(get(base + "ept-hierarchy/0-0-0-0.json", cache=True))
    while stack:
        k = stack.pop(); d, x, y, z = map(int, k.split("-")); s = size / 2 ** d
        x0, y0 = b[0] + x * s, b[1] + y * s
        if x0 > q[2] or x0 + s < q[0] or y0 > q[3] or y0 + s < q[1] or k not in hier:
            continue
        if hier[k] == -1:
            hier.update(get(base + f"ept-hierarchy/{k}.json", cache=True))
        keys.append(k)
        stack += [f"{d+1}-{2*x+i}-{2*y+j}-{2*z+l}" for i in (0, 1) for j in (0, 1) for l in (0, 1)]
    def node(k):
        las = laspy.read(io.BytesIO(get(base + f"ept-data/{k}.laz", raw=True, cache=True, timeout=120)))
        m = (las.x > q[0]) & (las.x < q[2]) & (las.y > q[1]) & (las.y < q[3])
        return np.c_[(np.asarray(las.x[m]) - X) * c, (np.asarray(las.y[m]) - Y) * c, las.z[m], las.classification[m]]
    with cf.ThreadPoolExecutor(12) as ex:
        P = np.vstack(list(ex.map(node, keys)))
    return P, survey["name"], year(survey)

def in_poly(x, y, ring):
    """Even-odd point-in-polygon for arrays of points."""
    ring = np.asarray(ring, float); inside = np.zeros(np.shape(x), bool)
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, 0)):
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1)
        inside ^= c
    return inside

def roof_heights(P, ground, half, ring):
    """Eave and ridge (m above ground) from LiDAR returns inside the outline: building class 6, or unclassified
    returns when the survey never classified buildings. None unless raised returns cover most of the footprint
    (a house built after the survey shows bare ground there)."""
    x, y, z, c = P.T
    cls = 6 if (c == 6).any() else 1
    m = in_poly(x, y, ring) & (c == cls)
    if m.sum() < 30:
        return None
    i = np.clip((half - y[m]).astype(int), 0, 2 * half - 1); j = np.clip((x[m] + half).astype(int), 0, 2 * half - 1)
    h = z[m] - ground[i, j]; ok = (h > 2) & (h < 25)
    xs, ys = np.meshgrid(np.arange(-half, half) + .5, np.arange(half, -half, -1) - .5)
    cells = in_poly(xs, ys, ring).sum()
    if len(set(zip(i[ok], j[ok]))) < 0.6 * cells:
        return None
    h = h[ok]
    return float(np.percentile(h, 8)), float(np.percentile(h, 99))   # eave ~ low roof edge, ridge ~ top

def canopy_from_lidar(P, ground, half, now_mask):
    """1 m canopy height model (first-surface minus 3DEP ground), restricted to canopy visible in
    *current* imagery (trees cut since the survey are dropped). Individual trees by variable-window
    local maxima (Popescu & Wynne 2004), crowns by nearest-treetop assignment."""
    n = ground.shape[0]
    i = np.clip((half - P[:, 1]).astype(int), 0, n - 1); j = np.clip((P[:, 0] + half).astype(int), 0, n - 1)
    veg = P[:, 3] != 2
    dsm = np.full((n, n), -1e9); np.maximum.at(dsm, (i[veg], j[veg]), P[veg, 2])
    chm = np.where(dsm > -1e8, dsm - ground, 0)
    chm = nd.grey_closing(np.clip(chm, 0, 45), size=3)               # fill pits between pulses
    chm = np.where(now_mask, chm, 0)
    sm = nd.gaussian_filter(chm, 1.0)
    tops = np.zeros_like(sm, bool)
    for lo_h, hi_h in ((3, 8), (8, 14), (14, 20), (20, 50)):          # window grows with height
        hm = (lo_h + hi_h) / 2; win = max(3, int(round(2.51503 + 0.00901 * hm * hm)) | 1)
        tops |= (sm >= nd.maximum_filter(sm, win)) & (sm >= lo_h) & (sm < hi_h)
    lab, nt = nd.label(tops)
    if nt == 0:
        return chm, []
    cen = np.array(nd.center_of_mass(tops, lab, range(1, nt + 1))).astype(int)
    mark = np.zeros_like(lab); mark[cen[:, 0], cen[:, 1]] = np.arange(1, nt + 1)
    _, (ii, jj) = nd.distance_transform_edt(mark == 0, return_indices=True)
    crown = np.where(chm > 2, mark[ii, jj], 0)
    area = nd.sum(np.ones_like(chm), crown, range(1, nt + 1))
    out = []
    for t, (r, cc) in enumerate(cen):
        h = float(sm[r, cc]); cw = 2 * math.sqrt(area[t] / math.pi)
        if h < 3 or cw < 2:
            continue
        out.append(dict(x=cc - half + 0.5, y=half - r - 0.5, h=round(h, 1), cw=round(min(cw, 0.9 * h + 4), 1)))
    return chm, out

def kzt_by_direction(dem, half_m, z_h=6.0):
    """ASCE 7 topographic factor Kzt for wind FROM each of 8 directions (2-D ridge/escarpment form, Exp. B)."""
    n = dem.shape[0]; c = (n - 1) / 2; step = 2 * half_m / (n - 1); out = []
    z0 = dem[int(c), int(c)]
    for k in range(8):
        a = math.radians(k * 45)                              # direction wind comes FROM (0 = north)
        d = np.arange(0, half_m, step)
        row, col = c - d * math.cos(a) / step, c + d * math.sin(a) / step
        prof = nd.map_coordinates(dem, [row, col], order=1)   # upwind profile from the house
        H = z0 - prof.min()
        if H < 4.5:
            out.append(dict(kzt=1.0, H=float(H), Lh=None)); continue
        half = np.nonzero(prof <= z0 - H / 2)[0]
        Lh = max(float(d[half[0]]) if len(half) else half_m, 1)
        if H / Lh > 0.5:
            Lh = 2 * H
        if H / Lh < 0.2:
            out.append(dict(kzt=1.0, H=float(H), Lh=Lh)); continue
        K1 = 1.30 * (H / Lh); K3 = math.exp(-3 * z_h / Lh)    # K2 = 1: house treated as at the crest (conservative)
        out.append(dict(kzt=float((1 + K1 * K3) ** 2), H=float(H), Lh=Lh))
    return out

# ---------------------------------------------------------------- neighbourhood CFD
def macdonald_z0(h, dx, upwind):
    """Macdonald et al. (1998) morphometric roughness length from an obstacle-height grid (wind along +x)."""
    hu = h[:upwind]
    obst = hu > 2
    if obst.mean() < 0.01:
        return 0.05
    lp = obst.mean()
    lf = np.maximum(np.diff(hu, axis=0), 0).sum() * dx / (hu.size * dx * dx)
    hm = hu[obst].mean()
    zd = hm * (1 + 4.43 ** -lp * (lp - 1))
    z0 = hm * (1 - zd / hm) * math.exp(-(0.5 * 1.2 / 0.4 ** 2 * (1 - zd / hm) * lf) ** -0.5)
    return float(np.clip(z0, 0.05, 1.5))

def _sector(args):
    """One wind direction: voxelise the site in a wind-aligned lattice, run LBM-LES, resample to the world grid."""
    import cfd
    th, M, dx, steps, out_half, out_nz, *seed = args
    E, Hb, Hc, Hbase, R = M["E"], M["Hb"], M["Hc"], M["Hbase"], M["R"]
    half = R / math.sqrt(2) - 2; n = int(2 * half / dx) // 2 * 2
    a = math.radians(th + 180)                                    # compass direction the wind blows toward
    d = np.array([math.sin(a), math.cos(a)]); p = np.array([-d[1], d[0]])
    s = (np.arange(n) - n / 2 + 0.5) * dx
    I, J = np.meshgrid(s, s, indexing="ij")
    wx, wy = I * d[0] + J * p[0], I * d[1] + J * p[1]
    rc = [R - wy - 0.5, wx + R - 0.5]
    smp = lambda A, o=1: nd.map_coordinates(A, rc, order=o, mode="nearest")
    e, hb, hc, hbs = smp(E), smp(Hb, 0), smp(Hc), smp(Hbase)
    z0 = macdonald_z0(np.maximum(hb, 0.5 * hc), dx, n // 2)
    zmin = e.min() - dx
    nz = int((e.max() - zmin + 2.5 * max(14, np.percentile(hc, 99.5), hb.max())) / dx)
    Z = zmin + (np.arange(nz) + 0.5) * dx
    solid = Z[None, None, :] < (e + hb)[..., None]
    solid[:, :, 0] = True
    # wind-tunnel style inflow conditioning: floor roughness over the first 8 % of the fetch makes the
    # boundary layer turbulent before it reaches the site (smooth LES inflow over-predicts wakes)
    rng = np.random.default_rng(int(th))
    ni = max(6, n // 12); g0 = solid.argmin(2)
    blocks = (rng.random((ni, n)) < 0.22) & (np.arange(ni)[:, None] > 2)
    for i, j in zip(*np.nonzero(blocks)):
        solid[i, j, g0[i, j]:g0[i, j] + rng.integers(1, 3)] = True
    lad = np.where(hc > hbs + 0.5, M.get("lai", K["tree_LAI"]) / np.maximum(hc - hbs, 1), 0)
    canopy = (Z[None, None, :] > (e + hbs)[..., None]) & (Z[None, None, :] < (e + hc)[..., None]) & ~solid
    drag = np.where(canopy, K["tree_Cd"] * lad[..., None], 0).astype(np.float32)
    steps = max(steps, int(4 * n / 0.06))           # >= 4 flow-throughs at the lattice inflow speed; mean over the last 55 %
    mu, sig = cfd.best_solver()(solid, drag, dx, z0, steps=steps, avg_from=int(steps * 0.45), seed=seed[0] if seed else 0,
                                **_wall_model(Z[None, None, :] < e[..., None], 0.05))
    # resample to a north-up world grid centred on the house
    nw = int(2 * out_half / dx); xs = (np.arange(nw) - nw / 2 + 0.5) * dx
    zw0 = math.floor(nd.map_coordinates(E, [[R], [R]], order=1)[0]) - 2
    zs = zw0 + (np.arange(out_nz) + 0.5) * dx
    Y, X, Zw = np.meshgrid(xs[::-1], xs, zs, indexing="ij")         # row 0 = north
    li = (X * d[0] + Y * d[1]) / dx + n / 2 - 0.5
    lj = (X * p[0] + Y * p[1]) / dx + n / 2 - 0.5
    lk = (Zw - zmin) / dx - 0.5
    smp3 = lambda A: nd.map_coordinates(A, [li, lj, lk], order=1, mode="nearest")
    ux, uy, uz, sg = smp3(mu[0]), smp3(mu[1]), smp3(mu[2]), smp3(sig)
    east, north = ux * d[0] + uy * p[0], ux * d[1] + uy * p[1]
    return dict(dir=th, z0=z0, dx=dx, z_base=zw0, u=np.stack([east, north, uz]).astype(np.float32),
                sig=sg.astype(np.float32), n_lattice=[n, n, nz])

def seg_dist(X, Y, ring):
    """Distance from points to a polygon's boundary."""
    d = np.full(np.shape(X), np.inf)
    for (x1, y1), (x2, y2) in zip(ring, np.roll(np.asarray(ring, float), -1, 0)):
        vx, vy = x2 - x1, y2 - y1; t = np.clip(((X - x1) * vx + (Y - y1) * vy) / (vx * vx + vy * vy or 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(X - x1 - t * vx, Y - y1 - t * vy))
    return d

def roof_field(X, Y, house, others):
    """Roof surface height above ground at points (house-centred metres). Hip roofs everywhere: eave + pitch x distance
    to the outline, capped at the ridge -- for any outline this is the straight-skeleton roof, so L-shapes get valleys."""
    out = np.zeros(np.shape(X), np.float32); pitch = K["roof_pitch"]
    for r in others:
        a = math.radians(r["heading"]); ux, uy = math.sin(a), math.cos(a); dx, dy = X - r["cx"], Y - r["cy"]
        d = np.minimum(r["L"] / 2 - np.abs(dx * ux + dy * uy), r["W"] / 2 - np.abs(dx * uy - dy * ux)); m = d >= 0
        eave = max(2.5, r["h"] - 1.5)
        out[m] = np.maximum(out[m], np.minimum(eave + pitch * d[m], r["h"] + 1.5))
    m = in_poly(X, Y, house["ring"])
    out[m] = np.maximum(out[m], np.minimum(house["eave_m"] + pitch * seg_dist(X[m], Y[m], house["ring"]), house["ridge_m"]))
    return out

def wall_list(ring):
    """Outline -> walls (midpoint, outward normal, length, facing bearing). Near-collinear and stub vertices are
    dropped so a wall is a wall, not a dozen survey points."""
    p = [list(map(float, q)) for q in ring]
    if len(p) > 1 and math.dist(p[0], p[-1]) < .05:
        p.pop()
    changed = True
    while changed and len(p) > 3:
        changed = False
        for i in range(len(p)):
            a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
            ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]; lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
            if lu < .7 or lv < .7 or abs(math.asin(max(-1, min(1, (ux * vy - uy * vx) / (lu * lv))))) < .2 and ux * vx + uy * vy > 0:
                p.pop(i); changed = True; break
    sg = 1 if sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(p, p[1:] + p[:1])) > 0 else -1
    out = []
    for a, b in zip(p, p[1:] + p[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy); nx, ny = sg * dy / L, -sg * dx / L
        out.append(dict(x=round((a[0] + b[0]) / 2, 2), y=round((a[1] + b[1]) / 2, 2), nx=round(nx, 4), ny=round(ny, 4),
                        len=round(L, 2), fb=round(math.degrees(math.atan2(nx, ny)) % 360, 1)))
    return out

HOUSE_CFD_VERSION = "sw12"   # bump when the nested-pressure method changes: cached results are then recomputed

def _house_sector(args):
    """Fine LBM-LES box around the building, nested in the coarse run for the same direction: the coarse mean wind and
    turbulence at the box inlet drive it, so upwind buildings and trees are felt. Obstacles come from the 1 m LiDAR roof
    grid (Hb). Returns mean pressure coefficients per wall and over the roof in the ASCE 7 / AIJ convention: referenced
    to the approach flow's mean dynamic pressure at mean roof height, plus that approach flow (mean, turbulence) so the
    loads can be scaled by the matching gust."""
    import cfd
    th, MF, F, dx, steps, half = args
    E, R, Hb, house = MF["E"], MF["R"], MF["Hb"], MF["house"]
    n = int(2 * half / dx) // 2 * 2
    a = math.radians(th + 180); d = np.array([math.sin(a), math.cos(a)]); p = np.array([-d[1], d[0]])
    s_ = (np.arange(n) - n / 2 + 0.5) * dx
    I, J = np.meshgrid(s_, s_, indexing="ij"); wx, wy = I * d[0] + J * p[0], I * d[1] + J * p[1]
    smp = lambda A, X, Y, o=1: nd.map_coordinates(A, [R - Y - 0.5, X + R - 0.5], order=o, mode="nearest")
    e, hc, hbs, hb = smp(E, wx, wy), smp(MF["Hc"], wx, wy), smp(MF["Hbase"], wx, wy), smp(Hb, wx, wy, 0)
    H = house["ridge_m"]; zmin = e.min() - dx
    nz = int((e.max() - zmin + max(32, 3 * H, np.percentile(hc, 99) + 12, 1.5 * hb.max())) / dx)   # room above zr for the reference
    Z = zmin + (np.arange(nz) + 0.5) * dx
    solid = Z[None, None, :] < (e + hb)[..., None]; solid[:, :, 0] = True
    lad = np.where(hc > hbs + 0.5, MF["lai"] / np.maximum(hc - hbs, 1), 0)
    canopy = (Z[None, None, :] > (e + hbs)[..., None]) & (Z[None, None, :] < (e + hc)[..., None]) & ~solid
    drag = np.where(canopy, K["tree_Cd"] * lad[..., None], 0).astype(np.float32)
    # inlet plane from the coarse field (world east/north/up -> wind-aligned along/across/up)
    nw, cdx = F["u"].shape[1], F["dx"]
    yy, zz = np.meshgrid(wy[0], Z, indexing="ij"); xx = np.broadcast_to(wx[0][:, None], yy.shape)
    cc = [nw / 2 - yy / cdx - 0.5, nw / 2 + xx / cdx - 0.5, (zz - F["z_base"]) / cdx - 0.5]
    ue, un, uz = (nd.map_coordinates(F["u"][k], cc, order=1, mode="nearest") for k in range(3))
    ui = np.stack([ue * d[0] + un * d[1], ue * p[0] + un * p[1], uz])
    sg = nd.map_coordinates(F["sig"], cc, order=1, mode="nearest")
    mu, _, pr = cfd.best_solver()(solid, drag, dx, F["z0"], steps=steps, avg_from=int(steps * 0.4), inflow=(ui, sg), pressure=True,
                                  **_wall_model(Z[None, None, :] < e[..., None], 0.03))
    # Reference = the undisturbed approach flow at mean roof height hm (wind-tunnel / ASCE 7 convention). Near the ground
    # the inlet plane can sit in the wake of upwind houses and trees, so the profile is read above the roughness sublayer
    # (zr = max(20 m, 2.5 x the upwind obstacle height)) across the middle 60 % of the inlet and carried down to hm along
    # the log law for the site's roughness z0 (no displacement: shelter belongs in Cp, as in the code's exposure profiles).
    # Static pressure is read at the same height a few cells into the box, where it is undisturbed.
    hm = (house["eave_m"] + house["ridge_m"]) / 2; jm = np.arange(int(.2 * n), int(.8 * n)); z0 = F["z0"]
    up = np.maximum(hb[:max(4, int(.25 * n))], hc[:max(4, int(.25 * n))])
    zr = float(min(max(20.0, 2.5 * (up[up > 2].mean() if (up > 2).any() else 0), 1.5 * hm), Z[-1] - e.max() - 3 * dx))
    kr = np.clip(np.rint((e[0, jm] + zr - zmin) / dx - .5).astype(int), 0, nz - 1)
    U_zr = float(ui[0, jm, kr].mean()); sig_ref = float(sg[jm, kr].mean())
    U_ref = U_zr * math.log(max(hm, 2 * z0) / z0) / math.log(zr / z0)
    # static reference: the free-stream static pressure upstream at zr (ASCE 7 / wind-tunnel convention), from the
    # inlet band across the middle of the box, which the building cannot reach
    rows = np.arange(3, max(4, int(.08 * n)))
    band = (np.abs((Z[None, None, :] - e[rows][:, :, None]) - zr) <= max(2.5, 2 * dx)) & ~solid[rows]
    band[:, :jm[0]] = False; band[:, jm[-1] + 1:] = False
    p_ref = float(np.median(pr[rows][band] if band.any() else pr[rows][~solid[rows]]))
    cp = 2 * (pr - p_ref) / max(U_ref, 0.05) ** 2
    def sample(X, Y, zagl):
        """Surface pressure from the first fluid cell outside the wall (solid cells are skipped)."""
        g = smp(E, X, Y)
        li = np.rint((X * d[0] + Y * d[1]) / dx + n / 2 - 0.5).astype(int); lj = np.rint((X * p[0] + Y * p[1]) / dx + n / 2 - 0.5).astype(int)
        lk = np.rint((g + zagl - zmin) / dx - 0.5).astype(int)
        ok = (li >= 0) & (li < n) & (lj >= 0) & (lj < n) & (lk >= 0) & (lk < nz)
        li, lj, lk = li[ok], lj[ok], lk[ok]; fl = ~solid[li, lj, lk]
        return cp[li[fl], lj[fl], lk[fl]]
    walls = []
    for w in MF["walls"]:
        if w["len"] < 3 * dx:          # under 3 cells: corner separation on a wall that short is not resolved
            walls.append(None); continue
        t = np.linspace(-0.4, 0.4, 5) * w["len"]; z = np.linspace(1.0, max(1.5, house["eave_m"] - 0.5), 4)
        T_, Z_ = np.meshgrid(t, z)
        X = w["x"] - w["ny"] * T_ + w["nx"] * 0.8 * dx; Y = w["y"] + w["nx"] * T_ + w["ny"] * 0.8 * dx   # first cell outside
        v = sample(X.ravel(), Y.ravel(), Z_.ravel()); walls.append(round(float(v.mean()), 3) if len(v) and np.isfinite(v).all() else None)
    lo_, hi_ = np.min(house["ring"], 0), np.max(house["ring"], 0)
    st = max(1.0, dx)
    GX, GY = np.meshgrid(np.arange(lo_[0], hi_[0], st), np.arange(lo_[1], hi_[1], st)); m = in_poly(GX, GY, house["ring"])
    rv = sample(GX[m], GY[m], smp(Hb, GX[m], GY[m], 0) + 0.8 * dx)
    ok = len(rv) and np.isfinite(rv).all()
    # internal pressure of an enclosed building with leakage spread over its walls: the length-weighted mean of the
    # external wall pressures (Holmes, Wind Loading of Structures, ch. 4); net loads are external minus internal
    wv = [(c, w["len"]) for c, w in zip(walls, MF["walls"]) if c is not None]
    cpi = round(sum(c * L for c, L in wv) / sum(L for _, L in wv), 3) if wv else None
    return th, dict(walls=walls, cpi=cpi, roof=[round(float(rv.mean()), 3), round(float(np.percentile(rv, 5)), 3)] if ok else None,
                    ref=dict(U=round(U_ref, 4), sig=round(sig_ref, 4), hm=round(hm, 2), zr=round(zr, 1), U_zr=round(U_zr, 4),
                             top=round(H, 2), z0=round(z0, 3)), dx=dx, n=[n, n, nz])

def _wall_model(terrain, z0_ground):
    """Ground surface-stress wall model (GPU solver only): terrain cells are free-slip with the log-law stress in the
    first cell above. z0_ground is the unresolved ground roughness (lawn, shrubs, cars); buildings and tree crowns are
    resolved explicitly."""
    import cfd
    return dict(terrain=terrain, z0_ground=z0_ground) if cfd.best_solver() is cfd.solve_metal else {}

def _workers(n):
    """CFD jobs in parallel on CPU cores; one at a time when they run on the GPU (it is already fully used)."""
    import cfd
    return 1 if cfd.best_solver() is not cfd.solve else min(n, max(1, (os.cpu_count() or 2) - 2))

def nested_box(house, out_half):
    """Fine-grid size for the building: 16 cells over its height on the GPU solver (8 on the CPU one) where the cell
    budget allows (AIJ / COST 732 ask for >= 10), a box reaching max(25 m, 4 H) past the outline, inside the coarse field,
    and five flow-throughs."""
    import cfd
    gpu = cfd.best_solver() is not cfd.solve
    per_h, budget, dmin = (16, 6e6, 0.5) if gpu else (8, 4e6, K["house_dx"])   # cube benchmark: 16/H gives Cp 0.62, X_R 1.75 H
    span = float(np.ptp(np.asarray(house["ring"]), 0).max()); H = house["ridge_m"]
    dx = float(np.clip(H / per_h, dmin, max(dmin, span / 60)))
    half = min(span / 2 + max(25, 4 * H), out_half - 5)
    nz = (max(32, 3 * H) + 10) / dx
    cells = (2 * half / dx) ** 2 * nz
    if cells > budget:
        dx *= (cells / budget) ** (1 / 3)
    steps = int(np.clip(5 * (2 * half / dx) / 0.06, 3600, 20000))
    return round(dx, 2), half, steps

def house_pressures(MF, fields, key):
    """Nested fine CFD for each simulated direction (cached like the coarse fields)."""
    dx, half, steps = nested_box(MF["house"], min(F["u"].shape[1] * F["dx"] / 2 for F in fields.values()))
    out, todo = {}, []
    for th, F in fields.items():
        p = CACHE / "cfd" / f"{key}-house-{th:05.1f}-{dx}-{steps}-{HOUSE_CFD_VERSION}.json"
        if p.exists():
            out[th] = json.loads(p.read_text())
        else:
            todo.append(th)
    if todo:
        print(f"  CFD: fine {dx:g} m grid around the building, {2 * half:.0f} m box, {len(todo)} direction(s) (parallel)…", file=sys.stderr)
        with cf.ProcessPoolExecutor(_workers(len(todo))) as ex:
            for th, r in ex.map(_house_sector, [(th, MF, fields[th], dx, steps, half) for th in todo]):
                p = CACHE / "cfd" / f"{key}-house-{th:05.1f}-{dx}-{steps}-{HOUSE_CFD_VERSION}.json"; tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(r)); tmp.replace(p); out[th] = r
    return out

def cp_ok(r):
    """Is one direction's nested-CFD result physically possible? Judged on net pressures (external minus the internal
    pressure, which an enclosed building takes from the mean of its walls), because a sheltered building can sit wholly
    in an upwind wake (every face negative) or in front of a tree belt (every face positive) and still be right: net wall
    pressures stay within about +-1.5 of the approach dynamic pressure, and the roof net mean between -2.5 and +0.5
    with its edges no worse than -4 (ASCE 7 / tunnel envelopes for low and mid-rise buildings)."""
    cpi = r.get("cpi")
    w = [v - cpi for v in r.get("walls") or [] if v is not None and math.isfinite(v)] if cpi is not None else []
    rf = r.get("roof") or [None, None]
    if len(w) < 2 or any(v is None or not math.isfinite(v) for v in rf):
        return False
    # walls rising well above the reference height see more than q(hm): scale the bound by (U(top) / U(hm))^2
    g = r.get("ref") or {}; z0 = g.get("z0", .3)
    k = (math.log(max(g["top"], 2 * z0) / z0) / math.log(max(g["hm"], 2 * z0) / z0)) ** 2 if g.get("top") and g.get("hm") else 1.0
    return max(abs(v) for v in w) <= 1.5 * k and -2.5 <= rf[0] - cpi <= 0.5 and rf[1] - cpi >= -4


def trim_walls(r, lens, share=0.10):
    """A separated corner or a short wall can read beyond the physical bound while the rest of the building is sound.
    Such walls are set aside (shown with ASCE 7 values) as long as they make up at most `share` of the sampled perimeter;
    a direction with more than that still fails cp_ok as a whole."""
    cpi, g = r.get("cpi"), r.get("ref") or {}
    if cpi is None:
        return r
    z0 = g.get("z0", .3)
    k = (math.log(max(g["top"], 2 * z0) / z0) / math.log(max(g["hm"], 2 * z0) / z0)) ** 2 if g.get("top") and g.get("hm") else 1.0
    W = r.get("walls") or []
    bad = [i for i, v in enumerate(W) if v is not None and math.isfinite(v) and abs(v - cpi) > 1.5 * k]
    tot = sum(lens[i] for i, v in enumerate(W) if v is not None)
    if bad and sum(lens[i] for i in bad) <= share * tot:
        r = dict(r, walls=[None if i in bad else v for i, v in enumerate(W)], trimmed=len(bad))
    return r


def leaf_state(lat, lon):
    """Current leaf-on fraction from NASA MODIS LAI (MCD15A2H, 500 m, 8-day; ~3-week latency): median of the 3x3 km
    block now vs. its July-August peak. Deciduous drag in autumn/winter is a fraction of summer's."""
    today = dt.date.today(); y = today.year if today.month >= 9 else today.year - 1
    A = lambda d: f"A{d.year}{d.timetuple().tm_yday:03d}"
    def med(d0, d1, cache):
        r = get(f"https://modis.ornl.gov/rst/api/v1/MCD15A2H/subset?latitude={lat:.4f}&longitude={lon:.4f}&band=Lai_500m"
                f"&startDate={A(d0)}&endDate={A(d1)}&kmAboveBelow=1&kmLeftRight=1", cache=cache, timeout=60,
                headers={"Accept": "application/json"})   # without an Accept header the service answers 500
        out = []
        for s in r["subset"]:
            v = np.array(s["data"], float); v = v[v <= 100] * 0.1          # 249-255 are fill / water / urban codes
            if len(v):
                out.append((s["calendar_date"], float(np.median(v))))
        return out
    summer, recent = med(dt.date(y, 7, 1), dt.date(y, 8, 31), True), med(today - dt.timedelta(days=60), today, False)
    if not summer or not recent:     # every 500 m pixel flagged urban / water: no leaf signal, keep full-leaf drag
        raise LookupError("MODIS has no vegetated pixels here (urban core)")
    peak = max(v for _, v in summer)
    when, now = recent[-1]
    return dict(factor=round(min(1.0, now / peak), 2), lai_now=round(now, 2), lai_peak=round(peak, 2), date=when)

def field_ok(r):
    """Physical check on a coarse field: above the roofs the wind must still be the approach flow. A run that went
    unstable or stalled comes out uniformly slow. Over 89 cached site runs the top output level carried 0.35-1.3 of the
    log-law speed for the inflow roughness (median 1.03); the three broken ones 0.10-0.17. The bar sits between."""
    import cfd
    top = (r["u"].shape[3] - 0.5) * r["dx"]
    return float(np.linalg.norm(r["u"][..., -1], axis=0).mean()) >= 0.25 * float(cfd.log_profile(np.array(top), r["z0"]))

def wind_fields(M, sectors, key, dx=2.5, steps=1900, out_half=140, out_nz=14):
    """Run (or load cached) CFD for each wind direction. Results depend only on the site, not the forecast,
    so they are cached and reused on every forecast update. A direction that fails field_ok is rerun once with new
    synthetic turbulence; if it fails again it is left out (members use the nearest good direction)."""
    todo, out = [], {}
    path = lambda th: CACHE / "cfd" / f"{key}-{th:05.1f}-{dx}-{steps}.npz"
    for th in sectors:
        if path(th).exists():
            z = np.load(path(th), allow_pickle=True); r = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
            if field_ok(r):
                out[th] = r; continue
            for f in path(th).parent.glob(f"{key}-house-{th:05.1f}-*"):   # nested runs driven by the bad field go too
                f.unlink()
        todo.append(th)
    for attempt in (0, 1):
        if not todo:
            break
        print(f"  CFD: {len(todo)} wind direction(s), LBM-LES, ~{2 * steps // 1000 + 1} min each…", file=sys.stderr)
        with cf.ProcessPoolExecutor(_workers(len(todo))) as ex:
            res = list(ex.map(_sector, [(th, M, dx, steps, out_half, out_nz, attempt) for th in todo]))
        todo = []
        for th, r in zip([r["dir"] for r in res], res):
            if field_ok(r):
                path(th).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path(th), **r); out[th] = r
            else:
                todo.append(th)
    for th in todo:
        print(f"  ! CFD for wind from {th:g}° failed its check twice; members from there use the nearest good direction", file=sys.stderr)
    return out

def probe(F, x, y, z_agl, E, R):
    """Sample a CFD field at world (x east, y north) and height above local ground. Returns (|U|/Uref, sigma/Uref)."""
    dx = F["dx"]; nw = F["u"].shape[1]
    ground = nd.map_coordinates(E, [[R - y - 0.5], [x + R - 0.5]], order=1)[0]
    c = [[nw / 2 - y / dx - 0.5], [nw / 2 + x / dx - 0.5], [(ground + z_agl - F["z_base"]) / dx - 0.5]]
    u = [nd.map_coordinates(F["u"][i], c, order=1, mode="nearest")[0] for i in range(3)]
    return float(math.sqrt(sum(v * v for v in u))), float(nd.map_coordinates(F["sig"], c, order=1, mode="nearest")[0])

def exposure_ratio(z0, z=10.0, z0_open=0.03, zb=60.0):
    """Mean-wind ratio local/open-terrain at height z with a blending height (Wieringa 1986 / ESDU)."""
    return math.log(zb / z0_open) / math.log(zb / z0) * math.log(z / z0) / math.log(z / z0_open)

def gust_ratio(Kmean, sig, z0, G_open=None, g=3.0, z_agl=10.0):
    """Local 3-s gust / open-terrain 10 m gust.
    gust = mean + g*sigma (Hunt et al. 1976; Lawson's gust-equivalent mean, g = 3). sigma is the CFD-resolved
    turbulence, which already contains the synthetic inflow turbulence. Open-terrain reference: same formula with
    surface-layer intensity 1/ln(10/z0_open). The NWS gust sets the absolute level; this ratio only redistributes it."""
    I_open = 1 / math.log(10 / 0.03)
    sig = max(sig, Kmean / math.log(max(z_agl, 1.0) / z0 + 1))   # floor: surface-layer intensity (ESDU) the coarse grid can't resolve
    return exposure_ratio(z0) * (Kmean + g * sig) / (1 + g * I_open)

# ---------------------------------------------------------------- tree mechanics
def tree_mechanics(h, cw):
    """Per-tree resistance (hybrid-mechanistic, after HWIND / ForestGALES: Peltola et al. 1999, Gardiner et al. 2000).
    DBH from crown size and height (Jucker et al. 2017 global angiosperm allometry)."""
    cw = max(cw, 0.45 * h)   # lidar crown splitting under-sizes crowns in clusters; typical open-grown crown/height ~0.45
    dbh = 0.557 * (h * cw) ** 0.809 * math.exp(0.3 ** 2 / 2) / 100          # m
    M_break = math.pi / 32 * K["tree_MOR"] * K["tree_fknot"] * dbh ** 3      # N m
    stem_kg = 0.5 * math.pi * (dbh / 2) ** 2 * h * K["tree_green_density"]
    M_uproot = K["tree_Creg"] * stem_kg                                    # N m (Nicoll et al. 2006)
    return dbh, M_break * TREE_CAL[0], M_uproot * TREE_CAL[0]

TREE_CAL = [1.0]

def calibrate_trees():
    h, cw, mph = K["tree_ref"]
    TREE_CAL[0] = 1.0
    _, Mb, Mu = tree_mechanics(h, cw)
    TREE_CAL[0] = float(tree_moment(mph * 0.44704, h, cw, leaf=1.0) / min(Mb, Mu))   # survey trees were in full leaf

def tree_moment(u_ms, h, cw, leaf=None):
    """Base bending moment from a gust at crown height: 0.5 rho Cd(u) A u^2 z_c x overhanging-weight factor.
    Cd falls with speed as crowns streamline (Mayhead 1973; Vollsinger et al. 2005); bare crowns catch ~1/3 the wind."""
    leaf = K["leaf"] if leaf is None else leaf
    cd = np.clip(0.6 - (u_ms - 10) * (0.35 / 15), 0.25, 0.6) * (K["leafoff_drag"] + (1 - K["leafoff_drag"]) * leaf)
    crown_depth = 0.65 * h
    A = math.pi / 4 * cw * crown_depth
    zc = h - crown_depth / 2
    return 0.5 * 1.226 * cd * A * u_ms ** 2 * zc * K["tree_overhang"]

# ---------------------------------------------------------------- physics
def holland(r_km, vmax, rmax, B):
    """Holland (1980) gradient-free radial profile normalised so V(rmax) = vmax."""
    x = (rmax / np.maximum(r_km, 0.5)) ** B
    return vmax * np.sqrt(x * np.exp(1 - x))

def willoughby_rmax(vmax_kt, lat):
    """Willoughby et al. (2006) climatological radius of max wind (km)."""
    return 46.4 * np.exp(-0.0155 * vmax_kt * 0.5144 + 0.0169 * abs(lat))

def kaplan_demaria(v0, t_h):
    """Kaplan & DeMaria (1995) inland decay: V(t) = Vb + (R V0 - Vb) exp(-a t)."""
    return K["kd_Vb"] + (K["kd_R"] * v0 - K["kd_Vb"]) * np.exp(-K["kd_alpha"] * t_h)

def frag(g, med_beta, scale=1.0):
    m, b = med_beta
    return ndtr(np.log(np.maximum(g, 1e-3) / (m * scale)) / b)

def ar1(rng, shape, rho):
    """Unit-variance AR(1) series along the last axis (serially correlated forecast errors)."""
    z = rng.standard_normal(shape); out = np.empty(shape); out[..., 0] = z[..., 0]
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + math.sqrt(1 - rho ** 2) * z[..., t]
    return out

def pick_sectors(dir_m, gust, cap, share=0.05, cover=0.9):
    """CFD directions (22.5 deg bins) carrying the damaging wind of the whole ensemble: bins in order of gust^2 weight
    (wind load) over the hours within half of the peak median gust, until `cover` of it is held or a bin adds < `share`."""
    med = np.median(gust, 0)
    wt = gust ** 2 * (med >= .5 * med.max())[None, :]
    b = np.round(dir_m / 22.5).astype(int) % 16
    tot = np.bincount(b.ravel(), wt.ravel(), 16)
    order = np.argsort(-tot); frac = tot[order] / max(tot.sum(), 1e-9)
    n = max(1, min(cap, int(np.searchsorted(np.cumsum(frac), cover) + 1), int((frac >= share).sum())))
    return [float(i * 22.5) for i in order[:n]]

def simulate(lat, lon, times, w, track, site, house, trees, rng, local=None):
    """NHC-style Monte Carlo. Returns hourly percentiles + per-member impact probabilities."""
    N, T = K["members"], len(times)
    tsec = np.array([t.timestamp() for t in times])
    g_nws = np.array([w["gust"].get(t) or (w["sust"].get(t) or 0) * 1.4 for t in times])
    qpf = np.array([w["qpf"].get(t) or 0 for t in times])
    cal = w.get("emos")
    if cal:
        # calibrated spread per hour (wider when models disagree or CAPE is high); the share of error common to the whole
        # event (rho) was measured from the same verification data
        s_t = np.array([s_ if s_ else math.hypot(K["nws_gust_sigma"], K["nws_gust_sigma_hourly"]) for s_ in cal["sigma"]])
        rho = cal["rho"]
        gust = g_nws[None, :] * np.exp(s_t[None, :] * (math.sqrt(rho) * rng.normal(0, 1, (N, 1)) + math.sqrt(1 - rho) * rng.normal(0, 1, (N, T))))
    else:
        sig = math.hypot(K["nws_gust_sigma"], w.get("spread", 0.0))   # model disagreement widens the event-scale spread
        gust = g_nws[None, :] * np.exp(rng.normal(0, sig, (N, 1)) + rng.normal(0, K["nws_gust_sigma_hourly"], (N, T)))
        # thunderstorm outflow: with P(thunder) per hour a member's gust gets a convective multiplier (grid gusts are synoptic).
        # The calibrated path learns this from CAPE instead.
        p_th = np.array([(w["thunder"].get(t) or 0) / 100 for t in times])
        storm = rng.random((N, T)) < p_th[None, :]
        gust = np.where(storm, gust * np.exp(rng.normal(math.log(K["convective_gust_mult"]), .15, (N, T))), gust)
    rain_mult = np.exp(rng.normal(0, K["rain_sigma"], (N, 1)))
    # wind direction per member and hour: the NWS direction plus its forecast error; a member whose storm passes on a
    # different side of the house turns further (added in the track block below)
    d0 = np.array([w["dir"].get(t) if w["dir"].get(t) is not None else np.nan for t in times], float)
    d0 = np.where(np.isnan(d0), np.nanmedian(d0) if np.isfinite(d0).any() else 180.0, d0)
    dir_m = d0[None, :] + rng.normal(0, K["dir_sigma"], (N, 1)) + K["dir_sigma_hourly"] * ar1(rng, (N, T), .9)
    out = {"tracks": None}
    if track:
        issue, pts = track
        live = [p for p in pts if p[1] is not None]
        tp = np.array([p[0].timestamp() for p in live]); la = np.array([p[1] for p in live]); lo = np.array([p[2] for p in live])
        vm = np.array([p[3] for p in live], float)
        end = next((p[0].timestamp() for p in pts if p[1] is None), None)
        # official track, hourly; linear motion extrapolated past the last point
        slope = lambda a: (a[-1] - a[-2]) / (tp[-1] - tp[-2]) if len(tp) > 1 else 0.0
        olat = np.where(tsec > tp[-1], la[-1] + slope(la) * (tsec - tp[-1]), np.interp(tsec, tp, la))
        olon = np.where(tsec > tp[-1], lo[-1] + slope(lo) * (tsec - tp[-1]), np.interp(tsec, tp, lo))
        ov = np.interp(tsec, tp, vm)
        past = tsec > tp[-1]
        # beyond the last forecast point: Kaplan-DeMaria decay, continuous at the last point, never re-intensifying
        ov[past] = np.minimum(vm[-1], kaplan_demaria(vm[-1] / K["kd_R"], (tsec[past] - tp[-1]) / 3600))
        if end:  # linear taper to zero at the advertised dissipation time
            ov[past] *= np.clip((end - tsec[past]) / max(end - tp[-1], 1), 0, 1)
        lead = (tsec - issue.timestamp()) / 3600
        sig_tr = np.array([interp_table(K["track_err_nm"], max(h, 0)) for h in lead]) * 1.852 / 1.2533  # km per axis
        sig_in = np.array([interp_table(K["int_err_kt"], max(h, 0)) for h in lead]) * 1.2533
        rho = K["err_autocorr_hourly"]
        e_al, e_cr, e_in = (ar1(rng, (N, T), rho) for _ in range(3))
        hd = np.gradient(np.c_[olat, olon * np.cos(np.radians(olat))], axis=0)
        hdx, hdy = hd[:, 1], hd[:, 0]; nrm = np.hypot(hdx, hdy) + 1e-9
        ux, uy = hdx / nrm, hdy / nrm                                      # along-track unit (east, north)
        speed_kmh = nrm * 111.0                                            # deg/hour -> km/h
        dx = e_al * sig_tr * ux + e_cr * sig_tr * uy                       # cross unit = (uy, -ux) (right of motion)
        dy = e_al * sig_tr * uy - e_cr * sig_tr * ux
        mlat = olat + dy / 110.54; mlon = olon + dx / (111.32 * np.cos(np.radians(olat)))
        mv = np.maximum(0, ov + e_in * sig_in * (ov > 0))
        rmax = willoughby_rmax(np.maximum(mv, 15), olat) * np.exp(rng.normal(0, K["rmax_lognorm_sigma"], (N, 1)))
        B = rng.uniform(*K["holland_B"], (N, 1))
        def site_gust(la_, lo_, v_, rm_, B_):
            r = hav_km(la_, lo_, lat, lon)
            V = holland(r, v_, rm_, B_)
            # translation asymmetry: cyclonic tangential wind direction vs storm motion
            ex, ey = (lon - lo_) * np.cos(np.radians(lat)), (lat - la_)
            tx, ty = -ey, ex; tn = np.hypot(tx, ty) + 1e-9
            V = V + 0.5 * speed_kmh / 1.852 * (tx * ux + ty * uy) / tn * (V / np.maximum(v_, 1))
            # direction the wind blows from: cyclonic tangent turned toward the centre by the surface inflow angle
            a = math.radians(K["inflow_deg"]); en = np.hypot(ex, ey) + 1e-9
            vx, vy = math.cos(a) * tx / tn - math.sin(a) * ex / en, math.cos(a) * ty / tn - math.sin(a) * ey / en
            return np.maximum(V, 0) * K["land_factor"] * K["gust_factor"] * KT2MPH, r, (np.degrees(np.arctan2(vx, vy)) + 180) % 360
        gv_m, dist_m, dv_m = site_gust(mlat, mlon, mv, rmax, B)
        gv_d, dist_d, dv_d = site_gust(olat, olon, ov, willoughby_rmax(np.maximum(ov, 15), olat), 1.3)
        C = K["blend_C_mph"]
        gust = gust * (gv_m + C) / (gv_d + C)
        # the NWS direction already holds the official storm's turning; a member adds its own track's difference, weighted
        # by how much of its wind the vortex supplies
        dir_m = dir_m + gv_m / (gv_m + C) * ((dv_m - dv_d[None, :] + 180) % 360 - 180)
        rain_mult = rain_mult * np.clip(np.exp(-(dist_m.min(1, keepdims=True) - dist_d.min()) / 400), 0.3, 3)
        out.update(official=dict(lat=olat.round(3).tolist(), lon=olon.round(3).tolist(), vmax=ov.round(1).tolist(),
                                 dist_mi=(dist_d / 1.609).round(1).tolist()),
                   tracks=[dict(lat=mlat[i, ::2].round(3).tolist(), lon=mlon[i, ::2].round(3).tolist()) for i in range(0, N, N // 60)],
                   track_pts=[dict(t=p[0].isoformat(), lat=p[1], lon=p[2], v=p[3]) for p in pts])
    rain = qpf[None, :] * rain_mult                                         # inches per hour, per member
    gq = np.percentile(gust, [5, 25, 50, 75, 95], axis=0).round(1)        # (5, T)
    peak = gust.max(1); hour_of_peak = gust.argmax(1)
    total_rain = rain.sum(1) + site["past_rain_in"]
    wet = np.clip((total_rain - 0.5) / 2.0, 0, 1)                           # 0 dry ... 1 saturated
    dirs_from = np.array([w["dir"].get(t) or 0 for t in times], float)
    dir_m = dir_m % 360
    loc = local or {}
    # each member-hour uses the CFD direction nearest its own wind direction (local ratios are per simulated direction)
    secs = np.array(loc.get("sectors") or [0.0])
    sidx = np.abs((dir_m[..., None] - secs + 180) % 360 - 180).argmin(-1)              # (N, T)
    per = lambda a: None if a is None else np.asarray(a, float)[sidx]
    # hills add ASCE 7 Kzt by each member's direction
    kzt_t = np.array([k["kzt"] for k in site["kzt"]])[((dir_m + 22.5) % 360 // 45).astype(int)] if site.get("kzt") else np.ones((N, T))
    # trees: per-tree bending moment from the CFD-local crown-top gust vs. stem-break and uprooting resistance
    p_tree_hit = np.zeros(N); p_limb_any = np.zeros(N); tree_out = []
    radius = math.sqrt(house["L"] * house["W"] / math.pi)
    Rt = loc.get("trees")
    calibrate_trees()
    for k, t in enumerate(trees):
        if t["h"] < K["tree_min_h"]:       # saplings and shrubs: flexible, outside the model's range, can't harm the house
            tree_out.append(dict(p_fail=0, p_hit=0, p_limb=0, dbh_cm=None, crit_mph=None, mode="small", gust50=None, gust95=None))
            continue
        g_loc = gust * (per(Rt[k]) if Rt is not None else 1.0)
        gpk = g_loc.max(1); u = gpk * 0.44704
        dbh, Mb, Mu = tree_mechanics(t["h"], t["cw"])
        Mu_wet = Mu * (1 - K["wet_soil_tree_reduction"] * wet)
        Mcrit = np.minimum(Mb, Mu_wet)
        p_fail = ndtr(np.log(tree_moment(u, t["h"], t["cw"]) / Mcrit) / K["tree_res_cov"])
        dist = math.hypot(t["x"], t["y"]); reach = t["h"] if Mu_wet.mean() < Mb else 0.75 * t["h"]
        p_hit = np.zeros(N)
        if dist - radius < reach:
            to_house = math.degrees(math.atan2(-t["x"], -t["y"])) % 360
            half_w = math.atan2(radius, max(dist, 1))
            downwind = (dir_m[np.arange(N), g_loc.argmax(1)] + 180) % 360
            dang = np.radians((to_house - downwind + 180) % 360 - 180)
            kap = K["fall_kappa"]
            p_dir = np.clip(np.exp(kap * np.cos(dang)) / (2 * math.pi * np.i0(kap)) * 2 * half_w, 0, 1)
            p_hit = p_fail * p_dir
            p_tree_hit = 1 - (1 - p_tree_hit) * (1 - p_hit)
        p_limb = frag(gpk, K["frag_limb"], 1 - K["wet_soil_tree_reduction"] * wet)
        if dist < 30 + radius:
            p_limb_any = 1 - (1 - p_limb_any) * (1 - 0.5 * p_limb)
        # critical gust (median resistance, current soil) for display
        ug = np.linspace(5, 80, 300); Mg = tree_moment(ug, t["h"], t["cw"])
        ucrit = float(np.interp(float(np.median(Mcrit)), Mg, ug)) / 0.44704
        tree_out.append(dict(p_fail=round(float(p_fail.mean()), 4), p_hit=round(float(p_hit.mean()), 4),
                             p_limb=round(float(p_limb.mean()), 3), dbh_cm=round(dbh * 100), crit_mph=round(ucrit),
                             mode="uproot" if Mu_wet.mean() < Mb else "snap", gust50=round(float(np.median(gpk)), 1),
                             gust95=round(float(np.percentile(gpk, 95)), 1)))
    age = site["roof_age"]; roof_scale = 1 - 0.01 * max(0, age - 10) if age is not None else 1.0
    # building damage: gust at mean roof height (CFD approach flow where the direction's run passed its checks, else open
    # terrain at that height with the hill factor) over the code design gust at the same height
    hm_ft = 3.281 * (house.get("eave_m", 6.6) + house.get("ridge_m", 9.0)) / 2
    kz = 2.01 * (max(hm_ft, 15) / 900) ** (2 / 9.5)                       # ASCE 7-22 Table 26.10-1, exposure C
    appr = per(loc.get("appr"))
    to_h = np.where(np.isfinite(appr), appr, math.sqrt(kz) * kzt_t) if appr is not None else math.sqrt(kz) * kzt_t
    g_h = gust * to_h
    rho_pk = ((g_h / (K["design_mph"] * math.sqrt(kz))) ** 2).max(1)
    cls = house.get("cls", "WSF"); age_k = roof_scale ** 2 if cls in ("WSF", "WMF") else 1.0   # shingle age weakens the roof cover
    dmg = [ndtr(np.log(rho_pk / (m * age_k)) / (2 * b)) for m, b in K["dmg_class"][cls]]
    g50 = K["outage_g50"] - 8 * wet - 6 * site["canopy_frac"]
    p_out = 1 / (1 + np.exp(-(peak - g50) / K["outage_scale"]))
    CN = {"A": 61, "B": 75, "C": 83, "D": 87}[site["soil_group"]]; S = 1000 / CN - 10
    P = rain.sum(1); Q = np.where(P > 0.2 * S, (P - 0.2 * S) ** 2 / (P + 0.8 * S), 0)
    hr_max = rain.max(1)
    r3 = np.array([np.convolve(x, np.ones(3), "valid").max() if len(x) >= 3 else x.sum() for x in rain])
    p_flash = (hr_max > 1.0) | (r3 > 2.0)
    rv = site.get("river")
    p_creek = None
    if rv and rv.get("minor") is not None and rv["minor"] > -900 and rv.get("stage") is not None:   # also guards older caches
        peak_stage = (rv["fcst_max"] if rv["fcst_max"] is not None else rv["stage"])
        sd = 0.5 + 0.2 * max(0, peak_stage - rv["stage"])
        p_creek = float(1 - ndtr((rv["minor"] - peak_stage) / sd))
    out.update(
        hours=[dict(t=t.isoformat(), gust=gq[:, i].tolist(), sust=round((w["sust"].get(t) or 0), 1),
                    dir=dirs_from[i], rain=[round(float(np.median(rain[:, i])), 3), round(float(np.percentile(rain[:, i], 90)), 3)],
                    pop=w["pop"].get(t), thunder=w["thunder"].get(t),
                    p40=round(float((gust[:, i] > 40).mean()), 3), p58=round(float((gust[:, i] > 58).mean()), 3),
                    models={k: (round(v[i], 1) if v and v[i] is not None else None) for k, v in w.get("models", {}).items()})
               for i, t in enumerate(times)],
        peak=dict(p5=float(np.percentile(peak, 5)), p50=float(np.median(peak)), p95=float(np.percentile(peak, 95)),
                  hour=times[int(np.median(hour_of_peak))].isoformat()),
        rain_total=dict(p10=float(np.percentile(P, 10)), p50=float(np.median(P)), p90=float(np.percentile(P, 90))),
        wet=float(wet.mean()),
        probs=dict(
            power=float(p_out.mean()), limb=float(p_limb_any.mean()), tree_hit=float(p_tree_hit.mean()),
            cosmetic=float(dmg[0].mean()), minor=float(dmg[1].mean()), moderate=float(dmg[2].mean()), severe=float(dmg[3].mean()),
            ponding=float((Q > 0.5).mean()), flash=float(p_flash.mean()), creek=p_creek),
        hist=np.histogram(peak, bins=np.arange(0, max(80, peak.max() + 5), 2.5))[0].tolist(),
        trees=tree_out,
        house_gust=dict(p50=float(np.median(g_h.max(1))), p95=float(np.percentile(g_h.max(1), 95)), rho50=float(np.median(rho_pk)),
                        design=K["design_mph"], kz=round(kz, 3), hm_ft=round(hm_ft, 1)),
    )
    if loc.get("house") is not None:
        hl = gust * per(loc["house"]); yl = gust * per(loc["yard"])
        ap = per(loc.get("appr"))
        for i, hr in enumerate(out["hours"]):
            hr["roof"] = np.percentile(hl[:, i], [5, 50, 95]).round(1).tolist()
            hr["yard"] = np.percentile(yl[:, i], [5, 50, 95]).round(1).tolist()
            ok_ = np.isfinite(ap[:, i]) if ap is not None else None
            if ok_ is not None and ok_.mean() > .5:   # approach-flow gust at mean roof height (loads), where checks passed
                hr["appr"] = np.percentile(gust[ok_, i] * ap[ok_, i], [5, 50, 95]).round(1).tolist()
    # how much of the damaging wind blew within half a sector of a simulated direction (the rest borrows the nearest)
    wt = gust ** 2 * (gq[2][None, :] >= .5 * gq[2].max())
    near = np.abs((dir_m - secs[sidx] + 180) % 360 - 180) <= 11.25
    out["dir_cover"] = round(float((wt * near).sum() / max(wt.sum(), 1e-9)), 3)
    out["_dir"], out["_gust"] = dir_m, gust                                   # for choosing CFD directions; dropped later
    # tornadoes: SPC outlook -> path scenarios -> house damage, combined with the straight-line damage
    if site.get("spc"):
        tr = tornado.tornado_risk(site["spc"], K["dmg_class"][cls], K["design_mph"] * math.sqrt(kz), rng, age_k)
        out["tornado"] = tr
        for k_ in ("cosmetic", "minor", "moderate", "severe"):
            out["probs"]["wind_" + k_] = out["probs"][k_]
            out["probs"][k_] = 1 - (1 - out["probs"][k_]) * (1 - tr["probs"][k_])
    return out

# ---------------------------------------------------------------- orchestration
def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def raster_rects(rects, R, pad=0.0, values=None):
    """Rasterise oriented rectangles onto the 1 m site grid (row 0 = north). Returns max value per pixel."""
    n = 2 * R; out = np.zeros((n, n), np.float32)
    ys, xs = np.mgrid[0:n, 0:n]; X = xs - R + 0.5; Y = R - ys - 0.5
    for r, v in zip(rects, values or [1.0] * len(rects)):
        a = math.radians(r["heading"]); ux, uy = math.sin(a), math.cos(a); dx, dy = X - r["cx"], Y - r["cy"]
        m = (np.abs(dx * ux + dy * uy) <= r["L"] / 2 + pad) & (np.abs(dx * uy - dy * ux) <= r["W"] / 2 + pad)
        out[m] = np.maximum(out[m], v)
    return out

def build(args):
    lat, lon, addr = geocode(args.address)
    outline = json.loads(args.outline) if args.outline else None   # [[lat, lon], ...] roof traced by the user
    if outline:
        lat, lon = float(np.mean([p[0] for p in outline])), float(np.mean([p[1] for p in outline]))
    elif args.house_at:  # street-interpolated geocodes can land on the wrong street; trust the user's point
        lat, lon = map(float, args.house_at.split(","))
    print(f"• {addr}  ({lat:.5f}, {lon:.5f})", file=sys.stderr)
    w = nws(lat, lon)
    gw = soft(google_weather, lat, lon); gnow = soft(google_now, lat, lon)
    tz = ZoneInfo(w["tz"])
    storm = None if args.no_storm else soft(pick_storm, lat, lon, args.storm)
    track = soft(storm_track, storm) if storm else None
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    hrs = args.hours
    if track and any(p[1] is None for p in track[1]):
        end = next(p[0] for p in track[1] if p[1] is None)
        hrs = int(min(hrs, max(18, (end - now).total_seconds() / 3600 + 6)))
    times = [now + dt.timedelta(hours=i) for i in range(hrs)]
    w["spread"] = blend(w, gw, times)
    if gw:
        print(f"• blended NWS + Google Weather forecasts (model disagreement {w['spread'] * 200:.0f}%)", file=sys.stderr)
    cal = soft(calib.calibrate, lat, lon, times)
    if cal:   # 7-model EMOS trained on the last 40 days at the nearest airports replaces the equal-weight median
        for t, m in zip(times, cal["median"]):
            if m:
                w["gust"][t] = m
        w["emos"] = cal; w["models"]["calibrated"] = cal["median"]
        wt = sorted(cal["weights"]["0-24 h"].items(), key=lambda kv: -kv[1] if kv[0] in calib.NAMES.values() else 0)[:3]
        print(f"• calibrated 7-model blend (EMOS, {cal['train_days']} days at {', '.join(cal['stations'])}); top weights "
              + ", ".join(f"{k} {v:.2f}" for k, v in wt), file=sys.stderr)
    archive_forecast(lat, lon, times, w, gw)
    w["nowcast"] = nowcast(w, gnow, times)
    if gnow:
        print(f"• nowcast: gust now {gnow['gust']:.0f} mph, forecast scaled x{w['nowcast']:.2f} near-term", file=sys.stderr)
    past_rain = sum(v or 0 for t, v in w["qpf"].items() if now - dt.timedelta(hours=48) <= t < now)
    if gnow:   # rain that actually fell in the last 24 h (Google history) + NWS grid for 24-48 h ago
        past_rain = gnow["past24_in"] + sum(v or 0 for t, v in w["qpf"].items() if now - dt.timedelta(hours=48) <= t < now - dt.timedelta(hours=24))

    print("• site data (flood zone, elevation, soil, river)…", file=sys.stderr)
    with cf.ThreadPoolExecutor(6) as ex:
        fz, el, so, rv = (ex.submit(soft, f, lat, lon) for f in (flood_zone, elevation_ft, soil, river))
        fz, el, so, rv = fz.result(), el.result(), so.result(), rv.result()

    # ---- 1. find the building at the address point
    print("• locating the building (footprint datasets + imagery)…", file=sys.stderr)
    RMAX = 320                                    # site half-width ceiling: footprints are fetched this far (+100 m) once
    fps = []
    for f in (fp_google, fp_osm, fp_microsoft, fp_usa_structures):
        fps.extend(soft(f, lat, lon, RMAX + 100) or [])
    to_m = enu(lat, lon)
    z = 19 if abs(lat) < 50 else 18
    mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z
    blds = merge_footprints(fps, to_m)
    traced = lambda r, ox, oy: dict(cx=r["cx"] + ox, cy=r["cy"] + oy, L=r["L"], W=r["W"], heading=r["heading"], area=r["area"], height=None,
                                    src="traced from satellite imagery", tags={}, rank=4,
                                    ring=np.asarray(rect_ring(r["cx"] + ox, r["cy"] + oy, r["L"], r["W"], r["heading"]), float))
    near, _ = mosaic(ESRI_SAT, lat, lon, z, int(90 / mpp))
    nimg = near.resize((near.width // 2, near.height // 2)); c0 = np.array(nimg.size) / 2
    for r in detect_roofs(nimg, mpp * 2):         # imagery only fills gaps the datasets leave (new builds, rural areas)
        d = traced(dict(r, cx=(r["px"][0] - c0[0]) * mpp * 2, cy=-(r["px"][1] - c0[1]) * mpp * 2), 0, 0)
        if 20 < d["area"] < 4000 and not any(in_poly(d["cx"], d["cy"], q["ring"]) or in_poly(q["cx"], q["cy"], d["ring"]) for q in blds):
            blds.append(d)
    target = pick_target(blds)
    if outline:
        xs, ys = to_m(np.array([p[0] for p in outline]), np.array([p[1] for p in outline]))
        ring = np.c_[xs, ys]; cx, cy, L, W, hd, A = min_rect(ring)
        target = dict(cx=cx, cy=cy, L=L, W=W, heading=hd, area=A, height=None, src="outlined by you", tags={}, rank=-1, ring=ring)
        blds = [target] + [q for q in blds if not in_poly(q["cx"], q["cy"], ring) and not in_poly(cx, cy, q["ring"])]
    elif target is None:
        target = dict(cx=0.0, cy=0.0, L=18.0, W=12.0, heading=0.0, area=216.0, height=None, src="assumed (nothing detected)",
                      tags={}, rank=9, ring=np.asarray(rect_ring(0, 0, 18, 12, 0), float))
        blds.append(target)
    if args.footprint:  # "LxW@heading" in feet
        m = re.fullmatch(r"([\d.]+)x([\d.]+)(?:@([\d.]+))?", args.footprint)
        L_, W_, hd_ = float(m[1]) * .3048, float(m[2]) * .3048, float(m[3] or target["heading"])
        target.update(L=L_, W=W_, heading=hd_, area=L_ * W_, src="user supplied",
                      ring=np.asarray(rect_ring(target["cx"], target["cy"], L_, W_, hd_), float))
    span = float(np.ptp(target["ring"], 0).max())
    # the site grows with the building: ~150 m of neighbourhood on every side, up to the footprint fetch radius
    R = int(min(RMAX, max(K["maps_half_m"], math.ceil(span / 2 + 150)))); half = int(min(R - 20, max(K["scene_half_m"], span / 2 + 100)))
    if target["area"] > 2500:
        print(f"• the address is a large building ({target['area']:,.0f} m², {span:.0f} m across)", file=sys.stderr)
    hx, hy = target["cx"], target["cy"]
    clat, clon = lat + hy / 110540, lon + hx / (111320 * math.cos(math.radians(lat)))
    for q in blds:                                 # building-centred frame
        q.update(cx=q["cx"] - hx, cy=q["cy"] - hy, ring=q["ring"] - [hx, hy])
    others = [q for q in blds if q is not target and (np.abs(q["ring"]).min(0) < R).all()]

    # ---- 2. everything else in a building-centred frame
    print("• satellite imagery, buildings, canopy…", file=sys.stderr)
    sat, _ = mosaic(ESRI_SAT, clat, clon, z, int(R / mpp))
    simg = sat.resize((sat.width // 2, sat.height // 2)); c0 = np.array(simg.size) / 2
    for r in detect_roofs(simg, mpp * 2):
        d = traced(dict(r, cx=(r["px"][0] - c0[0]) * mpp * 2, cy=-(r["px"][1] - c0[1]) * mpp * 2), 0, 0)
        if 20 < d["area"] < 4000 and abs(d["cx"]) < R - 5 and abs(d["cy"]) < R - 5 and \
                not any(in_poly(d["cx"], d["cy"], q["ring"]) or in_poly(q["cx"], q["cy"], d["ring"]) for q in [target, *others]):
            others.append(d)
    bmask = np.zeros((2 * R, 2 * R), bool)
    for q in [target, *others]:
        cells = _cells(q, R)
        if cells is not None:
            bmask[cells] = True
    bmask = nd.binary_dilation(bmask)
    sat1 = sat.resize((4 * R, 4 * R), Image.BILINEAR)                     # 0.5 m
    tm = detect_trees(sat1)
    now_mask = (tm.reshape(2 * R, 2, 2 * R, 2).mean((1, 3)) > 0.5) & ~bmask
    rgb1 = np.asarray(sat.resize((2 * R, 2 * R), Image.BILINEAR))

    print("• terrain (USGS 3DEP) + LiDAR roofs and canopy (USGS 3DEP point clouds)…", file=sys.stderr)
    E_abs = soft(dem_3dep, clat, clon, R, 2 * R)
    dem_src = "USGS 3DEP 1 m"
    if E_abs is None:
        E_abs = dem_grid(clat, clon, R, 2 * R, 15); dem_src = "AWS Terrain Tiles (~5 m)"
    E = E_abs - E_abs[R, R]
    lid = soft(lidar_points, clat, clon, R)
    lidar_meta = ndsm = None
    if lid is not None:
        P, survey, yr = lid
        chm, tl = canopy_from_lidar(P, E_abs, R, now_mask)
        young = now_mask & (chm < 2)
        Hc = np.where(young, 6.0, chm)
        lidar_meta = dict(survey=survey, year=yr, points=int(len(P)))
        ndsm = measure_buildings([target, *others], P, E_abs, R)
    else:
        Hc = now_mask * args.tree_height; tl = []
        dens = nd.uniform_filter(now_mask.astype(float), 6)
        for yy in range(3, 2 * R, 6):
            for xx in range(3, 2 * R, 6):
                if dens[yy, xx] > 0.6:
                    tl.append(dict(x=xx - R + .5, y=R - yy - .5, h=args.tree_height, cw=6.0))
    # unmeasured buildings: the dataset height (OSM height or levels, Microsoft's ML estimate), else two storeys
    for q in [target, *others]:
        if q.get("h_src") != "LiDAR":
            lv = q["tags"].get("building:levels")
            h = q["height"] or (float(lv) * K["story_h"] + 1.5 if lv and lv.replace(".", "", 1).isdigit() else None)
            q.update(h=h or K["story_h"] * 2 + 1.2, h_src="dataset" if h else "assumed (two storeys)")
            q["eave"] = q["h"] - (0 if q.get("flat") else min(q["h"] * .3, K["roof_pitch"] * min(q["W"], 14) / 2))
        q.update(classify(q))
    if target.get("h_src") != "LiDAR" and args.stories:
        target.update(h=args.stories * K["story_h"] + 0.6 + K["roof_pitch"] * min(target["W"], 14) / 2, eave=args.stories * K["story_h"] + 0.6)
    Hb = building_field([target, *others], R, ndsm)
    stories = args.stories or target["stories"] or K["stories"]
    eave, ridge = max(2.0, target["eave"]), max(target["h"], target["eave"] + .5)
    ring = target["ring"].round(2).tolist()
    house = dict(cx=0.0, cy=0.0, L=target["L"], W=target["W"], heading=target["heading"], area=round(target["area"], 1),
                 height=target["height"], src=target["src"], stories=stories, eave_m=round(eave, 2), ridge_m=round(ridge, 2), ring=ring,
                 roof_src=f"USGS LiDAR {lidar_meta['year']}" if target.get("h_src") == "LiDAR" else target["h_src"],
                 cls=target["cls"], cls_label=target["label"], flat=target["flat"], osm_type=target["osm_type"],
                 name=target["tags"].get("name"))
    print(f"• {house['cls_label'].lower()}, {house['area']:,.0f} m², {stories} storey{'s' * (stories != 1)}, "
          f"roof {eave:.1f}-{ridge:.1f} m ({house['roof_src']}, {'flat' if house['flat'] else 'pitched'}); "
          f"{len(others)} neighbouring buildings", file=sys.stderr)
    Hbase = 0.3 * Hc
    trees = [t for t in tl if abs(t["x"]) < half and abs(t["y"]) < half]
    for t in trees:
        rr, cc = int(R - t["y"]), int(t["x"] + R)
        t["color"] = int(rgb1[rr, cc, 0]) << 16 | int(rgb1[rr, cc, 1]) << 8 | int(rgb1[rr, cc, 2])
    canopy = float(now_mask[R - 100:R + 100, R - 100:R + 100].mean())

    dem_wide = soft(dem_3dep, clat, clon, 1600, 161) if dem_src.startswith("USGS") else None
    if dem_wide is None:
        dem_wide = soft(dem_grid, clat, clon, 1600, 81, 13)
    kzt = kzt_by_direction(dem_wide, 1600) if dem_wide is not None else [dict(kzt=1.0, H=0, Lh=None)] * 8

    # ---- 3. neighbourhood CFD for the wind directions the ensemble actually blows from
    site = dict(past_rain_in=past_rain, roof_age=args.roof_age, canopy_frac=canopy,
                soil_group=(so or {}).get("group", "C"), river=rv, kzt=kzt)
    g_nws = np.array([w["gust"].get(t) or 0 for t in times]); d_nws = [w["dir"].get(t) for t in times]
    pre = simulate(clat, clon, times, w, track, site, house, [], np.random.default_rng(args.seed))
    sectors = pick_sectors(pre["_dir"], pre["_gust"], args.cfd_dirs)
    ang = lambda a, b: abs((a - b + 180) % 360 - 180)
    hour_sector = [min(range(len(sectors)), key=lambda i: ang(sectors[i], d or sectors[0])) for d in d_nws]
    G_open = np.clip(g_nws / np.maximum([w["sust"].get(t) or 1 for t in times], 1), 1.25, 2.0)
    leaf = soft(leaf_state, clat, clon)
    K["leaf"] = leaf["factor"] if leaf else 1.0
    lai = round(K["tree_LAI"] * (K["wood_frac"] + (1 - K["wood_frac"]) * K["leaf"]), 1)
    if leaf:
        print(f"• leaf state (NASA MODIS {leaf['date']}): LAI {leaf['lai_now']} vs summer {leaf['lai_peak']} -> canopy drag x{K['leaf']:.2f}", file=sys.stderr)
    M = dict(E=E.astype(np.float32), Hb=Hb, Hc=Hc.astype(np.float32), Hbase=Hbase.astype(np.float32), R=R, lai=lai)
    key = hashlib.sha1(f"{clat:.5f},{clon:.5f},{R},{lidar_meta},{lai},cfd4,{house['ring']},{float(Hb.sum()):.0f}".encode()).hexdigest()[:12]
    # the coarse field must hold the nested box and the tallest building near it
    out_half = int(min(R / math.sqrt(2) - 4, max(140, nested_box(house, 1e9)[1] + 10)))
    out_nz = int(max(14, math.ceil((max(ridge, float(np.percentile(Hb[Hb > 0], 99)) if (Hb > 0).any() else 0) + 25) / 2.5)))
    if not args.no_cfd:
        print(f"• wind simulation (lattice-Boltzmann CFD, {len(sectors)} wind directions)…", file=sys.stderr)
    fields = {} if args.no_cfd else (soft(wind_fields, M, sectors, key, steps=args.cfd_steps, out_half=out_half, out_nz=out_nz) or {})
    if fields:
        sectors = [s_ for s_ in sectors if s_ in fields]
        hour_sector = [min(range(len(sectors)), key=lambda i: ang(sectors[i], d or sectors[0])) for d in d_nws]
    local = None
    house["walls"] = wall_list(house["ring"])
    if len(fields) == len(sectors):
        MF = dict(E=E.astype(np.float32), Hc=Hc.astype(np.float32), Hbase=Hbase.astype(np.float32), R=R, lai=lai, Hb=Hb,
                  house={k: house[k] for k in ("ring", "eave_m", "ridge_m")}, walls=house["walls"])
        hp = soft(house_pressures, MF, fields, key)
        if hp and len(hp) == len(sectors):   # per-wall Cp in sector order; the viewer picks the hour's sector
            lens = [wl["len"] for wl in house["walls"]]
            hp = {s: trim_walls(hp[s], lens) for s in hp}
            ok = {s: cp_ok(hp[s]) for s in sectors}   # a failed direction is null: the viewer uses ASCE 7 for it
            if not all(ok.values()):
                bad = ", ".join(f"{s:g}°" for s in sectors if not ok[s])
                print(f"• house pressures for wind from {bad} failed the physical check; those use ASCE 7 tables", file=sys.stderr)
            if any(ok.values()):   # all failed: leave both unset, so the report says ASCE 7 throughout
                # net pressure coefficients (external - internal) per wall and over the roof: what loads the envelope
                net = lambda v, s_: None if v is None else round(v - hp[s_]["cpi"], 3)
                for i, wl in enumerate(house["walls"]):
                    wl["cp"] = [net(hp[s]["walls"][i], s) if ok[s] else None for s in sectors]
                house["roof_cp"] = [[net(hp[s]["roof"][0], s), net(hp[s]["roof"][1], s)] if ok[s] else None for s in sectors]
                house["cpi"] = [hp[s]["cpi"] if ok[s] else None for s in sectors]
                # approach-flow gust at mean roof height per direction: the gust the CFD Cp is referenced to (p = Cp q_h)
                house["approach"] = [round(gust_ratio(hp[s]["ref"]["U"], hp[s]["ref"]["sig"], fields[s]["z0"], z_agl=hp[s]["ref"]["hm"]), 3)
                                     if ok[s] else None for s in sectors]
                house["cfd_grid"] = dict(dx=hp[sectors[0]]["dx"], cells=hp[sectors[0]]["n"])
        # probes just outside every wall (2 m) and out in the yard (7 m), not on a circle that can cut an L-shaped outline
        ring = [(wl["x"] + 2 * wl["nx"], wl["y"] + 2 * wl["ny"]) for wl in house["walls"]]
        yard = [(wl["x"] + 7 * wl["nx"], wl["y"] + 7 * wl["ny"]) for wl in house["walls"]]
        def ratios(pts, how):   # local / open-terrain gust per simulated direction
            out = []
            for s in sectors:
                F = fields[s]
                out.append(how([gust_ratio(*probe(F, x, y, zz, E, R), F["z0"], z_agl=zz) for x, y, zz in pts]))
            return np.array(out)
        local = dict(sectors=sectors, house=ratios([(x, y, eave + 1) for x, y in ring] + [(0, 0, ridge + 1.5)], max),
                     yard=ratios([(x, y, 2.0) for x, y in yard], np.mean),
                     trees=np.array([ratios([(t["x"], t["y"], t["h"] + 1)], max) for t in trees]) if trees else None)
        if house.get("approach"):
            local["appr"] = np.array([a if a is not None else np.nan for a in house["approach"]])

    spc = soft(tornado.spc_outlooks, lambda u: get(u, timeout=20), clat, clon, times[0], times[-1] + dt.timedelta(hours=1))
    if spc:
        site["spc"] = spc
        print("• SPC tornado outlook: " + "; ".join(f"day {d['day']} {d['p25']:.0%} within 25 mi" + (f", CIG{d['cig']}" if d["cig"] else "")
                                                   for d in spc), file=sys.stderr)
    print(f"• Monte Carlo ({K['members']} members)…", file=sys.stderr)
    sim = simulate(clat, clon, times, w, track, site, house, trees, np.random.default_rng(args.seed), local)
    sim.pop("_dir"); sim.pop("_gust")
    for t, r in zip(trees, sim["trees"]):
        t.update(r)

    # storm basemap (dark canvas + labels) spanning house + official track
    smap = None
    if sim.get("official"):
        la = sim["official"]["lat"] + [clat]; lo = sim["official"]["lon"] + [clon]
        for tr in sim["tracks"]:
            la += tr["lat"][::4]; lo += tr["lon"][::4]
        mz = 4
        for zz in range(9, 3, -1):
            x0, y0 = merc_px(max(la), min(lo), zz); x1, y1 = merc_px(min(la), max(lo), zz)
            if x1 - x0 < 1100 and y1 - y0 < 800:
                mz = zz; break
        mlat, mlon = (max(la) + min(la)) / 2, (max(lo) + min(lo)) / 2
        x0, y0 = merc_px(max(la), min(lo), mz); x1, y1 = merc_px(min(la), max(lo), mz)
        hw, hh = max(640, (x1 - x0) / 2 + 160), max(400, (y1 - y0) / 2 + 120)
        base, tl_ = mosaic(ESRI_DARK, mlat, mlon, mz, int(hw), int(hh))
        ref = soft(mosaic, ESRI_DARK_REF, mlat, mlon, mz, int(hw), int(hh))
        if ref:
            refa = ref[0].convert("L"); base = Image.composite(Image.new("RGB", base.size, (225, 225, 225)), base,
                                                               refa.point(lambda p: 255 if p > 70 else 0))
        smap = dict(img=data_uri(base, q=78), z=mz, x0=tl_[0], y0=tl_[1], w=base.width, h=base.height)

    # ---- 4. scene assets (all embedded: the artifact sandbox cannot fetch tiles)
    s0, s1 = R - half, R + half
    terr = E[s0:s1, s0:s1].reshape(half, 2, half, 2).mean((1, 3))           # 2 m
    scene = dict(half=half, img=data_uri(sat.crop((round(sat.width / 2 - half / mpp), round(sat.height / 2 - half / mpp),
                                                   round(sat.width / 2 + half / mpp), round(sat.height / 2 + half / mpp))).resize((2048, 2048)), q=82),
                 terrain=dict(n=half, dx=2.0, cm=b64((terr * 100).round().astype(np.int16))), dem_src=dem_src)
    if lid is not None:
        P = lid[0]
        ii = np.clip((R - P[:, 1]).astype(int), 0, 2 * R - 1); jj = np.clip((P[:, 0] + R).astype(int), 0, 2 * R - 1)
        hag = P[:, 2] - E_abs[ii, jj]
        keep = (P[:, 3] != 2) & (np.abs(P[:, 0]) < half) & (np.abs(P[:, 1]) < half) & (hag > 1.0) & now_mask[ii, jj] & (hag < Hc[ii, jj] + 3)
        idx = np.nonzero(keep)[0]
        idx = np.random.default_rng(0).choice(idx, min(len(idx), 180000), replace=False) if len(idx) else idx
        xyz = np.c_[P[idx, 0], P[idx, 1], P[idx, 2] - E_abs[R, R]] * 10
        scene["veg"] = dict(n=int(len(idx)), xyz=b64(xyz.round().astype(np.int16)), rgb=b64(rgb1[ii[idx], jj[idx]].astype(np.uint8)))
    wind = None
    if local is not None:
        wind = dict(sectors=[dict(dir=s, z0=round(fields[s]["z0"], 3), dx=fields[s]["dx"], z_base=float(fields[s]["z_base"]),
                                  nw=int(fields[s]["u"].shape[1]), nz=int(fields[s]["u"].shape[3]),
                                  exposure=round(exposure_ratio(fields[s]["z0"]), 3),
                                  u=b64(np.clip(fields[s]["u"] * 50, -127, 127).round().astype(np.int8)),
                                  sig=b64(np.clip(fields[s]["sig"] * 200, 0, 255).round().astype(np.uint8)),
                                  lattice=[int(v) for v in fields[s]["n_lattice"]]) for s in sectors],
                    hour_sector=hour_sector, G_open=G_open.round(3).tolist(), steps=args.cfd_steps,
                    house=local["house"][hour_sector].round(3).tolist(), yard=local["yard"][hour_sector].round(3).tolist(),
                    cover=sim["dir_cover"])

    house_r = math.sqrt(house["L"] * house["W"] / math.pi)
    hz = sim["probs"]; hg = sim["house_gust"]
    lvl = lambda p: "very unlikely" if p < .03 else "unlikely" if p < .15 else "possible" if p < .4 else "likely" if p < .75 else "very likely"
    zone = (fz or {}).get("FLD_ZONE", "unknown")
    big = [t for t in trees if t["h"] >= K["tree_min_h"]]
    near_trees = [t for t in big if math.hypot(t["x"], t["y"]) < 30 + house_r]
    reach_trees = [t for t in big if math.hypot(t["x"], t["y"]) - house_r < t["h"]]
    worst = max(trees, key=lambda t: t["p_hit"], default=None)
    noun = "house" if house["cls"] == "WSF" else "building"
    risks = [
        dict(key="power", title="Power outage", p=hz["power"],
             plain=f"{lvl(hz['power']).capitalize()}. Peak open-terrain gust near {sim['peak']['p50']:.0f} mph"
                   + (" with wet ground" if sim["wet"] > .3 else "") + ". Most outages come from branches on lines.",
             how="Per-customer outage probability is a logistic function of the peak gust, the form used in statistical "
                 "outage models (Guikema et al. 2014; Han et al. 2009). The 50 % point drops when soil is wet and tree cover is high.",
             calc=[["Peak gust (median)", f"{sim['peak']['p50']:.0f} mph"], ["50 % outage gust", f"{K['outage_g50'] - 8 * sim['wet'] - 6 * canopy:.0f} mph"],
                   ["Tree cover within 100 m", f"{canopy:.0%}"], ["Soil wetness index", f"{sim['wet']:.2f}"]]),
        dict(key="limb", title="Branches down in your yard" if noun == "house" else "Branches down around the building", p=hz["limb"],
             plain=f"{lvl(hz['limb']).capitalize()}. {len(near_trees)} trees within 100 ft of the walls, gusts at their crowns "
                   f"computed from the wind simulation.",
             how=f"Each tree within 30 m of the {noun} gets its own crown-top gust from the CFD field. Branch failure follows an "
                 "empirical lognormal curve (median 55 mph gust), weakened when the soil is wet.",
             calc=[["Trees within 100 ft", str(len(near_trees))], ["Median crown gust (nearest tree)",
                   f"{min(near_trees, key=lambda t: math.hypot(t['x'], t['y']))['gust50']:.0f} mph" if near_trees else "n/a"]]),
        dict(key="tree_hit", title=f"Tree falls on the {noun}", p=hz["tree_hit"],
             plain=f"{lvl(hz['tree_hit']).capitalize()}. {len(reach_trees)} trees are tall enough to reach the {noun}."
                   + (f" The riskiest is {worst['h'] * 3.28:.0f} ft tall, {max(0, math.hypot(worst['x'], worst['y']) - house_r) * 3.28:.0f} ft from the walls; it would need a "
                      f"~{worst['crit_mph']} mph gust to {'uproot' if worst['mode'] == 'uproot' else 'snap'}." if worst and worst["p_hit"] > 0 else ""),
             how="Mechanistic tree model in the style of HWIND / ForestGALES: wind moment 0.5·ρ·Cd(u)·A·u²·z vs. stem-break "
                 "(MOR·π·DBH³/32) and uprooting (Creg·stem mass) resistance, DBH from LiDAR height and crown width (Jucker et al. 2017). "
                 f"A failed tree falls downwind with von Mises scatter; it hits if the {noun} lies in that arc and within reach.",
             calc=[[f"Trees that can reach the {noun}", str(len(reach_trees))], ["Lowest critical gust nearby",
                   f"{min((t['crit_mph'] for t in reach_trees), default=0)} mph" if reach_trees else "n/a"],
                   ["Tree heights from", f"USGS LiDAR {lidar_meta['year']}" if lidar_meta else "assumed"]]),
        dict(key="cosmetic", title=DMG_TEXT[house["cls"]][0], p=hz["cosmetic"],
             plain=f"{lvl(hz['cosmetic']).capitalize()}. {DMG_TEXT[house['cls']][1]}",
             how=f"{house['cls_label']} ({house['cls']}). Wind load is judged against the design load: the 3-s gust at mean roof "
                 f"height ({hg['hm_ft']:.0f} ft; from the wind simulation's approach flow where it passed its checks) over the ASCE 7 "
                 f"design gust there ({hg['design']:.0f} mph basic speed, exposure C, Kz {hg['kz']:.2f}), squared. Lognormal damage "
                 f"states on that ratio; wood-frame homes follow Hazus-style curves, other classes are engineering estimates.",
             calc=[["Building", f"{house['cls_label']}, {house['area']:,.0f} m², {house['stories']} storeys"],
                   ["Peak gust at roof height", f"{hg['p50']:.0f} mph (95 %: {hg['p95']:.0f})"],
                   ["Load / design load (median)", f"{hg['rho50']:.2f}"],
                   ["Damage starts near (median curve)", f"{K['dmg_class'][house['cls']][0][0]:.2f} x design load"]]),
        dict(key="minor", title=DMG_TEXT[house["cls"]][2], p=hz["minor"],
             plain=f"{lvl(hz['minor']).capitalize()}. {DMG_TEXT[house['cls']][3]}",
             how=f"Same load ratio; the {house['cls']} 'minor' damage state has its median at "
                 f"{K['dmg_class'][house['cls']][1][0]:.2f} x the design load (spread β {K['dmg_class'][house['cls']][1][1]:.2f} on gust speed).",
             calc=[["Median of damage curve", f"{K['dmg_class'][house['cls']][1][0]:.2f} x design load"],
                   ["Equivalent gust at roof height", f"{K['design_mph'] * math.sqrt(hg['kz'] * K['dmg_class'][house['cls']][1][0]):.0f} mph"]]),
        dict(key="flash", title="Flash flooding nearby", p=hz["flash"],
             plain=f"{lvl(hz['flash']).capitalize()}. Storm total about {sim['rain_total']['p50']:.2f} in (90 % chance under {sim['rain_total']['p90']:.2f} in).",
             how="Share of ensemble members with more than 1 in/hr or 2 in in 3 hr (typical flash-flood guidance for the region).",
             calc=[["Rain total (median)", f"{sim['rain_total']['p50']:.2f} in"], ["Rain total (90 %)", f"{sim['rain_total']['p90']:.2f} in"]]),
        dict(key="ponding", title="Water pooling by the foundation", p=hz["ponding"],
             plain=f"{lvl(hz['ponding']).capitalize()}. {so['name'] if so else 'Local soil'} (group {site['soil_group']}) drains "
                   + {"A": "fast", "B": "moderately", "C": "slowly", "D": "very slowly"}[site["soil_group"]] + ".",
             how="SCS curve-number runoff (USDA TR-55) for a residential lot; more than 0.5 in of runoff counts as ponding.",
             calc=[["Curve number", str({"A": 61, "B": 75, "C": 83, "D": 87}[site["soil_group"]])], ["Rain in last 48 h", f"{past_rain:.2f} in"]]),
    ]
    tn = sim.get("tornado")
    odds = lambda p: "none" if p <= 0 else f"{p:.1%}" if p >= .001 else f"1 in {1 / p:,.0f}"
    spc_p = lambda p: f"{p:.0%}" if p else "under 2 %"
    if tn:
        tp = tn["probs"]; top = max(tn["days"], key=lambda d: d["p25"])
        risks.append(dict(key="tornado", title=f"Tornado winds at the {noun}", p=tp["ef0"],
                          plain=(f"{lvl(tp['ef0']).capitalize()}. SPC gives a {top['p25']:.0%} chance of a tornado within 25 miles "
                                 f"(day {top['day']}{', strong tornadoes possible' if top['cig'] else ''}). Even then, a path "
                                 f"crossing this lot is rare: about 1 in {1 / max(tp['ef0'], 1e-12):,.0f}."
                                 if top["p25"] else f"Very unlikely. The {noun} is outside SPC's tornado risk areas (under 2 % "
                                 "within 25 miles)."),
                          how="SPC's probabilistic tornado outlook gives the chance of a tornado within 25 mi. Path scenarios "
                              "(rating, length, width, forward speed) are drawn from US climatology, shifted toward strong "
                              f"tornadoes by SPC's conditional-intensity level. Each passes the {noun} as a translating modified "
                              "Rankine vortex; the share of the 25-mile circle where the lot would feel a given wind is "
                              "integrated exactly. Damage uses this building's own curves; EF speeds are damage-equivalent, so "
                              "no extra tornado load factor is applied. Included in the damage probabilities above.",
                          calc=[["Chance of a tornado within 25 mi", ", ".join(f"day {d['day']}: {spc_p(d['p25'])}" for d in tn["days"])],
                                ["Lot sees tornado-force wind (EF0+)", odds(tp["ef0"])], ["Lot sees EF2+ wind", odds(tp["ef2"])],
                                ["Minor damage from a tornado", odds(tp["minor"])]]))
    if hz["creek"] is not None:
        risks.append(dict(key="creek", title=f"{rv['name']} out of its banks", p=hz["creek"],
                          plain=f"{lvl(hz['creek']).capitalize()}. Now {rv['stage']:.1f} ft; flooding starts at {rv['minor']} ft. Gauge is {rv['dist_mi']} mi away.",
                          how="NWS gauge observation and official river forecast (NWPS), with forecast-error spread.",
                          calc=[["Stage now", f"{rv['stage']:.1f} ft"], ["Minor flood stage", f"{rv['minor']} ft"]]))
    return dict(
        meta=dict(address=addr, lat=clat, lon=clon, tz=w["tz"], office=w["office"],
                  generated=dt.datetime.now(tz).strftime("%a %b %-d, %-I:%M %p %Z"),
                  storm=storm and dict(name=storm["name"], id=storm["id"], cls=storm["classification"], kt=int(storm["intensity"]),
                                       mb=int(storm["pressure"]), adv=storm["publicAdvisory"]["advNum"],
                                       url=storm["publicAdvisory"]["url"], lat=storm["latitudeNumeric"], lon=storm["longitudeNumeric"],
                                       move=f"{storm['movementSpeed']} mph toward {storm['movementDir']}°"),
                  members=K["members"], engine=ENGINE_VERSION),
        sim=sim, risks=risks, alerts=w["alerts"],
        site=dict(elev_ft=el, flood_zone=zone, zone_sub=(fz or {}).get("ZONE_SUBTY"), sfha=(fz or {}).get("SFHA_TF"),
                  soil=so, river=rv, canopy=canopy, past_rain=past_rain, kzt=kzt, lidar=lidar_meta, dem=dem_src, leaf=leaf, lai=lai,
                  terrain_relief=float(dem_wide.max() - dem_wide.min()) if dem_wide is not None else None),
        house=house,
        buildings=[dict(cx=r["cx"], cy=r["cy"], L=r["L"], W=r["W"], heading=r["heading"], h=r["h"], src=r["src"])
                   for r in others if abs(r["cx"]) < half and abs(r["cy"]) < half],
        trees=trees, scene=scene, wind=wind, map=smap,
        knobs={k: v for k, v in K.items() if not isinstance(v, dict)},
        calib=w.get("emos") and {k: v for k, v in w["emos"].items() if k in ("weights", "stations", "train_days", "rho")} | dict(verify=soft(calib.backtest, lat, lon, 60)),
    )

def selftest():
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc); ts = [t0, t0 + dt.timedelta(hours=1)]
    wb = dict(gust={t0: 20.0, ts[1]: None}); sp = blend(wb, dict(gust={t0: 80.0, ts[1]: 30.0}), ts)
    assert abs(wb["gust"][t0] - 40) < 1e-9 and wb["gust"][ts[1]] == 30.0, "blend: geometric mean, fill gaps from Google"
    assert abs(sp - math.log(4) / 2) < 1e-9 and blend(dict(gust={t0: 20.0}), None, [t0]) == 0.0
    wn = dict(gust={t0: 20.0, ts[1]: 20.0}); r = nowcast(wn, dict(gust=30.0), ts, efold_h=1.0)
    assert r == 1.5 and abs(wn["gust"][t0] - 30) < 1e-9 and abs(wn["gust"][ts[1]] - 20 * 1.5 ** math.exp(-1)) < 1e-9, "nowcast decay"
    assert nowcast(dict(gust={t0: 10.0}), dict(gust=100.0), [t0]) == 2.0, "nowcast ratio clipped"
    assert abs(holland(np.array(30.0), 50, 30, 1.4) - 50) < 1e-9, "Holland V(rmax) must equal vmax"
    assert holland(np.array(300.0), 50, 30, 1.4) < holland(np.array(60.0), 50, 30, 1.4)
    assert abs(kaplan_demaria(100, 0) - K["kd_R"] * 100) < 1e-9, "Kaplan-DeMaria at t=0 is R*V0"
    assert kaplan_demaria(100, 48) < 40
    assert abs(frag(np.array(95.0), (95, .2)) - 0.5) < 1e-9
    txt = """2100 UTC SAT OCT 10 2026
POST-TROPICAL CYCLONE CENTER LOCATED NEAR 33.5N  86.9W AT 10/2100Z
MAX SUSTAINED WINDS  25 KT WITH GUSTS TO  35 KT.
FORECAST VALID 11/0600Z 35.2N  86.0W...POST-TROP/INLAND
MAX WIND  25 KT...GUSTS  35 KT.
FORECAST VALID 12/1800Z...DISSIPATED"""
    issue, pts = parse_tcm(txt)
    assert pts[0][1:] == (33.5, -86.9, 25) and pts[1][1:] == (35.2, -86.0, 25) and pts[2][3] == 0
    assert pts[2][0] == dt.datetime(2026, 10, 12, 18, tzinfo=dt.timezone.utc)
    cx, cy, L, W, hd, A = min_rect(rect_ring(5, -3, 20, 10, 30))
    assert abs(L - 20) < .3 and abs(W - 10) < .3 and abs(hd - 30) < 1.5 and abs(cx - 5) < .2 and abs(cy + 3) < .2, (cx, cy, L, W, hd)
    flat = np.zeros((41, 41)); ridge = flat.copy(); ridge[:, :] = -np.abs(np.arange(41) - 20)[None, :] * 10.0
    assert kzt_by_direction(flat, 800)[0]["kzt"] == 1.0
    assert kzt_by_direction(ridge, 800)[2]["kzt"] > 1.0, "house on a ridge should speed up east wind"
    # 25 kt storm cannot give hurricane-force gusts at the site
    times = [dt.datetime(2026, 10, 11, h, tzinfo=dt.timezone.utc) for h in range(12)]
    w = dict(gust={t: 30.0 for t in times}, sust={t: 15.0 for t in times}, qpf={t: .02 for t in times},
             dir={t: 200 for t in times}, pop={}, thunder={})
    site = dict(past_rain_in=0.5, roof_age=10, canopy_frac=.2, soil_group="C", river=None)
    K["members"] = 400
    s = simulate(35.05, -90.03, times, w, (issue, pts), site, dict(cx=0, cy=0, L=18, W=12),
                 [dict(x=0, y=20, h=15, cw=10)], np.random.default_rng(1))
    assert s["probs"]["minor"] < 0.01 and 20 < s["peak"]["p50"] < 45, s["peak"]
    # per-member wind direction: a hurricane passing 25 km east, moving north, blows from the north on the house; members
    # whose track crosses to the west of the house swing to southerly winds, which a single forecast direction would miss
    tt = [dt.datetime(2026, 10, 11, h, tzinfo=dt.timezone.utc) for h in range(13)]
    w2 = dict(gust={t: 60.0 for t in tt}, sust={t: 40.0 for t in tt}, qpf={t: .1 for t in tt}, dir={t: 0 for t in tt}, pop={}, thunder={})
    pts2 = [(tt[0], 35.05 - 1.5, -90.03 + .28, 90), (tt[-1] + dt.timedelta(hours=12), 35.05 + 1.5, -90.03 + .28, 90)]
    s2 = simulate(35.05, -90.03, tt, w2, (tt[0] - dt.timedelta(hours=24), pts2), site, dict(cx=0, cy=0, L=18, W=12), [], np.random.default_rng(2))
    d6 = s2["_dir"][:, 6]
    south = ((d6 > 90) & (d6 < 270)).mean()
    assert 0.03 < south < 0.6, south
    secs = pick_sectors(s2["_dir"], s2["_gust"], 6)
    assert 2 <= len(secs) <= 6 and 0.0 in secs, secs
    assert pick_sectors(np.full((50, 4), 200.0), np.full((50, 4), 30.0), 6) == [202.5]
    import cfd
    good = np.zeros((3, 8, 8, 20), np.float32); good[0] = cfd.log_profile((np.arange(20) + .5) * 2.5, .5)
    assert field_ok(dict(u=good, dx=2.5, z0=.5)) and not field_ok(dict(u=good * .15, dx=2.5, z0=.5)), "stalled CFD field must fail"
    tornado.selftest()
    # tree model: calibrated reference tree fails at its reference gust; a slender tree of the same height is weaker
    calibrate_trees(); h, cw, mph = K["tree_ref"]
    _, Mb, Mu = tree_mechanics(h, cw)
    assert abs(tree_moment(mph * 0.44704, h, cw) / min(Mb, Mu) - 1) < 1e-6
    crit = lambda h, cw: np.interp(min(tree_mechanics(h, cw)[1:]), tree_moment(np.linspace(5, 90, 400), h, cw), np.linspace(5, 90, 400))
    assert crit(18, 6) < crit(18, 12), "slender tree should fail at a lower gust"
    # exposure: open terrain maps to itself; rougher terrain slows the mean wind
    assert abs(exposure_ratio(0.03) - 1) < 1e-9 and exposure_ratio(0.5) < 0.9
    assert abs(gust_ratio(1.0, 1 / math.log(10 / 0.03), 0.03) - 1) < 1e-9, "open terrain maps to itself"
    assert gust_ratio(0.3, 0.1, 0.5) < 0.6, "sheltered wake gusts well below open terrain"
    # outline + LiDAR roof: L-shaped ring excludes its notch; eave/ridge come from building returns inside it
    Lr = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
    assert in_poly(np.array([5., 15.]), np.array([5., 15.]), Lr).tolist() == [True, False]
    g = np.zeros((60, 60)); rng = np.random.default_rng(0); px, py = rng.uniform(0, 20, (2, 4000))
    pz = np.where(px < 2, 5.5, 5.5 + 3 * (1 - abs(py % 10 - 5) / 5)); P = np.c_[px, py, pz, np.full(4000, 6)]
    e, r = roof_heights(P, g, 30, Lr)
    assert 5 < e < 6.5 and 8 < r < 9, (e, r)
    # CFD: a tiny run stays finite and leaves a strong velocity deficit in the wake of a block
    import cfd
    sol = np.zeros((40, 12, 10), bool); sol[:, :, 0] = True; sol[12:16, 4:8, 1:5] = True
    mu, sg = cfd.solve(sol, np.zeros(sol.shape, np.float32), 2.0, 0.1, steps=500, avg_from=250)
    assert np.isfinite(mu).all() and mu[0, 18:24, 5:7, 1:3].mean() < 0.4 * mu[0, 18:24, 0:2, 1:3].mean(), "wake behind block, speed-up beside it"
    # hip roof on the L outline: eave at the edge, ridge cap inside, zero outside; walls face outward
    hs = dict(ring=Lr, eave_m=6.0, ridge_m=9.0)
    rf = roof_field(np.array([0.05, 5.0, 15.0, 15.0]), np.array([5.0, 5.0, 5.0, 15.0]), hs, [])
    assert abs(rf[0] - 6.025) < 1e-3 and rf[1] == 8.5 and rf[2] == 8.5 and rf[3] == 0, rf
    wl = wall_list(Lr + [Lr[0]])
    assert len(wl) == 6 and {(w["fb"]) for w in wl} == {0.0, 90.0, 180.0, 270.0}, wl
    assert next(w for w in wl if w["y"] == 0)["fb"] == 180.0, "south wall faces south"
    assert cp_ok(dict(walls=[-.7, -.9, .87, .53, .41, -.1, -.54, -.51], cpi=-.105, roof=[-.789, -1.999]))   # a sane result
    assert cp_ok(dict(walls=[.6, .4, .5, .55], cpi=.51, roof=[.2, -.3]))     # sheltered: every face positive, net loads small
    assert not cp_ok(dict(walls=[3.3, -1.0, 1.9, 2.2], cpi=1.6, roof=[.5, -.2]))   # net wall pressure beyond any tunnel data
    assert not cp_ok(dict(walls=[.8, -.5], cpi=.15, roof=[float("nan"), -1])) and not cp_ok(dict(walls=[], cpi=0, roof=[-.5, -1]))
    assert not cp_ok(dict(walls=[.8, -.5], roof=[-.5, -1]))                  # no internal pressure: cannot judge
    assert cp_ok(dict(walls=[.8, None, -.5], cpi=.15, roof=[-.6, -1.5]))    # one unsampled wall: keep the direction
    t = trim_walls(dict(walls=[.8, -3.9, -.5, -.6], cpi=.0, roof=[-.6, -1.5]), [20, 2, 20, 20])
    assert t["walls"][1] is None and cp_ok(t), "a short separated corner wall is set aside"
    assert trim_walls(dict(walls=[.8, -3.9, -.5], cpi=.0, roof=[-.6, -1.5]), [20, 20, 20])["walls"][1] == -3.9, "a long bad wall still fails"
    calib.selftest()
    print("selftest ok")

def finite(o):
    """NaN / inf -> None: browsers reject them in JSON, which would blank the whole report."""
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return finite(o.tolist())
    return o

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("address", nargs="?", help='US street address or "lat,lon"')
    ap.add_argument("-o", "--out", help="output HTML (default: report-<slug>.html)")
    ap.add_argument("--storm", help="NHC storm id or name (default: nearest active storm)")
    ap.add_argument("--no-storm", action="store_true", help="weather-only mode (no tropical cyclone)")
    ap.add_argument("--hours", type=int, default=36)
    ap.add_argument("--house-at", help='"lat,lon" of the roof if the address pin lands on the street')
    ap.add_argument("--outline", help='roof outline as JSON [[lat, lon], ...] (overrides footprint detection)')
    ap.add_argument("--design-mph", type=float, help="ASCE 7 basic design wind speed for the site (3-s gust, mph; default 110)")
    ap.add_argument("--footprint", help='override house size "LxW@heading" in feet, e.g. 60x42@90')
    ap.add_argument("--stories", type=int)
    ap.add_argument("--roof-age", type=float, help="years since the roof was replaced")
    ap.add_argument("--tree-height", type=float, default=K["tree_height_m"], help="typical tree height, metres")
    ap.add_argument("--members", type=int, default=K["members"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfd-dirs", type=int, default=6, help="most wind directions to simulate with CFD; the ensemble picks "
                    "the ones holding 90 %% of its damaging wind (cached per site)")
    ap.add_argument("--cfd-steps", type=int, default=1900)
    ap.add_argument("--no-cfd", action="store_true")
    ap.add_argument("--json", action="store_true", help="also write the raw data bundle")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate-cfd", action="store_true", help="run the cube-in-boundary-layer CFD benchmark (~8 min) and exit")
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="verify the gust forecast at the address over the last DAYS days and exit")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.validate_cfd:
        import cfd
        for h in (10, 16, 20):   # grid convergence on the solver the engine actually uses
            print(cfd.benchmark_cube(h, solver=cfd.best_solver()), flush=True)
        return print("wind tunnels: X_R/H 1.4-1.6 (LES 1.5-1.8), windward Cp(2/3 H) +0.7-0.8")
    if a.backtest:
        lat, lon, _ = geocode(a.address)
        return print(json.dumps(calib.backtest(lat, lon, a.backtest), indent=1))
    if not a.address:
        ap.error("address required")
    K["members"] = a.members
    if a.design_mph:
        K["design_mph"] = a.design_mph
    data = finite(build(a))
    slug = re.sub(r"[^a-z0-9]+", "-", data["meta"]["address"].lower()).strip("-")[:60]
    out = Path(a.out or f"report-{slug}.html")
    m = data["meta"]; street = m["address"].split(",")[0].title()
    title = f"{m['storm']['name']} at {street}" if m["storm"] else f"StormVoyance · {street}"
    html = (HERE / "template.html").read_text().replace("__TITLE__", title, 1) \
        .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=float))
    out.write_text(html)
    if a.json:
        out.with_suffix(".json").write_text(json.dumps({k: v for k, v in data.items() if k not in ("scene", "map")}, indent=1, default=float))
    print(f"✓ wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

if __name__ == "__main__":
    main()
