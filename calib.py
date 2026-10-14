"""calib: verify and calibrate the gust forecast against what airport anemometers actually measured.

Data (all free, no key):
  * archived forecasts  Open-Meteo Previous Runs API: HRRR, NBM, GFS, ECMWF IFS, ICON, GEM, UKMO hourly 10 m gust,
                        as issued 0-24 h ("day0") and 24-48 h ("day1") before the valid time
  * observations        Iowa Environmental Mesonet ASOS 1-minute archive (true hourly peak 3-s gust), the NWS stations
                        nearest the address
                        (open-terrain 10 m anemometers, i.e. the same "open-terrain gust" the CFD rescales)
Method: EMOS / non-homogeneous regression (Gneiting et al. 2005; Thorarinsdottir & Gneiting 2010 for wind), the
standard statistical post-processing at NWS (NBM) and ECMWF. On log gust:
      ln y ~ N(mu, sigma^2),  mu = a + sum_m b_m ln x_m,  sigma^2 = c^2 + d^2 s^2   (s^2 = model spread)
fitted by minimum CRPS over a rolling 40-day window pooled across nearby stations. Weights b_m therefore track which
models have been right here lately, a corrects bias, and sigma is the error the models actually made (wider when they
disagree). Backtest = rolling-origin: each test day is predicted by a fit on the 40 days before it only.
"""
import datetime as dt, io, math
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

MODELS = ["ncep_hrrr_conus", "ncep_nbm_conus", "gfs_global", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ukmo_seamless"]
NAMES = dict(zip(MODELS, ["HRRR", "NBM", "GFS", "ECMWF", "ICON", "GEM", "UKMO"]))
CAPE_FROM = ["gfs_global", "ncep_hrrr_conus"]   # convective energy: thunderstorm outflow gusts the grids smooth out
LEADS = ("", "_previous_day1")       # issued 0-24 h / 24-48 h before valid time
TRAIN_DAYS, MIN_MPH = 40, 10.0       # rolling window; drop light-wind hours (storm impacts live well above this)
SQPI = 1 / math.sqrt(math.pi)


def _get():
    import stormvoyance   # shares the HTTP helper / cache; imported lazily (stormvoyance imports this module)
    return stormvoyance.get


def stations(lat, lon, n=4):
    get = _get()
    p = get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")["properties"]
    fs = get(p["observationStations"] + "?limit=12")["features"]
    out = []
    for f in fs:
        sid = f["properties"]["stationIdentifier"]
        if len(sid) == 4 and sid[0] in "KP":            # ASOS/AWOS (IEM drops the K)
            out.append(dict(id=sid, iem=sid[1:] if sid[0] == "K" else sid, name=f["properties"]["name"],
                            lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1]))
        if len(out) == n:
            break
    return out


def archived(lat, lon, d0, d1):
    """Forecasts valid d0..d1 (dates) for each lead: {lead: {model: array}} and the hourly UTC times."""
    q = ",".join(f"{v}{l}" for v in ("wind_gusts_10m", "cape") for l in LEADS)
    d = _get()(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
               f"&hourly={q}&models={','.join(MODELS)}&wind_speed_unit=mph&timezone=GMT&start_date={d0}&end_date={d1}", cache=d1 < dt.date.today())
    h = d["hourly"]; t = np.array([np.datetime64(x) for x in h["time"]])
    F = {l: {m: _arr(h, f"wind_gusts_10m{l}_{m}", len(t)) for m in MODELS} for l in LEADS}
    return t, F, {l: _cape(h, l, len(t)) for l in LEADS}


def _arr(h, k, n):
    return np.array([np.nan if v is None else v for v in h.get(k, [None] * n)], float)


def _cape(h, l, n):
    c = np.array([_arr(h, f"cape{l}_{m}", n) for m in CAPE_FROM])
    return np.where(np.isfinite(c).any(0), np.nanmean(np.where(np.isfinite(c), c, np.nan), 0), 0.0) if np.isfinite(c).any() else np.zeros(n)


def live(lat, lon, hours=96):
    d = _get()(f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}&hourly=wind_gusts_10m,cape"
               f"&models={','.join(MODELS)}&wind_speed_unit=mph&timezone=GMT&past_days=1&forecast_days={hours // 24 + 2}")
    h = d["hourly"]; t = np.array([np.datetime64(x) for x in h["time"]])
    return t, {m: _arr(h, f"wind_gusts_10m_{m}", len(t)) for m in MODELS}, _cape(h, "", len(t))


def observed(sid, d0, d1):
    """Hourly peak gust (mph) for the hour ending at each timestamp, from the ASOS 1-minute archive (peak 3-s gust
    each minute; ~1 week latency). An hour counts only with >= 45 minutes reported. Falls back to METARs (one 10-min
    gust per hour, so biased low) for stations without 1-minute data."""
    get = _get()
    raw = get(f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station={sid}&vars=gust_sknt&vars=sknt"
              f"&sts={d0}T00:00Z&ets={d1}T00:00Z&sample=1min&what=download&delim=comma&gis=no", raw=True, cache=d1 < dt.date.today(), timeout=180)
    peak, cnt = {}, {}
    for line in io.StringIO(raw.decode()).readlines()[1:]:
        f = line.strip().split(",")
        if len(f) < 5 or f[3] in ("", "M"):
            continue
        t = np.datetime64(f[2].replace(" ", "T")).astype("datetime64[h]") + 1
        peak[t] = max(peak.get(t, 0.0), float(f[3]) * 1.15078); cnt[t] = cnt.get(t, 0) + 1
    out = {t: v for t, v in peak.items() if cnt[t] >= 45}
    if len(out) > 24:
        return out
    raw = get(f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={sid}&data=sknt&data=gust"
              f"&year1={d0.year}&month1={d0.month}&day1={d0.day}&year2={d1.year}&month2={d1.month}&day2={d1.day}"
              f"&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&report_type=3&report_type=4", raw=True, cache=d1 < dt.date.today())
    for line in io.StringIO(raw.decode()).readlines()[1:]:
        _, ts, sk, gu = line.strip().split(",")
        if sk != "M":
            t = np.datetime64(ts.replace(" ", "T")).astype("datetime64[h]") + 1
            out[t] = max(out.get(t, 0.0), (float(gu) if gu != "M" else float(sk)) * 1.15078)
    return out


def dataset(lat, lon, days, end=None):
    """Pooled (time, station, X[n, models], y, cape) per lead over the last `days` days at the nearest stations."""
    end = end or dt.date.today() - dt.timedelta(days=1)
    d0 = end - dt.timedelta(days=days - 1)
    rows = {l: ([], [], [], [], []) for l in LEADS}
    st = stations(lat, lon)
    for s in st:
        t, F, C = archived(s["lat"], s["lon"], d0, end)
        ob = observed(s["iem"], d0, end + dt.timedelta(days=1))
        y = np.array([ob.get(x.astype("datetime64[h]"), np.nan) for x in t])
        for l in LEADS:
            T, S, X, Y, Cc = rows[l]
            T.append(t); S.append(np.full(len(t), s["id"])); X.append(np.c_[[F[l][m] for m in MODELS]].T); Y.append(y); Cc.append(C[l])
    return st, {l: tuple(np.concatenate(a) for a in rows[l]) for l in LEADS}


# ---------------------------------------------------------------- scores
def crps_norm(mu, sig, y):
    z = (y - mu) / sig
    return sig * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z * z / 2) / math.sqrt(2 * math.pi) - SQPI)


def crps_lognorm(mu, sig, y):
    """CRPS of LN(mu, sig) at y > 0, in y's units (Baran & Lerch 2015)."""
    z = (np.log(y) - mu) / sig
    return y * (2 * ndtr(z) - 1) - 2 * np.exp(mu + sig * sig / 2) * (ndtr(z - sig) + ndtr(sig / math.sqrt(2)) - 1)


# ---------------------------------------------------------------- EMOS
def _usable(X, keep=0.9):
    return [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).mean() >= keep]


def kappa(cape):
    return np.log1p(np.maximum(cape, 0) / 500)


def fit(X, y, cape=None):
    """Minimum-CRPS EMOS on log gust. X (n, M) mph with complete columns, y mph, cape J/kg. Returns params.
    mu = a + sum b ln x + k kappa;  sigma^2 = c^2 + d^2 s^2 + e^2 kappa,  kappa = ln(1 + CAPE/500)."""
    L, ly = np.log(np.maximum(X, 1)), np.log(np.maximum(y, 1))
    M = L.shape[1]; s2 = L.var(1); kp = kappa(np.zeros(len(y)) if cape is None else cape)
    def unpack(p):   # b = w^2, c, d, e via abs: unconstrained search, non-negative weights (bounded L-BFGS-B stalls on ties)
        return p[0], p[1:M + 1] ** 2, abs(p[M + 1]), abs(p[M + 2]), p[M + 3], abs(p[M + 4])
    def f(p):
        a, b, c, d, k, e = unpack(p)
        mu = a + L @ b + k * kp; sig = np.sqrt(c * c + d * d * s2 + e * e * kp) + 0.02
        return crps_norm(mu, sig, ly).mean() + 1e-3 * ((b - 1 / M) ** 2).sum()   # light ridge toward equal weights
    p0 = np.r_[0.0, np.full(M, math.sqrt(1 / M)), 0.25, 1.0, 0.0, 0.1]
    a, b, c, d, k, e = unpack(minimize(f, p0, method="BFGS").x)
    return dict(a=float(a), b=b.tolist(), c=float(c), d=float(d), k=float(k), e=float(e))


def predict(P, X, cape=None):
    L = np.log(np.maximum(X, 1)); s2 = L.var(1); kp = kappa(np.zeros(len(L)) if cape is None else cape)
    return P["a"] + L @ np.array(P["b"]) + P.get("k", 0) * kp, np.sqrt(P["c"] ** 2 + P["d"] ** 2 * s2 + P.get("e", 0) ** 2 * kp) + 0.02


def twcrps(mu, sig, y, u=20.0):
    """Threshold-weighted CRPS: only the part of the forecast above u mph counts (Gneiting & Ranjan 2011). Proper,
    unlike averaging CRPS over the hours that turned out windy (the 'forecaster's dilemma', Lerch et al. 2017)."""
    z = np.arange(u, 150, 0.5)[None, :]
    F = ndtr((np.log(z) - mu[:, None]) / sig[:, None])
    return (((F - (y[:, None] <= z)) ** 2).sum(1) * 0.5)


def baseline(X, sig_event=0.18, sig_hour=0.08):
    """What the engine did before: equal-weight geometric mean, fixed spread widened by model disagreement."""
    L = np.log(np.maximum(X, 1)); mu = L.mean(1)
    spread = np.abs(L - mu[:, None]).mean(1)
    return mu, np.sqrt(sig_event ** 2 + sig_hour ** 2 + spread ** 2)


def _rows(T, X, y, C, t0, t1, cols):
    m = (T >= t0) & (T < t1) & np.isfinite(y) & np.isfinite(X[:, cols]).all(1)
    m &= np.nanmean(np.where(np.isfinite(X), X, np.nan), 1) >= MIN_MPH
    return X[m][:, cols], np.maximum(y[m], 1.0), T[m], C[m]


def event_corr(res, T):
    """Share of forecast error that is common to all hours of a day (sets event vs hourly noise in the Monte Carlo)."""
    day = T.astype("datetime64[D]"); v = res.var()
    if v <= 0:
        return 0.7
    pairs = [(r[:, None] * r[None, :])[~np.eye(len(r), dtype=bool)].mean() for d in np.unique(day) if len(r := res[day == d]) > 4]
    return float(np.clip(np.mean(pairs) / v, 0, 0.95)) if pairs else 0.7


def backtest(lat, lon, test_days=21):
    """Rolling-origin verification of the old blend vs EMOS. Returns per-lead scores."""
    st, D = dataset(lat, lon, TRAIN_DAYS + test_days)
    out = dict(stations=[s["id"] for s in st], leads={})
    for l, (T, S, X, y, C) in D.items():
        cols = _usable(X[np.isfinite(y)])
        days = np.unique(T.astype("datetime64[D]"))[-test_days:]
        acc = {k: [] for k in ("base", "emos", "tw_b", "tw_e", "y", "mb", "me", "cov_b", "cov_e", "pit", "raw")}
        for d in days:
            Xtr, ytr, _, Ctr = _rows(T, X, y, C, d - TRAIN_DAYS, d, cols)
            Xte, yte, _, Cte = _rows(T, X, y, C, d, d + 1, cols)
            if len(ytr) < 200 or not len(yte):
                continue
            P = fit(Xtr, ytr, Ctr)
            me, se = predict(P, Xte, Cte); mb, sb = baseline(Xte)
            acc["emos"] += crps_lognorm(me, se, yte).tolist(); acc["base"] += crps_lognorm(mb, sb, yte).tolist()
            acc["tw_e"] += twcrps(me, se, yte).tolist(); acc["tw_b"] += twcrps(mb, sb, yte).tolist()
            acc["y"] += yte.tolist(); acc["mb"] += np.exp(mb).tolist(); acc["me"] += np.exp(me).tolist()
            q = lambda mu, sg, z: np.exp(mu + z * sg)
            acc["cov_b"] += ((yte >= q(mb, sb, -1.645)) & (yte <= q(mb, sb, 1.645))).tolist()
            acc["cov_e"] += ((yte >= q(me, se, -1.645)) & (yte <= q(me, se, 1.645))).tolist()
            acc["pit"] += ndtr((np.log(yte) - me) / se).tolist()
            acc["raw"] += np.abs(Xte - yte[:, None]).tolist()
        if not acc["y"]:
            continue
        y_ = np.array(acc["y"]); A = {k: np.array(v, float) for k, v in acc.items()}
        r2 = lambda v: round(float(v), 3)
        out["leads"]["0-24 h" if l == "" else "24-48 h"] = dict(
            n=len(y_), n_over_25mph=int((y_ >= 25).sum()),
            crps_old=r2(A["base"].mean()), crps_new=r2(A["emos"].mean()), skill=r2(1 - A["emos"].mean() / A["base"].mean()),
            tail_crps_old=r2(A["tw_b"].mean()), tail_crps_new=r2(A["tw_e"].mean()), tail_skill=r2(1 - A["tw_e"].mean() / A["tw_b"].mean()),
            mae_old=r2(np.abs(A["mb"] - y_).mean()), mae_new=r2(np.abs(A["me"] - y_).mean()),
            cover90_old=r2(A["cov_b"].mean()), cover90_new=r2(A["cov_e"].mean()),
            pit=np.histogram(A["pit"], 10, (0, 1))[0].tolist(),
            mae_by_model={NAMES[MODELS[c]]: r2(v) for c, v in zip(cols, A["raw"].mean(0))})
    return out


def calibrate(lat, lon, times):
    """Fit EMOS on the last TRAIN_DAYS and apply it to the live multi-model forecast at the site.
    Returns per-hour median gust (mph) and log-sigma, the event/hourly error split, weights and recent skill."""
    st, D = dataset(lat, lon, TRAIN_DAYS)
    tl, FL, CL = live(lat, lon)
    now = np.datetime64(times[0].replace(tzinfo=None), "h")
    med, sig, used = [], [], {}
    fits = {}
    for l, (T, S, X, y, C) in D.items():
        cols = [c for c in _usable(X[np.isfinite(y)]) if np.isfinite(FL[MODELS[c]]).any()]
        Xtr, ytr, Ttr, Ctr = _rows(T, X, y, C, T.min(), T.max() + 1, cols)
        if len(ytr) < 200:
            continue
        P = fit(Xtr, ytr, Ctr); mu, sg = predict(P, Xtr, Ctr)
        fits[l] = (P, cols, event_corr((np.log(np.maximum(ytr, 1)) - mu) / sg, Ttr))
        used["0-24 h" if l == "" else "24-48 h"] = {NAMES[MODELS[c]]: round(b, 3) for c, b in zip(cols, P["b"])} | dict(bias=round(P["a"], 3), cape_shift=round(P["k"], 3), cape_spread=round(P["e"], 3))
    if "" not in fits:
        return None
    for t in times:
        tt = np.datetime64(t.replace(tzinfo=None), "h"); lead = (tt - now).astype(int)
        P, cols, _ = fits["_previous_day1"] if lead >= 24 and "_previous_day1" in fits else fits[""]
        k = np.searchsorted(tl, tt)
        x = np.array([[FL[MODELS[c]][k] if k < len(tl) else np.nan for c in cols]])
        ok = np.isfinite(x[0])
        if not ok.any():
            med.append(None); sig.append(None); continue
        if not ok.all():   # a model ran out of forecast hours: fill with the mean of the rest (keeps the weights' meaning)
            x[0, ~ok] = np.exp(np.log(np.maximum(x[0, ok], 1)).mean())
        mu, sg = predict(P, x, np.array([CL[k]]))
        med.append(float(np.exp(mu[0]))); sig.append(float(sg[0]))
    ks = [np.searchsorted(tl, np.datetime64(t.replace(tzinfo=None), "h")) for t in times]
    raw = {NAMES[m]: [round(float(FL[m][k]), 1) if k < len(tl) and np.isfinite(FL[m][k]) else None for k in ks] for m in MODELS}
    return dict(median=med, sigma=sig, rho=fits[""][2], weights=used, stations=[s["id"] for s in st], train_days=TRAIN_DAYS, models=raw)


def selftest():
    rng = np.random.default_rng(1)
    # synthetic: truth = 1.1 x model A (B is noise); EMOS must weight A and correct the bias, beating equal weights
    n = 3000; t = rng.lognormal(math.log(15), .4, n); A = t / 1.1 * np.exp(rng.normal(0, .1, n)); B = t * np.exp(rng.normal(0, .5, n))
    X = np.c_[A, B]; P = fit(X, t)
    assert P["b"][0] > 2 * P["b"][1], P
    me, se = predict(P, X); mb, sb = baseline(X)
    assert crps_lognorm(me, se, t).mean() < 0.8 * crps_lognorm(mb, sb, t).mean()
    # lognormal CRPS agrees with the normal one in the small-sigma limit
    assert abs(crps_lognorm(np.log(20.0), 0.01, 21.0) - crps_norm(20.0, 0.2, 21.0)) < 0.02
    # CAPE: convective hours with fatter upper tail -> positive cape spread term
    cp = np.where(rng.random(n) < .2, 2500.0, 0.0); t2 = t * np.exp(np.where(cp > 0, rng.normal(.15, .35, n), 0))
    P2 = fit(X, t2, cp); assert P2["e"] > 0.15 and P2["k"] > 0, P2
    print("calib selftest ok")
