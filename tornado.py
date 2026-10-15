"""Tornado layer: the chance that a tornado path crosses the lot, and the wind and damage if one does.

The straight-line wind CFD cannot represent a tornado (a small, fast-moving vortex with updraft and a pressure deficit),
so this layer works the way tornado risk is assessed for siting and insurance:

1. Hazard. SPC's probabilistic tornado outlook gives p25, the probability of at least one tornado within 25 mi (40 km)
   of a point during the outlook day, plus a conditional-intensity level (CIG 1-3, or the older significant-tornado
   hatch) that shifts the expected ratings toward EF2+.
2. Path scenarios. Given a tornado in that circle, its rating, path length, path width, forward speed and position are
   drawn from US climatology. A path whose midpoint lies anywhere in the circle is equally likely, so the chance that
   the lot sees wind V is (area of the band around the path where wind >= V) / (circle area). That area is integrated
   exactly over cross-track offsets, so even tiny probabilities carry no sampling noise.
3. Wind on the house. A modified Rankine vortex (solid-body core, V ~ r^-0.6 outside; Wurman & Alexander 2005 radar
   profiles) translating with the storm. The rated EF speed is the peak ground-relative 3-s gust; the core radius is set
   so the rated damage width ends at the EF0 threshold (65 mph). Intensity varies along a path (the rating is its
   maximum), so the house sees a uniform 65-100 % of the peak.
4. Damage. The building's own class curves (load ratio against the ASCE 7 design gust). EF speeds are damage-equivalent
   (they were inferred from what broke), so no extra tornado load factor is applied; the self-test checks that the
   wood-frame curves land on the EF-scale degrees of damage for one- and two-family homes (DI 2).

Ceilings (ponytail): straight paths, one heading distribution for the whole US, climatological path sizes, expected
tornado count given one is a calibration knob. Upgrade path: storm-motion vectors from the HRRR, regional path
climatology, SPC mesoscale discussions and watches for the next few hours.
"""
import math, os, sys

import numpy as np
from scipy.special import ndtr

R25_KM = 40.234                                    # 25 statute miles
EF_MPH = [(65, 85), (86, 110), (111, 135), (136, 165), (166, 200), (201, 230)]   # EF-scale 3-s gust ranges
# Share of tornadoes by rating. Climatology: US 2007-2023 SPC database. Conditional-intensity levels shift the mix toward
# strong tornadoes the way SPC defines them (CIG1 ~ the old 10 % significant hatch: EF2+ likely; CIG2 EF3+; CIG3 EF4+).
EF_MIX = {0: [.53, .33, .10, .032, .007, .0006],
          1: [.30, .36, .20, .10, .035, .005],
          2: [.20, .30, .24, .16, .08, .02],
          3: [.12, .24, .24, .22, .14, .04]}
# Median path length (km) and maximum damage width (m) per rating, lognormal sigmas (SPC database; cf. Brooks 2004).
PATH_LEN_KM = [1.0, 3.2, 8.0, 17.0, 30.0, 45.0]
PATH_W_M = [25.0, 70.0, 150.0, 300.0, 500.0, 700.0]
SIG_LEN, SIG_W = 0.9, 0.7
VT_MPH = (30.0, 10.0)                              # forward speed, mean and sd (typical tornadic-storm motion)
DECAY = 0.6                                        # outer-flow decay exponent of the modified Rankine vortex
ALONG = (0.65, 1.0)                                # share of the rated peak seen at a point along the path (uniform)
N_GIVEN_ONE = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0}     # expected tornadoes in the circle given at least one (calibration knob)
EF_DI2 = [65, 79, 97, 122, 132]                    # EF-scale DI 2 expected winds: threshold, <20 % cover, deck uplift,
                                                   # large roof sections gone, exterior walls collapse


def spc_outlooks(get, lat, lon, t0, t1):
    """SPC day 1-2 tornado probabilities at a point for outlooks that overlap [t0, t1]. get(url) returns parsed JSON."""
    out = []
    for day in (1, 2):
        base = f"https://www.spc.noaa.gov/products/outlook/day{day}otlk_"
        torn = get(base + "torn.lyr.geojson")
        feats = torn.get("features") or []
        if not feats:
            continue
        pr = feats[0]["properties"]
        valid, expire = _iso(pr.get("VALID_ISO")), _iso(pr.get("EXPIRE_ISO"))
        if valid and expire and (expire <= t0 or valid >= t1):
            continue
        p = max([_num(f["properties"].get("LABEL")) for f in feats if _hit(f.get("geometry"), lon, lat)], default=0.0)
        cig = 0
        for layer in ("cigtorn", "sigtorn"):
            try:
                g = get(base + layer + ".lyr.geojson")
            except Exception:
                continue
            for f in g.get("features") or []:
                fp = f["properties"]
                if _iso(fp.get("VALID_ISO")) != valid or not _hit(f.get("geometry"), lon, lat):
                    continue
                lab = str(fp.get("LABEL", "")).upper()
                lvl = int(lab[-1]) if lab.startswith("CIG") and lab[-1:].isdigit() else 1 if lab.startswith("SIG") else 0
                cig = max(cig, lvl)
        out.append(dict(day=day, valid=pr.get("VALID_ISO"), expire=pr.get("EXPIRE_ISO"), p25=p, cig=cig,
                        label=pr.get("LABEL2") if p else "Less than 2 % (outside SPC's tornado areas)"))
    return out


def _iso(s):
    import datetime as dt
    return dt.datetime.fromisoformat(s) if s else None


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _hit(geom, x, y):
    if not geom:
        return False
    if geom["type"] == "GeometryCollection":
        return any(_hit(g, x, y) for g in geom["geometries"])
    if geom["type"] not in ("Polygon", "MultiPolygon"):
        return False
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    inside = False
    for poly in polys:
        for ring in poly:
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
                if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                    inside = not inside
    return inside


def peak_wind(vmax, w_m, vt, d_m):
    """Peak ground-relative wind (mph) felt at cross-track offset d_m (m, + = right of motion) as a vortex passes.
    vmax, w_m, vt: (n, 1) arrays; d_m: (k,) offsets. Returns (n, k)."""
    vrot = np.maximum(vmax - vt, 1.0)
    rc = w_m / 2 * np.clip((65.0 - vt) / vrot, 1e-3, 1.0) ** (1 / DECAY)    # rated width ends at 65 mph on the strong side
    x = np.linspace(-4, 4, 81)[None, None, :] * np.maximum(rc, np.abs(d_m))[..., None]
    y = d_m[None, :, None] + 0 * x
    r = np.hypot(x, y) + 1e-6
    v = np.where(r < rc[..., None], vrot[..., None] * r / rc[..., None], vrot[..., None] * (rc[..., None] / r) ** DECAY)
    # cyclonic rotation: tangential unit (-y, x)/r with y measured to the LEFT; d_m is to the right, so flip
    ux, uy = v * (y / r), v * (-x / r)        # rotation, (along-track, cross-track); right side adds to motion
    return np.hypot(ux + vt[..., None], uy).max(-1)


def hit_areas(fns, rng, cig=0, n=6000):
    """Expected area (km^2) around one tornado's path where fn(V) applies, for each fn in fns (fn: wind mph -> prob).
    Stratified by rating so strong-tornado tails are sampled as well as the common weak ones."""
    d = np.r_[-np.geomspace(4000, 1, 90), 0.0, np.geomspace(1, 4000, 90)]       # cross-track offsets (m)
    dd = np.gradient(d)
    tot = np.zeros(len(fns))
    for ef, share in enumerate(EF_MIX[cig]):
        k = max(200, int(n * share))
        lo, hi = EF_MPH[ef]
        vmax = rng.uniform(lo, hi, (k, 1))
        L = PATH_LEN_KM[ef] * np.exp(rng.normal(0, SIG_LEN, (k, 1)))
        W = PATH_W_M[ef] * np.exp(rng.normal(0, SIG_W, (k, 1)))
        vt = np.clip(rng.normal(*VT_MPH, (k, 1)), 10, 60)
        s = rng.uniform(*ALONG, (k, 1))
        V = peak_wind(vmax * s, W * s, vt, d)                                  # (k, len(d))
        for j, fn in enumerate(fns):
            p = fn(V)
            band = (L * (p * dd).sum(1, keepdims=True) / 1000).ravel()         # km * m / 1000 = km^2
            half = len(d) // 2                                                  # end caps: one disk, side-averaged
            rho, pr = d[half:], 0.5 * (p[:, half:] + p[:, half::-1])
            cap = (pr * np.pi * rho * np.gradient(rho)).sum(1) * 2 / 1e6
            tot[j] += share * (band + cap).mean()
    return tot


def tornado_risk(days, dmg_curves, design_gust_h, rng, age_k=1.0):
    """Per-day and combined probabilities for the lot. dmg_curves: [(median load ratio, beta on gust), ...] cosmetic..severe.
    design_gust_h: ASCE 7 design 3-s gust at mean roof height (mph)."""
    fns = [lambda V: (V >= 65).astype(float), lambda V: (V >= 86).astype(float), lambda V: (V >= 111).astype(float),
           lambda V: (V >= 136).astype(float)]
    fns += [(lambda m, b: lambda V: ndtr(np.log((V / design_gust_h) ** 2 / (m * age_k)) / (2 * b)))(m, b) for m, b in dmg_curves]
    keys = ["ef0", "ef1", "ef2", "ef3", "cosmetic", "minor", "moderate", "severe"]
    circle = math.pi * R25_KM ** 2
    out, comb = [], {k: 0.0 for k in keys}
    for dy in days:
        a = hit_areas(fns, rng, dy["cig"])
        lam = dy["p25"] * N_GIVEN_ONE[dy["cig"]]
        pr = {k: float(1 - math.exp(-lam * v / circle)) for k, v in zip(keys, a)}
        out.append(dict(dy, probs=pr, given_tornado_nearby={k: float(v / circle) for k, v in zip(keys, a)}))
        comb = {k: 1 - (1 - comb[k]) * (1 - pr[k]) for k in keys}
    return dict(days=out, probs=comb)


def selftest():
    rng = np.random.default_rng(1)
    # 1. the rated width ends at 65 mph on the strong side, the strongest pass equals the rating, the left side is weaker
    dg_ = np.linspace(-300, 300, 601)
    V = peak_wind(np.array([[150.0]]), np.array([[300.0]]), np.array([[30.0]]), dg_)[0]
    assert abs(V[450] - 65) < 3 and abs(V.max() - 150) < 3 and dg_[V.argmax()] > 0 and V[150] < V[450], V[[150, 300, 450]]
    # 2. band area of a fixed path matches the stadium formula 2 r L + pi r^2 (wind >= 65 mph out to r = W/2 on the right)
    d = np.r_[-np.geomspace(4000, 1, 90), 0.0, np.geomspace(1, 4000, 90)]
    Vd = peak_wind(np.array([[120.0]]), np.array([[200.0]]), np.array([[30.0]]), d)
    width = ((Vd[0] >= 65) * np.gradient(d)).sum()
    assert 100 < width < 200, width                                   # 100 m on the right, less on the weaker left
    # 3. wood-frame curves vs the EF-scale DI 2 degrees of damage (expected winds 79 / 97 / 122 / 132 mph)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling module under python -I
    from stormvoyance import K
    kz = 2.01 * (15 / 900) ** (2 / 9.5); dg = 110 * math.sqrt(kz)
    med = [dg * math.sqrt(m) for m, _ in K["dmg_class"]["WSF"]]
    for got, want in zip(med, EF_DI2[1:]):
        assert abs(got - want) / want < 0.20, (med, EF_DI2)
    # 4. probabilities: zero without a hazard, monotone in severity, tiny but nonzero at a 10 % day, more with CIG
    r0 = tornado_risk([dict(day=1, p25=0.0, cig=0)], K["dmg_class"]["WSF"], dg, rng)
    assert r0["probs"]["ef0"] == 0
    r1 = tornado_risk([dict(day=1, p25=0.10, cig=0)], K["dmg_class"]["WSF"], dg, rng)["probs"]
    r2 = tornado_risk([dict(day=1, p25=0.10, cig=2)], K["dmg_class"]["WSF"], dg, rng)["probs"]
    assert 1e-6 < r1["ef0"] < 1e-2 and r1["ef0"] >= r1["ef1"] >= r1["ef2"] >= r1["ef3"], r1
    assert r1["cosmetic"] >= r1["minor"] >= r1["moderate"] >= r1["severe"], r1
    assert r2["ef2"] > 2 * r1["ef2"], (r1, r2)
    print(f"tornado selftest ok (10 % day: P(EF0+ wind at lot) {r1['ef0']:.1e}, P(minor damage) {r1['minor']:.1e}; "
          f"WSF medians {[round(m) for m in med]} mph vs DI 2 {EF_DI2[1:]})")


if __name__ == "__main__":
    selftest()
