"""Site validation matrix: runs the full engine on very different buildings and checks that every result is physically
plausible. A run fails loudly instead of quietly drawing a wrong building.

    python3 validate_sites.py              # all sites
    python3 validate_sites.py house tower  # a subset by key

Checks per site: the building under the address was found (not assumed) and has the expected construction class; its
roof height is measured; neighbours were found; the CFD finished; at least one wind direction passed the pressure check
and every reported net Cp (external - internal) is inside the physical bounds; the approach-flow gust ratio is in a sane range; probabilities are
finite and ordered (cosmetic >= minor >= moderate)."""
import json, math, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = {   # key: (address, acceptable construction classes, minimum neighbours the datasets must yield)
    "house":     ("3764 Elvis Presley Blvd, Memphis, TN 38116", {"WSF", "MLR"}, 5),   # Graceland: a large house open to the public,
    "landmark":  ("30 W Pershing Rd, Kansas City, MO 64108", {"MLR", "MRB"}, 10),
    "tower":     ("333 Commerce St, Nashville, TN 37201", {"MRB"}, 20),
    "warehouse": ("500 Duke Dr, Lebanon, TN 37090", {"MBI", "MLR"}, 0),       # a distribution centre stands alone
    "apartments": ("1101 McGavock St, Nashville, TN 37203", {"WMF", "MRB", "MLR"}, 10),
}


def check(D, want, min_nb):
    H, S, W = D["house"], D["sim"], D.get("wind") or {}
    errs = []
    ok = lambda cond, msg: cond or errs.append(msg)
    ok(not H["src"].startswith("assumed"), f"building not found ({H['src']})")
    ok(H.get("cls") in want, f"class {H.get('cls')} not in {sorted(want)}")
    ok(str(H.get("roof_src", "")).startswith("USGS LiDAR"), f"roof height not measured ({H.get('roof_src')})")
    ok(0 < H["eave_m"] <= H["ridge_m"] < 600, f"roof heights {H['eave_m']}..{H['ridge_m']}")
    ok(len(D["buildings"]) >= min_nb, f"only {len(D['buildings'])} neighbours (expected >= {min_nb})")
    ok(bool(W.get("sectors")), "no CFD wind field")
    cps = [c for w in H.get("walls", []) for c in (w.get("cp") or []) if c is not None]
    ok(len(cps) > 0, "no direction passed the pressure check")
    # same bound as the engine's check: walls rising above the reference height see more than q(hm)
    z0 = max((x["z0"] for x in W.get("sectors") or []), default=.3); hm = (H["eave_m"] + H["ridge_m"]) / 2
    k = (math.log(max(H["ridge_m"], 2 * z0) / z0) / math.log(max(hm, 2 * z0) / z0)) ** 2
    ok(all(abs(c) <= 1.5 * k for c in cps), f"net wall Cp out of bounds: {min(cps, default=0):.2f}..{max(cps, default=0):.2f} (limit {1.5 * k:.2f})")
    ok(not cps or max(cps) - min(cps) > 0.15, "walls barely differ: no windward/leeward contrast")
    for a in H.get("approach") or []:
        ok(a is None or 0.3 < a < 2.2, f"approach gust ratio {a}")
    P = S["probs"]
    ok(all(v is None or (math.isfinite(v) and 0 <= v <= 1) for v in P.values()), "probability outside [0, 1]")
    ok(P["cosmetic"] >= P["minor"] >= P["moderate"] >= P["severe"], "damage states out of order")
    ok(S.get("dir_cover", 0) >= 0.6, f"simulated directions hold only {S.get('dir_cover', 0):.0%} of the damaging wind")
    T = S.get("tornado")
    ok(T is None or all(0 <= T["probs"][x] <= 1 for x in T["probs"]) and T["probs"]["ef0"] >= T["probs"]["ef1"] >= T["probs"]["ef2"],
       "tornado probabilities out of order")
    return errs


def main(keys):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    bad = 0
    for k in keys or SITES:
        addr, want, min_nb = SITES[k]
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "r.html")
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "stormvoyance.py"), addr, "--members", "300", "--json",
                                "-o", out], capture_output=True, text=True, env=env, cwd=HERE)
            log = [l for l in p.stderr.splitlines() if l.startswith(("•", "  !", "  CFD"))]
            try:
                with open(os.path.splitext(out)[0] + ".json") as fh:   # --json writes the data bundle beside the report
                    D = json.load(fh)
                errs = check(D, want, min_nb)
            except Exception as e:   # noqa: BLE001 - any failure is a failed site
                errs = [f"engine failed: {e}", *p.stderr.splitlines()[-5:]]
                D = None
        bad += bool(errs)
        print(f"{'PASS' if not errs else 'FAIL'}  {k:10s} {time.time() - t0:5.0f}s  {addr}")
        if D:
            H = D["house"]
            print(f"      {H.get('cls_label')}, {H['area']:,.0f} m², roof {H['eave_m']}-{H['ridge_m']} m, "
                  f"{len(D['buildings'])} neighbours, grid {H.get('cfd_grid')}, approach {H.get('approach')}")
            Sm = D["sim"]; T = Sm.get("tornado")
            dirs = [x["dir"] for x in (D.get("wind") or {}).get("sectors", [])]
            spc = ", ".join("day %d %.0f%%" % (d["day"], 100 * d["p25"]) for d in T["days"]) if T else "n/a"
            print(f"      wind dirs {dirs} hold {Sm.get('dir_cover', 0):.0%} of damaging wind; SPC tornado: {spc}")
        for e in errs:
            print("      ✗", e)
        for l in log:
            if "!" in l or "failed" in l:
                print("     ", l.strip())
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
