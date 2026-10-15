# StormVoyance

Storm impact simulator for any US address. Give it an address and it finds the building, measures it from USGS LiDAR,
simulates the wind around it on the GPU, and estimates damage to the building, trees, power and flooding from live
NHC/NWS forecasts.

## What it does

- **Building detection**: merges Google, OpenStreetMap, Microsoft and FEMA/ORNL footprints (consensus outline wins),
  measures roof heights from USGS 3DEP LiDAR, and classes the building (wood-frame house, multi-family, low-rise
  commercial, metal/industrial, engineered mid/high-rise).
- **Storm ensemble**: NHC-style 1000-member Monte Carlo (track/intensity error, Holland vortex, Kaplan-DeMaria decay),
  anchored to the NWS point forecast.
- **Per-run wind direction**: every ensemble run has its own wind direction (forecast direction plus its error, turned
  further when that run's hurricane passes on a different side of the house). CFD is run for each direction holding a
  real share of the damaging wind (up to 6, chosen to cover 90 %), and each run reads the flow for its own direction.
- **Tornadoes** (`tornado.py`): SPC's tornado outlook (chance within 25 mi, conditional-intensity level) drives path
  scenarios drawn from US climatology (rating, length, width, forward speed). Each passes the house as a translating
  modified Rankine vortex; the share of the 25-mile circle where the lot feels each wind level is integrated exactly, then
  scored with the building's damage curves and combined with the straight-line damage.
- **Wind CFD**: D3Q19 lattice-Boltzmann LES (regularized BGK + Smagorinsky) with a log-law ground wall model and
  synthetic inflow turbulence. Runs as a Metal kernel on Apple GPUs via PyTorch, falls back to numpy.
- **Damage**: Hazus-style lognormal fragilities per construction class, driven by the load ratio against the ASCE 7
  design gust.

## Validation

| Check | Result | Reference |
|---|---|---|
| Surface-mounted cube, reattachment length X_R/H | 1.38-1.40 | 1.4-1.6 (AIJ / COST 732) |
| Cube windward Cp | 0.85-0.89 | 0.7-0.8 |
| 5-site matrix (house, landmark, tower, warehouse, low-rise) | 4/5 pass on last full run | `validate_sites.py` |

Tornado layer check (`python3 tornado.py`): wood-frame damage medians land at 64 / 88 / 111 / 134 mph against the EF-scale
expected winds for one- and two-family homes (DI 2: 79 / 97 / 122 / 132 mph), within 20 %, and the swath geometry matches
the rated width.

Known limits: side/roof suction ~30-50% weaker than wind-tunnel data at the default resolution; suburban approach gust
~20-30% conservative versus ASCE 7 exposure B; non-house damage medians are engineering estimates; design wind speed
defaults to 110 mph (`--design-mph` to override). Tornado paths are straight with one
climatological heading and size distribution; tornado counts given one is nearby are a calibration knob.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # optional Google Maps key for 3D tiles
.venv/bin/python stormvoyance.py "3764 Elvis Presley Blvd, Memphis, TN 38116"
```

Web app:

```bash
cd web && npm install && npm run build && cd ..
.venv/bin/uvicorn server:app --port 8000
```

Checks:

```bash
.venv/bin/python stormvoyance.py --selftest
.venv/bin/python stormvoyance.py --validate-cfd
.venv/bin/python validate_sites.py
```

Not an official forecast. The National Weather Service governs.
