"""cfd: neighbourhood wind solver for StormVoyance.  VERSION 3

Two backends with the same inputs and outputs:
  * solve_metal: one fused Metal kernel per step on Apple GPUs (~10-20x faster): regularized BGK (Latt & Chopard 2006)
    + Smagorinsky LES, half-way bounce-back on buildings, and a log-law surface-stress wall model on the ground
    (free-slip floor + tau = rho (kappa / ln(z1/z0))^2 |u| u in the first cell; Moeng 1984, as in PALM). The engine uses
    it whenever it is available (best_solver()).
  * solve: the numpy reference (BGK + Smagorinsky, full-way bounce-back), used on machines without a GPU.
Validation (benchmark_cube, AIJ / COST 732 cube in a turbulent boundary layer, referenced to the undisturbed approach
flow from an identical empty run): GPU solver at 10 / 16 cells per H gives reattachment X_R/H 1.40 / 1.38 (tunnels
1.4-1.6) and windward Cp(2/3 H) 0.85 / 0.89 (tunnels 0.7-0.8).

D3Q19 lattice-Boltzmann, BGK collision + Smagorinsky LES closure (Hou et al. 1994),
the same family of solver used by rapid urban-wind tools (Autodesk Forma, Orbital
Stack, PALM-LBM studies). Obstacles:
  * terrain and buildings  -> no-slip bounce-back walls
  * tree crowns            -> porous canopy drag  F = -Cd * LAD * |u| u  (Shaw & Schumann 1992)
Inflow: neutral log-law profile for the upwind roughness length z0 (ESDU / ASCE 7) with
synthetic turbulence from the random-flow-generation method (Kraichnan 1970; Smirnov
et al. 2001): divergence-free Fourier modes, frozen-convected, scaled to the surface-layer
turbulence sigma_u = 2.5 u*, sigma_v = 0.75 sigma_u, sigma_w = 0.5 sigma_u (ESDU 85020).
Output is the time-averaged velocity normalised by the inflow speed at 10 m (the
"speed-up ratio" K used in pedestrian-wind and wind-tunnel practice) plus the
resolved turbulence. Because flow around sharp-edged bluff bodies is Reynolds-number
independent at storm speeds (AIJ guidelines, COST 732), one run per wind direction
can be scaled to any forecast speed.
"""
import math
import numpy as np

C = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
              (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
              (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, np.float32)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
Cf = C.astype(np.float32)


PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
BLOCK = 16384                       # cells per collision block in solve(): measured best on Apple M4 (8k-32k all close)


def feq(rho, u):
    cu = np.einsum("qd,d...->q...", Cf, u)
    usq = (u * u).sum(0)
    return W.reshape(-1, 1, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)


def log_profile(z_m, z0, zref=10.0):
    return np.log((np.maximum(z_m, 0) + z0) / z0) / math.log((zref + z0) / z0)


def rfg_modes(rng, n=64, min_freq=0.3):
    """Random-flow-generation modes (Smirnov et al. 2001). A mode whose frequency nearly cancels its convection
    (|omega - k_x| small) is a frozen pattern at the inlet: it never averages out and shows up as a fake steady streak in
    the mean wind, so such modes are redrawn until every one completes cycles within the averaging window."""
    k = rng.normal(0, math.sqrt(0.5), (n, 3)); om = rng.normal(0, 1, n)
    while (bad := np.abs(om - k[:, 0]) < min_freq).any():
        k[bad] = rng.normal(0, math.sqrt(0.5), (bad.sum(), 3)); om[bad] = rng.normal(0, 1, bad.sum())
    p = np.cross(rng.normal(0, 1, (n, 3)), k); q = np.cross(rng.normal(0, 1, (n, 3)), k)
    return k, p, q, om


def _streams(solid):
    """Per-direction streaming plan for a C-ordered (nx, ny, nz) field: np.roll(a, C[q], (0, 1, 2)) is a flat roll by
    s = cx*ny*nz + cy*nz + cz (one memcpy; the x wrap comes out right by itself) plus a re-copy of the y and z faces
    where the flat roll would carry a value into the neighbouring row instead of wrapping. With an open lid the z wrap
    is never read (downward populations there are overwritten by the mirror, upward ones land in a solid floor and
    never return to fluid), so it is skipped."""
    nx, ny, nz = solid.shape
    skip_dn = not solid[:, :, -1].any()
    skip_up = skip_dn and bool(solid[:, :, 0].all())
    plan = []
    for q in range(1, 19):
        cx, cy, cz = (int(v) for v in C[q])
        faces = []
        if cz and not (skip_dn if cz < 0 else skip_up):
            faces.append(((slice(None), slice(None), 0 if cz > 0 else nz - 1),
                          (slice(None), slice(None), nz - 1 if cz > 0 else 0), (cx, cy)))
        if cy:
            faces.append(((slice(None), 0 if cy > 0 else ny - 1, slice(None)),
                          (slice(None), ny - 1 if cy > 0 else 0, slice(None)), (cx, cz)))
        plan.append((q, cx * ny * nz + cy * nz + cz, faces))
    return plan


def _stream(src, dst, s, faces, shape):
    """dst = np.roll(src, C[q]) for one population (flat views in, flat views out)."""
    if s > 0:
        dst[s:] = src[:-s]; dst[:s] = src[-s:]
    else:
        dst[:s] = src[-s:]; dst[s:] = src[:-s]
    d3, s3 = dst.reshape(shape), src.reshape(shape)
    for dsl, ssl, sh in faces:
        d3[dsl] = np.roll(s3[ssl], sh, axis=(0, 1))


def _walls(solid):
    """Full-way bounce-back reduced to what can reach fluid: for each direction only the wall nodes whose neighbour that
    way is fluid. Elsewhere inside walls the textbook swap just ping-pongs the initial populations between neighbouring
    wall nodes, which matters only where a boundary copies wall nodes: the zero-gradient outlet (planes nx-3, nx-2) and,
    if a wall pierces it, the free-slip lid (z = 0 and nz-1, joined by the z wrap); those keep the full swap. Every
    other wall population is never read, so fluid nodes get exactly what a swap of every solid node would give them."""
    fluid = ~solid
    keep = solid.copy(); keep[0] = keep[-1] = False                  # inlet and outlet planes are overwritten
    full = np.zeros_like(solid); full[-3:-1] = solid[-3:-1]
    if solid[:, :, -1].any():
        full[1:-1, :, 0], full[1:-1, :, -1] = solid[1:-1, :, 0], solid[1:-1, :, -1]
    out = []
    for q in range(1, 19, 2):
        pair = []
        for d in (q, OPP[q]):
            nb = np.roll(fluid, tuple(-C[d]), axis=(0, 1, 2))         # nb[x] = fluid[x + c_d]
            if C[d, 2] < 0:
                nb[:, :, 0] = False                                  # wraps into the lid's downward populations
            pair.append(np.flatnonzero(keep & nb | full))
        out.append((q, OPP[q], pair[0], pair[1]))
    return out


def solve(solid, drag, dx, z0, U0=0.06, steps=1800, avg_from=900, Cs=0.17, nu0=6e-4, log=None, L_m=20.0, seed=0,
          inflow=None, pressure=False):
    """solid: bool (nx, ny, nz); drag: float Cd*LAD [1/m] (nx, ny, nz). Wind blows toward +x.
    inflow: optional (u (3, ny, nz), sigma (ny, nz)) in units of the reference speed, e.g. sampled from a coarser run
    (one-way nesting); default is the log law for z0. Returns mean velocity / U0 (3, nx, ny, nz), resolved turbulence
    sigma / U0 (nx, ny, nz) and, with pressure=True, the mean kinematic pressure / U0^2 (p = c_s^2 rho).

    Implementation (same numerics as the textbook form, restructured for numpy speed): float32 throughout, collision runs
    over cache-sized cell blocks with preallocated buffers. Per block one (10 x 19) @ (19 x n) product gives rho, j and
    the momentum flux Pi; the Smagorinsky stress uses Pi_neq = Pi - rho/3 I - rho u u (exact for the D3Q19 second-order
    equilibrium) and feq is one (19 x 10) product, so neither feq nor f - feq is built for the whole grid. Streaming is
    a flat memcpy per population and bounce-back skips wall nodes no fluid node can see (see _walls), so the mean
    pressure returned inside walls is not meaningful; every fluid value matches the textbook loop to float32 rounding."""
    nx, ny, nz = solid.shape
    N = nx * ny * nz
    zc = (np.arange(nz) + 0.5) * dx
    jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    if inflow is None:
        # inflow profile measured from the local ground at the inlet column
        prof = U0 * log_profile(zc, z0)
        uin = np.zeros((3, 1, ny, nz), np.float32)
        ground = solid[0].argmin(1) if solid[0].any() else np.zeros(ny, int)
        for j in range(ny):
            g = ground[j]; uin[0, 0, j, g:] = prof[: nz - g]
        sx = (U0 / math.log((10 + z0) / z0)) * np.array([1, 0.75, 0.5], np.float32)[:, None, None]
    else:
        ui, sg = inflow
        uin = (U0 * np.asarray(ui, np.float32))[:, None] * ~solid[0][None, None]
        # isotropic sigma -> surface-layer component ratios 1 : 0.75 : 0.5 with the same total energy
        sx = (U0 * np.asarray(sg, np.float32))[None] * np.array([1.29, 0.97, 0.64], np.float32)[:, None, None]
    # synthetic inflow turbulence (frozen turbulence convected at U0, integral length L_m):
    # phase = A(y, z) + B * step, so cos/sin of A are tabulated once and each step is two small matrix products
    k_, p_, q_, om = rfg_modes(np.random.default_rng(seed))
    Lc = L_m / dx
    fluid_in = (np.abs(uin[0, 0]) > 0).astype(np.float32)
    ph0 = (k_[:, 1, None, None] * jj + k_[:, 2, None, None] * kk) / Lc
    cA, sA = np.cos(ph0).reshape(len(om), -1), np.sin(ph0).reshape(len(om), -1)
    dph = U0 * (om - k_[:, 0]) / Lc
    amp = math.sqrt(2 / len(om))
    kd = (drag * dx).astype(np.float32).reshape(N)  # lattice drag coefficient per step
    porous = bool((kd > 0).any())
    tau0 = 0.5 + 3 * nu0
    f = feq(np.ones((nx, ny, nz), np.float32), np.broadcast_to(uin, (3, nx, ny, nz)).copy())
    g = np.empty_like(f)
    plan, walls = _streams(solid), _walls(solid)
    up_, dn = [q for q in range(19) if C[q, 2] > 0], [q for q in range(19) if C[q, 2] < 0]
    mirror = [(q, next(p for p in up_ if (C[p, :2] == C[q, :2]).all())) for q in dn]
    # moments: rho, j (3), Pi (xx, yy, zz, xy, xz, yz).  feq is linear in X = rho [1 - 1.5 u^2, u, u_a u_b]:
    # feq_q = w_q (X0 + 3 c.X1..3 + 4.5 c_a c_b X_ab), so all 19 come from one (19 x 10) product
    Cm = np.vstack([np.ones(19), Cf.T] + [Cf[:, a] * Cf[:, b] for a, b in PAIRS]).astype(np.float32)
    Bm = (W[:, None] * np.hstack([np.ones((19, 1)), 3 * Cf] + [(4.5 if a == b else 9) * Cf[:, a:a + 1] * Cf[:, b:b + 1]
                                                             for a, b in PAIRS])).astype(np.float32)
    w6 = (18 * math.sqrt(2) * Cs * Cs) ** 2 * np.array([1, 1, 1, 2, 2, 2], np.float32)
    fluid = (~solid).reshape(N).astype(np.float32); fluid2 = 2 * fluid
    B = min(N, BLOCK)                                  # cells per block: the step runs block by block in cache
    Mo, X, FE = np.empty((10, B), np.float32), np.empty((10, B), np.float32), np.empty((19, B), np.float32)
    U, T3 = np.empty((3, B), np.float32), np.empty((3, B), np.float32)
    usq, sp, fac, tau, t = (np.empty(B, np.float32) for _ in range(5))
    den = np.empty(B, np.float32) if porous else None
    su = np.zeros((3, N), np.float32); su2 = np.zeros(N, np.float32); sr = np.zeros(N, np.float32); n = 0
    with np.errstate(divide="ignore"):
        for it in range(steps):
            F, G = f.reshape(19, N), g.reshape(19, N)
            avg, umax = it >= avg_from, 0.0
            for i0 in range(0, N, B):
                m = min(B, N - i0); bl = slice(i0, i0 + m)
                Fb, Mb, Xb, FEb, Ub, Tb = F[:, bl], Mo[:, :m], X[:, :m], FE[:, :m], U[:, :m], T3[:, :m]
                usqb, spb, facb, taub, tb = usq[:m], sp[:m], fac[:m], tau[:m], t[:m]
                np.matmul(Cm, Fb, out=Mb)
                rho, J, P = Mb[0], Mb[1:4], Mb[4:10]
                np.divide(J, rho, out=Ub)
                np.multiply(Ub, Ub, out=Tb); np.sum(Tb, 0, out=spb); np.sqrt(spb, out=spb)
                if porous:                              # semi-implicit canopy drag (stable for dense crowns)
                    denb = den[:m]; np.multiply(kd[bl], spb, out=denb); denb += 1; spb /= denb
                # Mach limiter (rare gusty cells near sharp corners) and u = 0 in walls, as one factor
                np.maximum(spb, 1e-9, out=facb); np.divide(np.float32(0.22), facb, out=facb)
                np.minimum(facb, 1, out=facb); facb *= fluid[bl]
                np.multiply(spb, facb, out=usqb); np.square(usqb, out=usqb)
                if porous:
                    facb /= denb
                Ub *= facb
                np.multiply(Ub, rho, out=Xb[1:4])
                np.multiply(Xb[1:4], Ub, out=Xb[4:7])
                np.multiply(Xb[1], Ub[1], out=Xb[7]); np.multiply(Xb[1], Ub[2], out=Xb[8]); np.multiply(Xb[2], Ub[2], out=Xb[9])
                np.multiply(usqb, -1.5, out=Xb[0]); Xb[0] += 1; Xb[0] *= rho
                # Smagorinsky: local relaxation time from the non-equilibrium momentum flux (Hou et al. 1996),
                # Pi_neq = Pi - rho/3 I - rho u u (exact for the D3Q19 second-order equilibrium)
                P -= Xb[4:10]; np.divide(rho, 3, out=tb); P[:3] -= tb
                np.square(P, out=P); np.matmul(w6, P, out=spb); np.sqrt(spb, out=spb)
                spb /= rho; spb += tau0 * tau0; np.sqrt(spb, out=taub); taub += tau0
                taub /= fluid2[bl]                      # 2 in fluid, 0 in walls: tau = inf, solid nodes do not collide
                # BGK collision  f -= (f - feq) / tau
                np.matmul(Bm, Xb, out=FEb)
                np.subtract(Fb, FEb, out=FEb); FEb /= taub; Fb -= FEb
                if avg:
                    su[:, bl] += Ub; su2[bl] += usqb
                    if pressure:
                        sr[bl] += rho
                if log and it % 300 == 0:
                    umax = max(umax, float(np.abs(Ub).max()))
            n += avg
            np.copyto(G[0], F[0])
            for q, s, faces in plan:
                _stream(F[q], G[q], s, faces, (nx, ny, nz))
            # walls: full-way bounce-back
            for q, o, iq, io in walls:
                a, b = G[o][iq], G[q][io]; G[q][iq] = a; G[o][io] = b
            # top: free slip (specular reflection)
            for q, p in mirror:
                g[q, :, :, -1] = g[p, :, :, -1]
            # inlet: equilibrium at the log-law profile; outlet: zero gradient
            cb, sb = np.cos(dph * it), np.sin(dph * it)
            upr = amp * ((p_ * cb[:, None] + q_ * sb[:, None]).T @ cA + (q_ * cb[:, None] - p_ * sb[:, None]).T @ sA)
            ui = uin[:, 0] + (sx * upr.reshape(3, ny, nz) * fluid_in).astype(np.float32)
            ui = ui.reshape(3, -1); usi = (ui * ui).sum(0)
            Xi = np.concatenate([1 - 1.5 * usi[None], ui] + [ui[a:a + 1] * ui[b:b + 1] for a, b in PAIRS])
            g[:, 0] = (Bm @ Xi).reshape(19, ny, nz)       # feq(rho = 1, ui)
            g[:, -1] = g[:, -2]
            if log and it % 300 == 0:
                log(it, umax)
            if not np.isfinite(g[:, nx // 2].sum()):
                raise FloatingPointError(f"LBM diverged at step {it}")
            f, g = g, f
    mu = (su / n).reshape(3, nx, ny, nz)
    var = np.maximum(su2.reshape(nx, ny, nz) / n - (mu * mu).sum(0), 0)
    sig = np.sqrt(var / 3)                          # isotropic-equivalent per-component sigma
    if pressure:
        return mu / U0, sig / U0, (sr.reshape(nx, ny, nz) / n / 3) / (U0 * U0)
    return mu / U0, sig / U0


def solve_torch(solid, drag, dx, z0, U0=0.06, steps=1800, avg_from=900, Cs=0.17, nu0=6e-4, log=None, L_m=20.0, seed=0,
                inflow=None, pressure=False, regularized=True, device=None):
    """solve() on PyTorch (Apple GPU via Metal when present, else CPU threads): same lattice, boundaries, inflow and
    outputs. regularized=True replaces plain BGK by regularized BGK (Latt & Chopard 2006): the non-equilibrium part is
    projected onto its second-order (stress) moment before relaxing, which removes the ghost modes that make BGK-LES noisy
    and over-dissipative near sharp building edges at high Reynolds number. Walls use full-way bounce-back on every
    solid node (the textbook form; solid nodes do not collide)."""
    import torch
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    f32 = dict(dtype=torch.float32, device=dev)
    nx, ny, nz = solid.shape
    N = nx * ny * nz
    zc = (np.arange(nz) + 0.5) * dx
    jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    if inflow is None:
        prof = U0 * log_profile(zc, z0)
        uin = np.zeros((3, ny, nz), np.float32)
        ground = solid[0].argmin(1) if solid[0].any() else np.zeros(ny, int)
        for j in range(ny):
            g = ground[j]; uin[0, j, g:] = prof[: nz - g]
        sx = (U0 / math.log((10 + z0) / z0)) * np.array([1, 0.75, 0.5], np.float32)[:, None, None]
    else:
        ui, sg = inflow
        uin = (U0 * np.asarray(ui, np.float32)) * ~solid[0][None]
        sx = (U0 * np.asarray(sg, np.float32))[None] * np.array([1.29, 0.97, 0.64], np.float32)[:, None, None]
    k_, p_, q_, om = rfg_modes(np.random.default_rng(seed))
    Lc = L_m / dx
    fluid_in = (np.abs(uin[0]) > 0).astype(np.float32)
    ph0 = (k_[:, 1, None, None] * jj + k_[:, 2, None, None] * kk) / Lc
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), **f32)
    cA, sA = T(np.cos(ph0).reshape(len(om), -1)), T(np.sin(ph0).reshape(len(om), -1))
    dph = T(U0 * (om - k_[:, 0]) / Lc)
    amp = math.sqrt(2 / len(om))
    P_, Q_ = T(p_), T(q_)
    uin_t, sxin = T(uin).reshape(3, -1), (T(sx) * T(fluid_in)[None]).reshape(3, -1)
    sol = torch.as_tensor(solid, device=dev).reshape(N); sol4 = sol.reshape(1, nx, ny, nz)
    fluid = (~sol).to(torch.float32)
    kd = T(drag * dx).reshape(N); porous = bool((kd > 0).any())
    tau0 = 0.5 + 3 * nu0
    Cm = T(np.vstack([np.ones(19), Cf.T] + [Cf[:, a] * Cf[:, b] for a, b in PAIRS]))
    Bm = T(W[:, None] * np.hstack([np.ones((19, 1)), 3 * Cf] + [(4.5 if a == b else 9) * Cf[:, a:a + 1] * Cf[:, b:b + 1] for a, b in PAIRS]))
    # second-order Hermite projection of Pi_neq: f_neq_q = w_q 4.5 (c_a c_b - delta_ab / 3) Pi_neq_ab (off-diagonals twice)
    Hm = T(W[:, None] * np.stack([(4.5 if a == b else 9) * (Cf[:, a] * Cf[:, b] - (1 / 3 if a == b else 0)) for a, b in PAIRS], 1))
    w6 = T((18 * math.sqrt(2) * Cs * Cs) ** 2 * np.array([1, 1, 1, 2, 2, 2]))
    opp = torch.as_tensor(OPP, device=dev)
    shifts = [tuple(int(v) for v in C[q]) for q in range(19)]
    up_, dn = [q for q in range(19) if C[q, 2] > 0], [q for q in range(19) if C[q, 2] < 0]
    mirror = [(q, next(p for p in up_ if (C[p, :2] == C[q, :2]).all())) for q in dn]
    mq, mp = torch.as_tensor([a for a, _ in mirror], device=dev), torch.as_tensor([b for _, b in mirror], device=dev)

    def feq_of(X):
        return Bm @ X
    def Xof(rho, u):
        usq = (u * u).sum(0)
        return torch.cat([(rho * (1 - 1.5 * usq))[None], rho * u] + [(rho * u[a] * u[b])[None] for a, b in PAIRS])
    rho1 = torch.ones(N, **f32)
    u_init = uin_t[:, None].expand(3, nx, ny * nz).reshape(3, N)
    f = feq_of(Xof(rho1, u_init))
    su = torch.zeros(3, N, **f32); su2 = torch.zeros(N, **f32); sr = torch.zeros(N, **f32); n = 0
    lim = torch.tensor(0.22, **f32)
    for it in range(steps):
        M = Cm @ f
        rho, J, Pi = M[0], M[1:4], M[4:10]
        u = J / rho
        sp = torch.sqrt((u * u).sum(0))
        if porous:
            den = 1 + kd * sp; sp = sp / den
        fac = torch.minimum(lim / torch.clamp(sp, min=1e-9), torch.ones_like(sp)) * fluid
        if porous:
            fac = fac / den
        u = u * fac
        X = Xof(rho, u)
        Pneq = Pi - X[4:10]; Pneq[:3] -= rho / 3
        tau = 0.5 * (tau0 + torch.sqrt(tau0 * tau0 + torch.sqrt(w6 @ (Pneq * Pneq)) / rho))   # Hou et al. (1996)
        tau = torch.where(sol, torch.full_like(tau, float("inf")), tau)
        feq = Bm @ X
        if regularized:
            fn = torch.where(sol[None], f, feq + (1 - 1 / tau) * (Hm @ Pneq))
        else:
            fn = f - (f - feq) / tau
        if it >= avg_from:
            su += u; su2 += (u * u).sum(0)
            if pressure:
                sr += rho
            n += 1
        fn4 = fn.reshape(19, nx, ny, nz)
        g = torch.empty_like(fn4); g[0] = fn4[0]
        for q in range(1, 19):
            g[q] = torch.roll(fn4[q], shifts[q], (0, 1, 2))
        g = torch.where(sol4, g[opp], g)                                      # full-way bounce-back
        g[mq, :, :, -1] = g[mp, :, :, -1]                                      # top: free slip
        cb, sbv = torch.cos(dph * it), torch.sin(dph * it)
        upr = amp * ((P_ * cb[:, None] + Q_ * sbv[:, None]).T @ cA + (Q_ * cb[:, None] - P_ * sbv[:, None]).T @ sA)
        ui_ = uin_t + sxin * upr
        g[:, 0] = feq_of(Xof(torch.ones(ny * nz, **f32), ui_)).reshape(19, ny, nz)
        g[:, -1] = g[:, -2]
        if log and it % 300 == 0:
            log(it, float(sp.max()))
        if it % 1000 == 999 and not bool(torch.isfinite(g[:, nx // 2]).all()):
            raise FloatingPointError(f"LBM diverged at step {it}")
        f = g.reshape(19, N)
    mu = (su / n).reshape(3, nx, ny, nz)
    var = torch.clamp(su2.reshape(nx, ny, nz) / n - (mu * mu).sum(0), min=0)
    out = [(mu / U0).cpu().numpy(), (torch.sqrt(var / 3) / U0).cpu().numpy()]
    if pressure:
        out.append(((sr.reshape(nx, ny, nz) / n / 3) / (U0 * U0)).cpu().numpy())
    return tuple(out)


_METAL = r"""
#include <metal_stdlib>
using namespace metal;
constant int CX[19] = {0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0};
constant int CY[19] = {0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1};
constant int CZ[19] = {0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1};
constant int OP[19] = {0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
constant int MZ[19] = {0,1,2,3,4,6,5,7,8,9,10,13,14,11,12,17,18,15,16};      // q -> (cx, cy, -cz)
constant int MR[19] = {0,1,2,3,4,5,5,7,8,9,10,11,14,11,14,15,18,15,18};   // downward q -> upward twin (same cx, cy)
constant float WQ[19] = {1.f/3, 1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
                         1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};

// one lattice update per cell, pull scheme: gather streamed populations (half-way bounce-back where the upstream
// neighbour is solid, free-slip mirror at the lid, periodic sides), then regularized (or plain) BGK with a Smagorinsky
// relaxation time, semi-implicit canopy drag and the Mach limiter, and accumulate the time averages.
kernel void lbm(device const float* fin [[buffer(0)]], device float* fout [[buffer(1)]], device const uchar* solid [[buffer(2)]],
                device const float* kd [[buffer(3)]], device float* acc [[buffer(4)]],
                constant uint& nx [[buffer(5)]], constant uint& ny [[buffer(6)]], constant uint& nz [[buffer(7)]],
                constant float& tau0 [[buffer(8)]], constant float& smag [[buffer(9)]], constant uint& avg [[buffer(10)]],
                constant uint& reg [[buffer(11)]], uint i [[thread_position_in_grid]]) {
  uint N = nx * ny * nz;
  if (i >= N || solid[i]) return;
  int z = i % nz, y = (i / nz) % ny, x = i / (nz * ny);
  float f[19];
  for (int q = 0; q < 19; q++) {
    int qq = q, sx = x - CX[q], sy = y - CY[q], sz = z - CZ[q];
    if (sz >= int(nz)) { qq = MR[q]; sx = x - CX[qq]; sy = y - CY[qq]; sz = z - CZ[qq]; }   // lid: free slip
    sx = clamp(sx, 0, int(nx) - 1); sy = (sy + int(ny)) % int(ny);                         // x ends are overwritten
    uint s = (uint(sx) * ny + uint(sy)) * nz + uint(sz);
    if (solid[s] == 2 && CZ[qq] > 0) {       // ground below: specular reflection (free slip); the wall model adds the stress
      uint s2 = (uint(sx) * ny + uint(sy)) * nz + uint(z);
      if (!solid[s2]) { f[q] = fin[MZ[qq] * N + s2]; continue; }
    }
    f[q] = solid[s] ? fin[OP[qq] * N + i] : fin[qq * N + s];
  }
  float rho = 0, jx = 0, jy = 0, jz = 0, pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
  for (int q = 0; q < 19; q++) {
    float v = f[q], cx = CX[q], cy = CY[q], cz = CZ[q];
    rho += v; jx += v * cx; jy += v * cy; jz += v * cz;
    pxx += v * cx * cx; pyy += v * cy * cy; pzz += v * cz * cz; pxy += v * cx * cy; pxz += v * cx * cz; pyz += v * cy * cz;
  }
  float ux = jx / rho, uy = jy / rho, uz = jz / rho, sp = sqrt(ux * ux + uy * uy + uz * uz);
  float den = 1 + kd[i] * sp; sp /= den;
  float fac = min(0.22f / max(sp, 1e-9f), 1.f) / den;
  ux *= fac; uy *= fac; uz *= fac;
  float usq = ux * ux + uy * uy + uz * uz, r3 = rho / 3;
  float nxx = pxx - rho * ux * ux - r3, nyy = pyy - rho * uy * uy - r3, nzz = pzz - rho * uz * uz - r3;
  float nxy = pxy - rho * ux * uy, nxz = pxz - rho * ux * uz, nyz = pyz - rho * uy * uz;
  float Q = sqrt(nxx * nxx + nyy * nyy + nzz * nzz + 2 * (nxy * nxy + nxz * nxz + nyz * nyz));
  float tau = 0.5f * (tau0 + sqrt(tau0 * tau0 + smag * Q / rho)), om = 1 / tau;
  for (int q = 0; q < 19; q++) {
    float cx = CX[q], cy = CY[q], cz = CZ[q], cu = cx * ux + cy * uy + cz * uz;
    float feq = WQ[q] * rho * (1 + 3 * cu + 4.5f * cu * cu - 1.5f * usq);
    float out;
    if (reg) {
      float h = 4.5f * ((cx * cx - 1.f/3) * nxx + (cy * cy - 1.f/3) * nyy + (cz * cz - 1.f/3) * nzz)
              + 9.f * (cx * cy * nxy + cx * cz * nxz + cy * cz * nyz);
      out = feq + (1 - om) * WQ[q] * h;
    } else out = f[q] - om * (f[q] - feq);
    fout[q * N + i] = out;
  }
  if (avg) { acc[i] += ux; acc[N + i] += uy; acc[2 * N + i] += uz; acc[3 * N + i] += usq; acc[4 * N + i] += rho; }
}
"""
_lib = None


def _check_tables():
    """The kernel's lattice tables must match C (a wrong mirror entry silently turns the lid into a brake)."""
    import re
    tab = {k: [int(v) for v in re.search(k + r"\[19\] = \{([^}]*)\}", _METAL)[1].split(",")] for k in ("CX", "CY", "CZ", "OP", "MZ", "MR")}
    for q in range(19):
        cx, cy, cz = (int(v) for v in C[q])
        assert (tab["CX"][q], tab["CY"][q], tab["CZ"][q]) == (cx, cy, cz) and tab["OP"][q] == OPP[q], q
        assert tuple(C[tab["MZ"][q]]) == (cx, cy, -cz), ("MZ", q)
        assert tuple(C[tab["MR"][q]]) == (cx, cy, abs(cz)), ("MR", q)

def solve_metal(solid, drag, dx, z0, U0=0.06, steps=1800, avg_from=900, Cs=0.17, nu0=6e-4, log=None, L_m=20.0, seed=0,
                inflow=None, pressure=False, regularized=True, terrain=None, z0_ground=0.03):
    """solve() as one fused Metal kernel per step (Apple GPU): streaming + half-way bounce-back + regularized
    Smagorinsky-BGK + canopy drag. ~20-40x faster than the numpy solver, which is what makes building-resolving grids
    (>= 10-20 cells over the building height) affordable. Same inflow (log law or nested plane + synthetic turbulence),
    lid and outlet conditions and outputs as solve(). Half-way bounce-back puts walls midway between a solid and a
    fluid node (second-order), where the numpy solver's full-way scheme is first-order in time.
    terrain: optional bool mask of the solid cells that are ground (not buildings). Their surface is free-slip and the
    first fluid cell above carries the log-law wall stress tau = rho (kappa / ln(z1 / z0_ground))^2 |u| u, z1 = dx / 2
    (the surface-stress wall model of atmospheric LES, Moeng 1984; PALM): without it a no-slip floor on a metre-scale
    grid under-resolves the surface layer, the approach wind decays near the ground and pressures on buildings shrink."""
    global _lib
    import torch
    if _lib is None:
        _check_tables()
        _lib = torch.mps.compile_shader(_METAL)
    dev = torch.device("mps"); f32 = dict(dtype=torch.float32, device=dev)
    nx, ny, nz = solid.shape; N = nx * ny * nz
    zc = (np.arange(nz) + 0.5) * dx
    jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    if inflow is None:
        prof = U0 * log_profile(zc, z0)
        uin = np.zeros((3, ny, nz), np.float32)
        ground = solid[0].argmin(1) if solid[0].any() else np.zeros(ny, int)
        for j in range(ny):
            g = ground[j]; uin[0, j, g:] = prof[: nz - g]
        sx = (U0 / math.log((10 + z0) / z0)) * np.array([1, 0.75, 0.5], np.float32)[:, None, None]
    else:
        ui, sg = inflow
        uin = (U0 * np.asarray(ui, np.float32)) * ~solid[0][None]
        sx = (U0 * np.asarray(sg, np.float32))[None] * np.array([1.29, 0.97, 0.64], np.float32)[:, None, None]
    k_, p_, q_, om = rfg_modes(np.random.default_rng(seed))
    Lc = L_m / dx
    fluid_in = (np.abs(uin[0]) > 0).astype(np.float32)
    ph0 = (k_[:, 1, None, None] * jj + k_[:, 2, None, None] * kk) / Lc
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), **f32)
    cA, sA = T(np.cos(ph0).reshape(len(om), -1)), T(np.sin(ph0).reshape(len(om), -1))
    dph = T(U0 * (om - k_[:, 0]) / Lc); amp = math.sqrt(2 / len(om)); P_, Q_ = T(p_), T(q_)
    uin_t, sxin = T(uin).reshape(3, -1), (T(sx) * T(fluid_in)[None]).reshape(3, -1)
    Bm = T(W[:, None] * np.hstack([np.ones((19, 1)), 3 * Cf] + [(4.5 if a == b else 9) * Cf[:, a:a + 1] * Cf[:, b:b + 1] for a, b in PAIRS]))
    def feq(u):                                   # rho = 1
        usq = (u * u).sum(0)
        return Bm @ torch.cat([(1 - 1.5 * usq)[None], u] + [(u[a] * u[b])[None] for a, b in PAIRS])
    flag = solid.astype(np.uint8)
    kdn = (drag * dx).astype(np.float32)
    if terrain is not None:
        g_ = terrain & solid; flag[g_] = 2
        above = np.zeros_like(solid); above[:, :, 1:] = g_[:, :, :-1] & ~solid[:, :, 1:]
        z0g = min(z0_ground, dx / 4)
        kdn = kdn + above * np.float32((0.4 / math.log((dx / 2) / z0g)) ** 2)
    sol = torch.as_tensor(np.ascontiguousarray(flag), device=dev).reshape(N)
    kd = T(kdn).reshape(N)
    fa = feq(uin_t[:, None].expand(3, nx, ny * nz).reshape(3, N)).contiguous(); fb = fa.clone()
    acc = torch.zeros(5, N, **f32)
    tau0 = 0.5 + 3 * nu0; smag = 18 * math.sqrt(2) * Cs * Cs
    for it in range(steps):
        _lib.lbm(fa, fb, sol, kd, acc, nx, ny, nz, tau0, smag, int(it >= avg_from), int(regularized), threads=N)
        g = fb.view(19, nx, ny * nz)
        cb, sbv = torch.cos(dph * it), torch.sin(dph * it)
        upr = amp * ((P_ * cb[:, None] + Q_ * sbv[:, None]).T @ cA + (Q_ * cb[:, None] - P_ * sbv[:, None]).T @ sA)
        g[:, 0] = feq(uin_t + sxin * upr)                                    # inlet
        g[:, -1] = g[:, -2]                                                  # outlet: zero gradient
        if log and it % 300 == 0:
            log(it, 0.0)
        if it % 1000 == 999 and not bool(torch.isfinite(g[:, nx // 2]).all()):
            raise FloatingPointError(f"LBM diverged at step {it}")
        fa, fb = fb, fa
    n = max(1, steps - avg_from)
    a = (acc / n).cpu().numpy()
    mu = a[:3].reshape(3, nx, ny, nz); var = np.maximum(a[3].reshape(nx, ny, nz) - (mu * mu).sum(0), 0)
    out = [mu / U0, np.sqrt(var / 3) / U0]
    if pressure:
        out.append((a[4].reshape(nx, ny, nz) / 3) / (U0 * U0))
    return tuple(out)


def best_solver():
    """Fastest available backend: the fused Metal kernel on Apple GPUs, else the numpy solver."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return solve_metal
    except ImportError:
        pass
    return solve


def benchmark_cube(H=10, steps=None, solver=None, **kw):
    """Surface-mounted cube in a turbulent boundary layer, the standard validation case for building CFD
    (Castro & Robins 1977; Meinders 1998; AIJ benchmark B; COST 732). Cube height H cells, z0 = H/100.
    Returns the centreline reattachment length behind the cube X_R/H and the windward-face Cp at 2/3 H.
    Reference: X_R/H ~ 1.4-1.6 (boundary-layer experiments), windward Cp(2/3 H) ~ +0.7-0.8."""
    nx, ny, nz = 13 * H, 7 * H, 4 * H
    solid = np.zeros((nx, ny, nz), bool); solid[:, :, 0] = True
    x0, y0 = 4 * H, ny // 2 - H // 2
    solid[x0:x0 + H, y0:y0 + H, 1:H + 1] = True
    dx = 1.0; z0 = H / 100
    steps = steps or int(5 * nx / 0.06)              # five flow-throughs; mean over the last 60 %
    if (solver or solve) is not solve:                # GPU solver: the floor gets the surface-stress wall model
        kw = dict(dict(terrain=np.zeros(solid.shape, bool), z0_ground=z0), **kw); kw["terrain"][:, :, 0] = True
    mu, _, p = (solver or solve)(solid, np.zeros(solid.shape, np.float32), dx, z0, steps=steps, avg_from=int(steps * .4),
                                 pressure=True, L_m=2.0 * H, **kw)
    jc = y0 + H // 2
    u_low = mu[0, x0 + H:, jc, 1]                   # first fluid cell above the floor, downstream of the lee face
    i0 = int(np.argmin(u_low))                     # core of the reverse-flow bubble (skips the small corner eddy)
    pos = np.nonzero(u_low[i0:] > 0)[0]
    xr = (i0 + pos[0]) / H if len(pos) and u_low[i0] < 0 else float("nan")
    zr = 1 + int(round(2 * H / 3))
    # reference: the undisturbed approach flow, as in the experiments -- an identical run without the cube, read at the
    # cube's windward face (velocity at H; static pressure at the same point)
    empty = solid.copy(); empty[x0:x0 + H, y0:y0 + H, 1:H + 1] = False
    mu0, _, p0 = (solver or solve)(empty, np.zeros(solid.shape, np.float32), dx, z0, steps=steps, avg_from=int(steps * .4),
                                   pressure=True, L_m=2.0 * H, **kw)
    U_H = mu0[0, x0, :, 1 + H].mean()
    face = (slice(x0 - 2, x0), slice(jc - 1, jc + 2), slice(zr - 1, zr + 2))   # 2x3x3 patch: averages out the odd-even mode
    up = (slice(3, x0 - 2 * H), slice(None), slice(2 * H, 3 * H))   # shared upstream band: aligns the runs' pressure levels
    cp = 2 * (p[face].mean() - p0[face].mean() - (p[up].mean() - p0[up].mean())) / U_H ** 2
    return dict(xr_over_h=round(float(xr), 2), cp_windward=round(float(cp), 2), cells_per_h=H)


if __name__ == "__main__":
    # self-check on rough terrain with a 1-cell wall, an overhang and canopy, plus a hole in the floor (open lid) or a
    # tower through the lid: streaming equals np.roll wherever the wrap is read, and the reduced bounce-back leaves every
    # fluid value bit-identical to the textbook swap of every solid node
    rng = np.random.default_rng(7)
    for lid in (False, True):
        sol = np.arange(12) < rng.integers(1, 4, (30, 20))[..., None]
        sol[8:12, 5:9, :6] = sol[15:18, 12:15, 6:8] = sol[22, :, :5] = True
        if lid:
            sol[12:14, 8:10, :] = True
        else:
            sol[20, 3, 0] = False
        a = rng.random(sol.shape, np.float32); b = np.empty_like(a)
        for q, s, faces in _streams(sol):
            _stream(a.reshape(-1), b.reshape(-1), s, faces, sol.shape)
            assert np.array_equal(b, np.roll(a, tuple(C[q]), (0, 1, 2))) or not lid and C[q, 2] < 0, q
        drag = np.where(~sol & (np.arange(12) < 7), 0.2, 0).astype(np.float32)
        reduced = solve(sol, drag, 2.0, 0.3, steps=80, avg_from=40, pressure=True)
        _walls, _w = (lambda s: [(q, OPP[q], np.flatnonzero(s), np.flatnonzero(s)) for q in range(1, 19, 2)]), _walls
        textbook = solve(sol, drag, 2.0, 0.3, steps=80, avg_from=40, pressure=True)
        _walls = _w
        assert all(np.array_equal(x, y) for x, y in zip(reduced[:2], textbook[:2]))
        assert np.array_equal(reduced[2][~sol], textbook[2][~sol]) and np.isfinite(reduced[0]).all()
    if best_solver() is solve_metal:
        # GPU kernel: lattice tables, and uniform flow between a free-slip floor and the free-slip lid must pass through
        # unchanged (a wrong mirror entry turned the lid into a brake that halved the wind)
        _check_tables()
        sol = np.zeros((60, 12, 14), bool); sol[:, :, 0] = True
        ui = np.zeros((3, 12, 14), np.float32); ui[0, :, 1:] = 1
        mu, _ = solve_metal(sol, np.zeros(sol.shape, np.float32), 1.0, .03, steps=2500, avg_from=1500,
                            inflow=(ui, np.zeros((12, 14), np.float32)), terrain=sol.copy(), z0_ground=1e-12)
        assert np.allclose(mu[0, 2:-2, :, 1:], 1, atol=.01) and np.abs(mu[1:, 2:-2]).max() < .01, mu[0, :, 6, 7]
    print("cfd selftest ok")
