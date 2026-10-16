// Google's own photorealistic 3D map (Maps JavaScript API <gmp-map-3d>; works on the free Maps Demo Key) as a
// background layer. Our three.js canvas sits on top, transparent, and copies Google's camera every frame.
import * as THREE from "three";
import type { Model } from "./model";

type G = { maps: { importLibrary(n: string): Promise<Record<string, unknown>> } };
type LLA = { lat: number; lng: number; altitude: number };
type Map3D = HTMLElement & { center: LLA; range: number; tilt: number; heading: number; fov?: number };

let loading: Promise<G> | null = null;
function loadMaps(key: string, onAuthFail: () => void): Promise<G> {
  (window as unknown as { gm_authFailure: () => void }).gm_authFailure = onAuthFail;   // Maps JS calls this on a rejected key
  return loading ??= new Promise((ok, fail) => {
    const cb = "__stormvoyanceMaps";
    (window as unknown as Record<string, () => void>)[cb] = () => ok((window as unknown as { google: G }).google);
    const s = document.createElement("script");
    s.src = `https://maps.googleapis.com/maps/api/js?${new URLSearchParams({ key, v: "beta", libraries: "maps3d", loading: "async", callback: cb })}`;
    s.onerror = () => { loading = null; fail(new Error("Maps JavaScript API failed to load")); };
    document.head.appendChild(s);
  });
}

export async function createMap3d(M: Model, host: HTMLElement, key: string, onFail: (why: string) => void) {
  const { D } = M, lat0 = D.meta.lat, lon0 = D.meta.lon, g0 = (D.site?.elev_ft ?? 0) * .3048;   // house ground, m above sea level
  const mE = 111320 * Math.cos(lat0 * Math.PI / 180), mN = 110540;
  const google = await loadMaps(key, () => onFail("the API key was rejected"));
  const { Map3DElement } = await google.maps.importLibrary("maps3d") as { Map3DElement: new (o: object) => Map3D };
  // keep the camera over the simulated neighbourhood: no zooming out to the globe, no panning off to other towns
  const RMAX = Math.max(650, Math.hypot(D.house.L, D.house.W) * 2.8);   // big buildings need the room on a portrait screen
  const BOX = Math.max(D.scene.half + 300, RMAX), dLat = BOX / mN, dLng = BOX / mE;
  const map = new Map3DElement({ center: { lat: lat0, lng: lon0, altitude: g0 }, range: 300, tilt: 60, heading: 0, mode: "SATELLITE", defaultUIHidden: true,
    bounds: { south: lat0 - dLat, north: lat0 + dLat, west: lon0 - dLng, east: lon0 + dLng }, maxAltitude: RMAX, maxTilt: 82 });
  map.style.cssText = "position:absolute;inset:0;width:100%;height:100%";
  host.appendChild(map);

  // scene frame: x east, y up, z south, origin at the house on the ground
  const enu = (c: LLA) => new THREE.Vector3((c.lng - lon0) * mE, c.altitude - g0, -(c.lat - lat0) * mN);
  const T = new THREE.Vector3(), P = new THREE.Vector3(), F = new THREE.Vector3();
  type Pose = { lat: number; lng: number; alt: number; range: number; tilt: number; heading: number };
  const put = (p: Pose) => Object.assign(map, { center: { lat: p.lat, lng: p.lng, altitude: p.alt }, range: p.range, tilt: p.tilt, heading: p.heading });
  let anim = 0, down = 0, still = 0, last = "";
  const STOP = ["pointerdown", "wheel", "touchstart", "keydown"], stop = () => { cancelAnimationFrame(anim); anim = 0; };
  const press = (e: PointerEvent) => { down = e.buttons ? 1 : 0; still = 0; };
  for (const e of STOP) host.addEventListener(e, stop, { capture: true, passive: true });
  host.addEventListener("pointerdown", press, { capture: true, passive: true });
  for (const e of ["pointerup", "pointercancel"] as const) addEventListener(e, press, { capture: true, passive: true });   // a release can land off the map
  // Google sometimes moves the pivot onto the camera itself (range 0, seen after resizes, tab switches and its own
  // gestures): drags then spin the view in place and the camera, not the ground, runs into `bounds` and sticks.
  // Once the view has settled, put the pivot back on the ground point at the middle of the view. The camera
  // doesn't move, so nothing visibly jumps.
  function repivot(P: THREE.Vector3, F: THREE.Vector3) {
    let s = F.y < -.05 ? -P.y / F.y : 300;                                         // to the house's ground level
    for (const k of ["x", "z"] as const) if (Math.abs(F[k]) > 1e-6) {              // stay inside bounds: Google would clamp (and jump) it
      const lim = ((F[k] > 0 ? BOX : -BOX) * .97 - P[k]) / F[k]; if (lim > 0) s = Math.min(s, lim);
    }
    s = THREE.MathUtils.clamp(s, 20, RMAX); const g = T.copy(P).addScaledVector(F, s);
    Object.assign(map, { center: { lat: lat0 - g.z / mN, lng: lon0 + g.x / mE, altitude: g0 + g.y }, range: s });
  }
  return {
    sync(cam: THREE.PerspectiveCamera) {
      const c = map.center; if (!c) return;
      if ((map.range ?? 0) > RMAX) map.range = RMAX;   // backstop: bounds/maxAltitude alone still let the wheel pull far back
      const t = (map.tilt ?? 0) * Math.PI / 180, h = (map.heading ?? 0) * Math.PI / 180, r = map.range ?? 300;
      // view direction from heading/tilt (three: x east, y up, z south); camera sits r behind the centre along it
      F.set(Math.sin(t) * Math.sin(h), -Math.cos(t), -Math.sin(t) * Math.cos(h));
      P.copy(enu(c)).addScaledVector(F, -r);
      const pose = `${c.lat},${c.lng},${c.altitude},${r},${map.tilt},${map.heading}`;
      still = pose === last ? still + 1 : 0; last = pose;
      if (r < 1 && !down && !anim && still > 8) { repivot(P, F); still = 0; }   // ~150 ms without motion (Google's glide has ended)
      cam.position.copy(P); cam.up.set(0, 1, 0); cam.lookAt(T.copy(P).add(F));
      const fov = map.fov ?? 35; if (cam.fov !== fov) { cam.fov = fov; cam.updateProjectionMatrix(); }
    },
    // fly to a three.js camera pose (position + target in scene coordinates) over `dur` s, eased; any user input on
    // the map cancels the flight. Stepped property by property rather than flyCameraTo: a Google flight ends with the
    // pivot on the camera itself (range 0), so drags would spin in place instead of orbiting the house.
    fly(pos: THREE.Vector3, target: THREE.Vector3, dur = 0) {
      cancelAnimationFrame(anim); anim = 0;
      const d = pos.clone().sub(target), r = d.length(), tilt = Math.acos(THREE.MathUtils.clamp(d.y / r, -1, 1)) * 180 / Math.PI;
      const heading = (Math.atan2(-d.x, d.z) * 180 / Math.PI + 360) % 360;   // look from the camera toward the target
      const end = { lat: lat0 - target.z / mN, lng: lon0 + target.x / mE, alt: g0 + target.y, range: r, tilt, heading };
      if (map.center && !((map.range ?? 0) > 1)) repivot(P, F);   // P, F: this frame's camera, from sync()
      const c = map.center;
      if (!(dur > 0) || !c) { put(end); return; }
      const a = { lat: c.lat, lng: c.lng, alt: c.altitude, range: map.range, tilt: map.tilt ?? 0, heading: map.heading ?? 0 };
      const dh = ((end.heading - a.heading) % 360 + 540) % 360 - 180, t0 = performance.now();
      const step = (now: number) => {
        const k = Math.min(1, (now - t0) / 1000 / dur), e = k < .5 ? 4 * k * k * k : 1 - (-2 * k + 2) ** 3 / 2, L = (x: number, y: number) => x + (y - x) * e;
        put({ lat: L(a.lat, end.lat), lng: L(a.lng, end.lng), alt: L(a.alt, end.alt), range: Math.exp(L(Math.log(a.range), Math.log(end.range))),
          tilt: L(a.tilt, end.tilt), heading: (a.heading + dh * e + 360) % 360 });
        anim = k < 1 ? requestAnimationFrame(step) : 0;
      };
      anim = requestAnimationFrame(step);
    },
    dispose() {
      cancelAnimationFrame(anim); for (const e of STOP) host.removeEventListener(e, stop, true);
      host.removeEventListener("pointerdown", press, true); for (const e of ["pointerup", "pointercancel"] as const) removeEventListener(e, press, true); map.remove();
    },
  };
}
