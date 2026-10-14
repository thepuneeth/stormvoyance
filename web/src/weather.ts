// Plain-language weather from the bundle's hourly forecast. Every word here is derived from the numbers; nothing is invented.
// Hour shape (sim.hours[i]): t ISO time, gust [p5,p25,p50,p75,p95] mph, rain [p50,p90] in/hr, pop %, thunder %.
import type { IconName } from "./icons";

export type Cond = { key: "thunder" | "rain" | "showers" | "windy" | "breezy" | "calm"; label: string; icon: IconName };
type Wx = { gust: number[]; rain: number[]; pop?: number; thunder?: number };
type WxHour = Wx & { t: string };

const C: Record<Cond["key"], Cond> = {
  thunder: { key: "thunder", label: "Thunderstorms", icon: "thunder" },
  rain: { key: "rain", label: "Rain", icon: "rain" },
  showers: { key: "showers", label: "Showers", icon: "rain" },
  windy: { key: "windy", label: "Windy", icon: "gust" },
  breezy: { key: "breezy", label: "Breezy", icon: "wind" },
  calm: { key: "calm", label: "Calm", icon: "cloud" },   // no sky-cover data, so no sun/moon claim
};

export function condition(h: Wx): Cond {
  if ((h.thunder ?? 0) >= 30) return C.thunder;
  if (h.rain[0] >= .1) return C.rain;
  if (h.rain[0] >= .02 || (h.pop ?? 0) >= 50) return C.showers;
  if (h.gust[2] >= 25) return C.windy;    // NWS "windy" / "breezy" bands, on the gust
  if (h.gust[2] >= 12) return C.breezy;
  return C.calm;
}

// "9 PM" in the house's time zone (bundle meta.tz). A bad zone falls back to the viewer's clock rather than throwing mid-render.
export function fmtHour(t: string, tz?: string): string {
  const d = new Date(t);
  try { return new Intl.DateTimeFormat("en-US", { hour: "numeric", timeZone: tz }).format(d); }
  catch { return new Intl.DateTimeFormat("en-US", { hour: "numeric" }).format(d); }
}

// One sentence about the next 6 hours from startIdx, most serious thing first.
// tz is optional (pass meta.tz); without it times are in the viewer's zone.
// "now" only when the window starts in the current hour; a selected future hour is named by its time.
export function sentence(hours: WxHour[], startIdx: number, tz?: string, now = Date.now()): string {
  const w = hours.slice(Math.max(0, startIdx), Math.max(0, startIdx) + 6);
  if (!w.length) return "";
  const cur = (i: number) => Math.abs(now - Date.parse(w[i].t) - 1.8e6) <= 1.8e6;   // inside hour i
  const at = (i: number) => cur(i) ? "now" : `around ${fmtHour(w[i].t, tz)}`;
  const first = (f: (h: WxHour) => boolean) => w.findIndex(f);
  const pk = w.reduce((a, h, i) => h.gust[2] > w[a].gust[2] ? i : a, 0), g = Math.round(w[pk].gust[2]);
  const gusts = g >= 12 ? `, with gusts to ${g} mph` : "";
  let i: number;
  if ((i = first(h => (h.thunder ?? 0) >= 30)) >= 0) return `Thunderstorms ${(w[i].thunder ?? 0) >= 60 ? "likely" : "possible"} ${at(i)}${gusts}`;
  if ((i = first(h => h.rain[0] >= .1)) >= 0) return `Rain ${at(i)}${gusts}`;
  if (g >= 25) return `Gusts up to ${g} mph ${at(pk)}`;
  if ((i = first(h => h.rain[0] >= .02 || (h.pop ?? 0) >= 50)) >= 0) return `Showers possible ${at(i)}${gusts}`;
  if (g >= 12) return `Breezy, with gusts near ${g} mph ${at(pk)}`;
  const span = w.length === 1 ? "the next hour" : `the next ${w.length} hours`;
  return `Gusts stay under ${Math.floor(w[pk].gust[2] / 5) * 5 + 5} mph for ${span}`;
}

// Not wired into the UI. Throws on the first broken threshold.
export function weatherSelfTest(): true {
  const H = (g: number, r = 0, pop = 0, thunder = 0, t = "2026-10-10T22:00:00Z"): WxHour => ({ t, gust: [0, 0, g, 0, 0], rain: [r, r], pop, thunder });
  const cases: [WxHour, Cond["key"]][] = [
    [H(5, 0, 0, 30), "thunder"], [H(5, .5, 90, 29.9), "rain"], [H(5, .1), "rain"], [H(5, .099), "showers"], [H(5, .02), "showers"],
    [H(40, 0, 50), "showers"], [H(25, .019, 49), "windy"], [H(24.9), "breezy"], [H(12), "breezy"], [H(11.9), "calm"],
  ];
  for (const [h, k] of cases) if (condition(h).key !== k) throw new Error(`condition(${JSON.stringify(h)}) = ${condition(h).key}, want ${k}`);
  const day = (f: (i: number) => WxHour) => Array.from({ length: 8 }, (_, i) => ({ ...f(i), t: `2026-10-11T${String(i).padStart(2, "0")}:00:00Z` }));
  const ok = (s: string, want: RegExp) => { if (!want.test(s)) throw new Error(`sentence "${s}" !~ ${want}`); };
  ok(sentence(day(() => H(9)), 0, "UTC"), /^Gusts stay under 10 mph for the next 6 hours$/);
  ok(sentence(day(i => H(9, 0, 0, i === 3 ? 65 : 0)), 0, "UTC"), /^Thunderstorms likely around 3 AM$/);
  ok(sentence(day(i => H(i === 2 ? 44 : 20, i ? 0 : .2)), 0, "UTC", Date.parse("2026-10-11T00:20:00Z")), /^Rain now, with gusts to 44 mph$/);
  ok(sentence(day(i => H(i === 2 ? 44 : 20, i ? 0 : .2)), 0, "UTC", Date.parse("2026-10-10T20:00:00Z")), /^Rain around 12 AM, with gusts to 44 mph$/);
  ok(sentence(day(() => H(9)), 7, "UTC"), /the next hour$/);
  ok(sentence(day(() => H(12)), 99, "UTC"), /^$/);
  if (fmtHour("2026-10-11T02:00:00Z", "America/Chicago") !== "9 PM") throw new Error("fmtHour tz");
  return true;
}
