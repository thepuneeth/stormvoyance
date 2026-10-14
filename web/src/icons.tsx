// One line-icon set for the whole app: 24x24 grid, 1.5 stroke, round caps and joins, currentColor.
// Each icon is a single path so they all share one optical weight. Keep new ones inside the 2..22 live area.
export type IconName = "wind" | "gust" | "rain" | "thunder" | "cloud" | "sun" | "moon" | "tree" | "house" |
  "power" | "flood" | "compass" | "pin" | "play" | "pause" | "close" | "arrow-right" | "arrow-left" | "layers" | "info" | "search" |
  "check" | "clock" | "bell" | "spark";

const ring = (cx: number, cy: number, r: number) => `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
const CLOUD_SM = "M15.8 14.5H6.8a3.5 3.5 0 1 1 .7-6.93A5 5 0 0 1 16.9 6.6a4 4 0 0 1-1.1 7.9Z";   // sits above rain / a bolt
const WAVE = (y: number) => `M3 ${y}q1.5-1.4 3 0t3 0 3 0 3 0 3 0 3 0`;

const P: Record<IconName, string> = {
  wind: "M3 8h9.5a2.5 2.5 0 1 0-2.5-2.5M3 12h15.5a2.5 2.5 0 1 1-2.5 2.5M3 16h7",
  gust: "M3 7.5h8M3 12h13M3 16.5h9M17 8l4 4-4 4",
  rain: `${CLOUD_SM}M8.5 17.5l-1 2.5M12.5 17.5l-1 2.5M16.5 17.5l-1 2.5`,
  thunder: `${CLOUD_SM}M12.5 16l-2.5 3h3.5l-2.5 3`,
  cloud: "M16.6 19H7.1a4.4 4.4 0 1 1 .9-8.7A5.9 5.9 0 0 1 19 9.8a4.6 4.6 0 0 1-2.4 9.2Z",
  sun: `${ring(12, 12, 4)}M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4`,
  moon: "M12.5 3.5a6.5 6.5 0 0 0 8 8 8.6 8.6 0 1 1-8-8Z",
  tree: "M8.4 16.5a4.2 4.2 0 0 1-1.6-8A5.3 5.3 0 0 1 17.2 8.5a4.2 4.2 0 0 1-1.6 8ZM12 21.5v-9M12 15.5l-2.5-2M12 14l2-1.6",
  house: "M3.5 11L12 4l8.5 7M6 9v11h12V9M10 20v-5h4v5",
  power: "M9 3v4M15 3v4M7 7h10v3.5a5 5 0 0 1-10 0ZM12 15.5V21",
  flood: `M6.5 11.5L12 7l5.5 4.5M8.5 10v3.5M15.5 10v3.5${WAVE(16.5)}${WAVE(20.5)}`,
  compass: `${ring(12, 12, 9)}M15.5 8.5l-2 5-5 2 2-5Z`,
  pin: `M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11Z${ring(12, 10, 2.3)}`,
  play: "M8 5.5v13l10.5-6.5Z",
  pause: "M8.5 5.5v13M15.5 5.5v13",
  close: "M6 6l12 12M18 6L6 18",
  "arrow-right": "M4.5 12h15M13.5 6l6 6-6 6",
  "arrow-left": "M19.5 12h-15M10.5 6l-6 6 6 6",
  layers: "M12 3.5l9 4.6-9 4.6-9-4.6ZM3 12.4l9 4.6 9-4.6M3 16.3l9 4.6 9-4.6",
  info: `${ring(12, 12, 9)}M12 11v5.5M12 7.8v.01`,
  search: `${ring(11, 11, 6.5)}M20 20l-4.4-4.4`,
  check: "M5 12.5l4.5 4.5L19 7.5",
  clock: `${ring(12, 12, 9)}M12 7v5l3.5 2`,
  bell: "M6 16.5V11a6 6 0 0 1 12 0v5.5l1.5 2h-15ZM10 20.5a2 2 0 0 0 4 0",
  spark: "M12 3c.7 5 3.9 8.3 9 9-5.1.7-8.3 4-9 9-.7-5-3.9-8.3-9-9 5.1-.7 8.3-4 9-9Z",
};

export function Icon({ name, size = 20, className }: { name: IconName; size?: number; className?: string }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" className={className}>
      <path d={P[name]} />
    </svg>
  );
}
