// The eye of the storm: two spiral bands winding in counter-clockwise (a northern-hemisphere storm) around the eye wall and a
// teal pupil. Each band's outer tail dissolves into halftone dots, the brand motif. The favicon in index.html is this same
// geometry, pre-computed; change both together.
const pt = (s: number, rot: number) => {
  const a = (rot - 40 - 165 * s) * Math.PI / 180, r = 14.2 - 3.8 * s;           // s: 0 = outer tail, 1 = head at the eye
  return [+(16 + r * Math.cos(a)).toFixed(2), +(16 + r * Math.sin(a)).toFixed(2)];
};
const BANDS = [0, 180].map(rot => ({
  d: "M" + Array.from({ length: 13 }, (_, i) => pt(.3 + .7 * i / 12, rot).join(" ")).join("L"),
  dots: [[0, .55], [.1, .8], [.2, 1.05]].map(([s, r]) => [...pt(s, rot), r]),
}));

export default function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true" className="logo">
      <path d={BANDS.map(b => b.d).join("")} stroke="var(--accent)" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
      <g fill="var(--accent)">{BANDS.flatMap(b => b.dots).map(([x, y, r], i) => <circle key={i} cx={x} cy={y} r={r} />)}</g>
      <circle cx="16" cy="16" r="5.3" stroke="var(--ink)" strokeWidth="2.3" />
      <circle cx="16" cy="16" r="2" fill="var(--accent)" />
    </svg>
  );
}

export const Brand = () => <span className="brand"><Logo size={18} /><span>StormVoyance</span></span>;
