export const CLUSTER_PALETTE = [
  "#4F6EF7",
  "#E8564A",
  "#2EC4B6",
  "#F5A623",
  "#9B5DE5",
  "#00B4D8",
  "#F72585",
  "#3A86FF",
  "#06D6A0",
  "#FB8500",
  "#8338EC",
  "#FF006E",
  "#3D405B",
  "#E07A5F",
  "#81B29A",
  "#F2CC8F",
  "#118AB2",
  "#073B4C",
  "#FFD166",
  "#EF476F",
];

export function clusterColor(id: number): string {
  if (id < 0) return "#C8C8D8";
  const hue = (id * 137.508) % 360;
  return `oklch(0.62 0.17 ${hue.toFixed(1)})`;
}

export function densityToRGB(
  density: number,
  clusterId: number,
  darkMode = false,
): [number, number, number] {
  if (clusterId < 0) {
    if (darkMode) {
      const v = 0.08 + density * 0.08; // very dim grey, max ~0.20
      return [v, v, v + 0.04];
    }
    const v = 0.55 + density * 0.1;
    return [v, v, v + 0.05];
  }

  const hex = CLUSTER_PALETTE[clusterId % CLUSTER_PALETTE.length]!;
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;

  if (darkMode) {
    // Blend toward near-black at low density; only the densest cores get color
    const t = 0.25 + density * 0.75; // max luminance contribution ~0.63 of hue
    const br = 0.02,
      bg_ = 0.02,
      bb = 0.04; // dark navy floor
    return [r * t + br * (1 - t), g * t + bg_ * (1 - t), b * t + bb * (1 - t)];
  }

  // Light mode (original)
  const t = 0.25 + density * 0.75;
  return [
    r * t + (1 - t) * 0.95,
    g * t + (1 - t) * 0.95,
    b * t + (1 - t) * 0.98,
  ];
}

export function hexToRGB(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
}
