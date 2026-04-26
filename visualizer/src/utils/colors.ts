// Warm light-mode galaxy palette: cream/linen background, vibrant cluster colors
export const CLUSTER_PALETTE = [
  '#4F6EF7', // cobalt blue
  '#E8564A', // coral red
  '#2EC4B6', // teal
  '#F5A623', // amber
  '#9B5DE5', // violet
  '#00B4D8', // sky
  '#F72585', // hot pink
  '#3A86FF', // azure
  '#06D6A0', // mint
  '#FB8500', // orange
  '#8338EC', // purple
  '#FF006E', // rose
  '#3D405B', // slate
  '#E07A5F', // terracotta
  '#81B29A', // sage
  '#F2CC8F', // sand
  '#118AB2', // cerulean
  '#073B4C', // dark teal
  '#FFD166', // yellow
  '#EF476F', // pink-red
];

export function clusterColor(id: number): string {
  if (id < 0) return '#C8C8D8';
  const hue = (id * 137.508) % 360;
  return `oklch(0.62 0.17 ${hue.toFixed(1)})`;
}

/**
 * Map (density, clusterId) → [r, g, b] in [0,1].
 *
 * Light mode: blend cluster hue toward white at low density.
 * Dark mode:  blend cluster hue toward a deep navy at low density,
 *             let high-density points glow brightly.
 */
export function densityToRGB(
  density: number,
  clusterId: number,
  darkMode = false,
): [number, number, number] {
  if (clusterId < 0) {
    // Noise points
    if (darkMode) {
      const v = 0.18 + density * 0.18;
      return [v, v, v + 0.04];
    }
    const v = 0.55 + density * 0.1;
    return [v, v, v + 0.05];
  }

  const hex = CLUSTER_PALETTE[clusterId % CLUSTER_PALETTE.length]!;
  const r   = parseInt(hex.slice(1, 3), 16) / 255;
  const g   = parseInt(hex.slice(3, 5), 16) / 255;
  const b   = parseInt(hex.slice(5, 7), 16) / 255;

  if (darkMode) {
    // Low density → deep navy-black; high density → vivid cluster hue
    const t = 0.15 + density * 0.85;
    // Background floor: very dark blue-grey
    const br = 0.04, bg_ = 0.05, bb = 0.09;
    return [
      r * t + br * (1 - t),
      g * t + bg_ * (1 - t),
      b * t + bb  * (1 - t),
    ];
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