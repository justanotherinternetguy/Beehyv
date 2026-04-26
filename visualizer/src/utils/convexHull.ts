export type Vec2 = [number, number];

// Graham scan convex hull
export function convexHull(points: Vec2[]): Vec2[] {
  if (points.length < 3) return points;

  // Find bottom-most (then left-most) point
  let pivot = 0;
  for (let i = 1; i < points.length; i++) {
    if (
      points[i][1] < points[pivot][1] ||
      (points[i][1] === points[pivot][1] && points[i][0] < points[pivot][0])
    ) {
      pivot = i;
    }
  }
  [points[0], points[pivot]] = [points[pivot], points[0]];

  const base = points[0];
  const rest = points.slice(1).sort((a, b) => {
    const angleA = Math.atan2(a[1] - base[1], a[0] - base[0]);
    const angleB = Math.atan2(b[1] - base[1], b[0] - base[0]);
    return angleA - angleB;
  });

  const stack: Vec2[] = [base, rest[0]];
  for (let i = 1; i < rest.length; i++) {
    while (stack.length > 1 && cross(stack[stack.length - 2], stack[stack.length - 1], rest[i]) <= 0) {
      stack.pop();
    }
    stack.push(rest[i]);
  }

  return stack;
}

function cross(O: Vec2, A: Vec2, B: Vec2): number {
  return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
}

// Expand hull outward by `padding` units
export function expandHull(hull: Vec2[], padding: number): Vec2[] {
  if (hull.length < 2) return hull;

  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;

  return hull.map(([x, y]) => {
    const dx = x - cx;
    const dy = y - cy;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    return [x + (dx / len) * padding, y + (dy / len) * padding] as Vec2;
  });
}

// Smooth hull points using Catmull-Rom → SVG path
export function hullToSmoothPath(hull: Vec2[]): string {
  if (hull.length < 2) return '';
  const pts = [...hull, hull[0], hull[1], hull[2] ?? hull[0]];
  let d = `M ${hull[0][0]} ${hull[0][1]}`;

  for (let i = 0; i < hull.length; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const p2 = pts[i + 2];
    const p3 = pts[(i + 3) % pts.length] ?? pts[0];

    const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6;

    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
  }

  return d + ' Z';
}

export function centroid(points: Vec2[]): Vec2 {
  const x = points.reduce((s, p) => s + p[0], 0) / points.length;
  const y = points.reduce((s, p) => s + p[1], 0) / points.length;
  return [x, y];
}