// Lightweight DBSCAN for 2D points
// Uses a grid index for O(n) average performance instead of O(n²)

export interface Point2D {
  x: number;
  y: number;
  index: number;
}

export function dbscan(
  points: Point2D[],
  epsilon: number,
  minPts: number
): Int32Array {
  const n = points.length;
  const labels = new Int32Array(n).fill(-2); // -2 = unvisited, -1 = noise
  let clusterId = 0;

  // Grid index for fast neighbor lookup
  const cellSize = epsilon;
  const grid = new Map<string, number[]>();

  const cellKey = (cx: number, cy: number) => `${cx},${cy}`;

  for (let i = 0; i < n; i++) {
    const cx = Math.floor(points[i].x / cellSize);
    const cy = Math.floor(points[i].y / cellSize);
    const key = cellKey(cx, cy);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key)!.push(i);
  }

  function getNeighbors(idx: number): number[] {
    const px = points[idx].x;
    const py = points[idx].y;
    const cx = Math.floor(px / cellSize);
    const cy = Math.floor(py / cellSize);
    const eps2 = epsilon * epsilon;
    const neighbors: number[] = [];

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = grid.get(cellKey(cx + dx, cy + dy));
        if (!cell) continue;
        for (const j of cell) {
          if (j === idx) continue;
          const ddx = points[j].x - px;
          const ddy = points[j].y - py;
          if (ddx * ddx + ddy * ddy <= eps2) {
            neighbors.push(j);
          }
        }
      }
    }
    return neighbors;
  }

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;

    const neighbors = getNeighbors(i);

    if (neighbors.length < minPts) {
      labels[i] = -1; // noise
      continue;
    }

    labels[i] = clusterId;
    const queue = [...neighbors];

    while (queue.length > 0) {
      const q = queue.pop()!;
      if (labels[q] === -1) labels[q] = clusterId; // noise → border
      if (labels[q] !== -2) continue;
      labels[q] = clusterId;
      const qNeighbors = getNeighbors(q);
      if (qNeighbors.length >= minPts) {
        queue.push(...qNeighbors);
      }
    }

    clusterId++;
  }

  return labels;
}

// Compute density estimate for each point using grid-based neighbor count
export function computeDensity(points: Point2D[], radius: number): Float32Array {
  const n = points.length;
  const density = new Float32Array(n);
  const cellSize = radius;
  const grid = new Map<string, number[]>();

  for (let i = 0; i < n; i++) {
    const cx = Math.floor(points[i].x / cellSize);
    const cy = Math.floor(points[i].y / cellSize);
    const key = `${cx},${cy}`;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key)!.push(i);
  }

  const r2 = radius * radius;
  for (let i = 0; i < n; i++) {
    const px = points[i].x;
    const py = points[i].y;
    const cx = Math.floor(px / cellSize);
    const cy = Math.floor(py / cellSize);
    let count = 0;

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = grid.get(`${cx + dx},${cy + dy}`);
        if (!cell) continue;
        for (const j of cell) {
          const ddx = points[j].x - px;
          const ddy = points[j].y - py;
          if (ddx * ddx + ddy * ddy <= r2) count++;
        }
      }
    }
    density[i] = count;
  }

  // Normalize to [0,1]
  let maxD = 0;
  for (let i = 0; i < n; i++) if (density[i] > maxD) maxD = density[i];
  if (maxD > 0) for (let i = 0; i < n; i++) density[i] /= maxD;

  return density;
}