/**
 * Measures real rendered label dimensions via an off-screen Canvas 2D context.
 * Returns pixel sizes only — no normalisation needed since ClusterRings works
 * entirely in screen space.
 */

export interface LabelSizePx {
  width:  number;
  height: number;
}

/** Mirror of wrapLabel() in ClusterRings.tsx — must stay in sync. */
export function wrapLabel(text: string, maxChars = 16): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    if (!current) {
      current = word;
    } else if ((current + ' ' + word).length <= maxChars) {
      current += ' ' + word;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  return lines;
}

export interface MeasureOptions {
  font:       string;
  maxChars?:  number;
  paddingX?:  number;
  paddingY?:  number;
  fontSize?:  number;
  lineHeightMultiplier?: number;
}

export function measureLabels(labels: string[], opts: MeasureOptions): LabelSizePx[] {
  const {
    font,
    maxChars             = 16,
    paddingX             = 6,
    paddingY             = 4,
    fontSize             = 12,
    lineHeightMultiplier = 1.3,
  } = opts;

  const canvas = document.createElement('canvas');
  const ctx    = canvas.getContext('2d')!;
  ctx.font     = font;

  const testM        = ctx.measureText('Mg');
  const lineHeightPx = 'fontBoundingBoxAscent' in testM
    ? testM.fontBoundingBoxAscent + testM.fontBoundingBoxDescent
    : fontSize * lineHeightMultiplier;

  return labels.map(text => {
    const lines    = wrapLabel(text, maxChars);
    const maxWidth = Math.max(...lines.map(l => ctx.measureText(l).width));
    return {
      width:  maxWidth + paddingX * 2,
      height: lines.length * lineHeightPx + paddingY * 2,
    };
  });
}