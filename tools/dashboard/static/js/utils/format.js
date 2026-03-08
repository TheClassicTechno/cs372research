export function fmt(v, decimals) {
  if (v == null) return '\u2014';
  return Number(v).toFixed(decimals != null ? decimals : 4);
}

export function fmtPct(v) {
  if (v == null) return '\u00b7';
  if (v === 0) return '\u00b7';
  return v.toFixed(2);
}

export function numFmt(n) {
  if (n == null) return '\u2014';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Return a CSS class encoding score magnitude as grayscale shading. */
export function scoreClass(v) {
  if (v == null) return '';
  if (v >= 0.80) return 'score-high';
  if (v >= 0.70) return 'score-mid';
  if (v >= 0.60) return 'score-low';
  return 'score-bad';
}
