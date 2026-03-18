/**
 * utils/format.js
 *
 * Number and value formatting utilities for dashboard display.
 */

/** Format a numeric value to a fixed number of decimal places, defaulting to 4. */
export function fmt(v, decimals) {
  if (v == null) return '\u2014';
  return Number(v).toFixed(decimals != null ? decimals : 4);
}

/** Format a percentage value to 2 decimal places, returning a dot for zero or null. */
export function fmtPct(v) {
  if (v == null) return '\u00b7';
  if (v === 0) return '\u00b7';
  return v.toFixed(2);
}

/** Format a number with locale-aware thousand separators and 2 decimal places. */
export function numFmt(n) {
  if (n == null) return '\u2014';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/**
 * Format a duration in seconds as a human-readable string.
 * Returns "Xm Ys" for durations >= 60s, "Xs" for shorter.
 */
export function fmtDuration(s) {
  if (s == null) return '\u2014';
  let sec = Math.round(s);
  if (sec < 60) return sec + 's';
  let m = Math.floor(sec / 60);
  let rem = sec % 60;
  if (m >= 60) {
    let h = Math.floor(m / 60);
    m = m % 60;
    return h + 'h ' + m + 'm';
  }
  return m + 'm ' + rem + 's';
}

/** Return a CSS class for a p-value: green < 0.05, yellow < 0.10, red otherwise. */
export function pvalueClass(p) {
  if (p < 0.05) return 'perf-profit';
  if (p < 0.10) return 'score-mid';
  return 'perf-loss';
}

/** Format a p-value for display: "< 0.001" or 4-decimal. */
export function fmtPvalue(p) {
  if (p < 0.001) return '< 0.001';
  return fmt(p, 4);
}

/** Return a CSS class encoding score magnitude as grayscale shading. */
export function scoreClass(v) {
  if (v == null) return '';
  if (v >= 0.80) return 'score-high';
  if (v >= 0.70) return 'score-mid';
  if (v >= 0.60) return 'score-low';
  return 'score-bad';
}
