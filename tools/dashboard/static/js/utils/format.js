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
