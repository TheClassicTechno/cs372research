import { esc } from '../utils/dom.js';
import { fmtPct } from '../utils/format.js';

export function buildSimpleAllocTable(portfolio) {
  var sorted = Object.entries(portfolio)
    .sort(function (a, b) { return b[1] - a[1]; });
  var h = '<table class="data-table" id="judge-alloc-table">';
  h += '<tr><th>Ticker</th><th>JUDGE</th></tr>';
  for (var i = 0; i < sorted.length; i++) {
    h += '<tr><td style="font-weight:600;">' + esc(sorted[i][0]) + '</td>';
    h += '<td style="font-weight:600;">' + fmtPct(sorted[i][1]) + '</td></tr>';
  }
  h += '</table>';
  return h;
}
