import { esc } from '../utils/dom.js';
import { fmtPct, numFmt } from '../utils/format.js';

/**
 * Build a single-agent performance metrics table.
 * Returns an HTML string with Metric/Value rows matching the Judge perf table style.
 *
 * @param {object} perf  - {initial_capital, final_value, profit, return_pct}
 * @param {string} label - Column heading (e.g. "VALUE_ENRICHED")
 * @param {string} [id]  - Optional HTML id for the table element
 * @returns {string} HTML table string
 */
export function buildAgentPerfTable(perf, label, id) {
  var profitCls = perf.profit >= 0 ? 'perf-profit' : 'perf-loss';
  var profitSign = perf.profit >= 0 ? '+' : '';
  var idAttr = id ? ' id="' + esc(id) + '"' : '';
  var h = '<table class="data-table"' + idAttr + '>';
  h += '<tr><th>' + esc(label) + '</th><th>Value</th></tr>';
  h += '<tr><td>Initial Capital</td><td>$' + numFmt(perf.initial_capital) + '</td></tr>';
  h += '<tr><td>Final Value</td><td class="' + profitCls + '">$' + numFmt(perf.final_value) + '</td></tr>';
  h += '<tr><td>Profit/Loss</td><td class="' + profitCls + '">' + profitSign + '$' + numFmt(Math.abs(perf.profit)) + '</td></tr>';
  h += '<tr><td>Return</td><td class="' + profitCls + '">' + profitSign + perf.return_pct.toFixed(2) + '%</td></tr>';
  h += '</table>';
  return h;
}

/** Build a simple single-column allocation table (fallback). */
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
