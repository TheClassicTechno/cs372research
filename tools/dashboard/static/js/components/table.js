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

/**
 * Collect and sort tickers across all agents by max weight descending.
 *
 * @param {object} agents      - {role: {ticker: weight}}
 * @param {string[]} agentNames - Agent role keys
 * @returns {string[]} Sorted ticker names
 */
function collectSortedTickers(agents, agentNames) {
  var seen = {};
  for (var a = 0; a < agentNames.length; a++) {
    var alloc = agents[agentNames[a]];
    if (alloc !== undefined && alloc !== null) {
      var keys = Object.keys(alloc);
      for (var k = 0; k < keys.length; k++) { seen[keys[k]] = true; }
    }
  }
  return Object.keys(seen).sort(function (x, y) {
    var maxX = 0; var maxY = 0;
    for (var i = 0; i < agentNames.length; i++) {
      var ag = agents[agentNames[i]];
      if (ag !== undefined && ag !== null) {
        var wx = ag[x] !== undefined ? ag[x] : 0;
        var wy = ag[y] !== undefined ? ag[y] : 0;
        if (wx > maxX) { maxX = wx; }
        if (wy > maxY) { maxY = wy; }
      }
    }
    return maxY - maxX;
  });
}

/**
 * Build a multi-agent allocation table for a specific round phase.
 * Returns an HTML string with agent columns and ticker rows.
 *
 * @param {object} agents      - {role: {ticker: weight}}
 * @param {string[]} agentNames - Sorted agent role keys
 * @param {function} agentLabel - Maps role to display label
 * @param {string} [testId]     - Optional data-testid attribute
 * @returns {string} HTML table string
 */
export function buildRoundAllocTable(agents, agentNames, agentLabel, testId) {
  var tickers = collectSortedTickers(agents, agentNames);
  var tid = testId ? ' data-testid="' + esc(testId) + '"' : '';
  var h = '<table class="data-table"' + tid + '>';
  h += '<tr><th></th>';
  for (var a = 0; a < agentNames.length; a++) {
    h += '<th>' + esc(agentLabel(agentNames[a]).toUpperCase()) + '</th>';
  }
  h += '</tr>';
  for (var t = 0; t < tickers.length; t++) {
    var ticker = tickers[t];
    h += '<tr><td style="font-weight:600;">' + esc(ticker) + '</td>';
    for (var a = 0; a < agentNames.length; a++) {
      var w = agents[agentNames[a]] ? agents[agentNames[a]][ticker] : null;
      h += '<td style="font-weight:600;text-align:right;">' + fmtPct(w) + '</td>';
    }
    h += '</tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build a table showing per-agent debate impact (R1 proposal vs final revision).
 * Columns: Agent, Initial Return, Final Return, Delta $, Delta %.
 *
 * @param {object} agentDeltas - {role: {initial, final, delta_dollars, delta_pct}}
 * @param {function} agentLabel - Maps role to display label
 * @returns {string} HTML table string
 */
export function buildDebateImpactTable(agentDeltas, agentLabel) {
  var roles = Object.keys(agentDeltas).sort();
  var h = '<table class="data-table" data-testid="debate-impact-agents">';
  h += '<tr><th>Agent</th><th>R1 Proposal</th><th>Final Revision</th>';
  h += '<th>\u0394 $</th><th>\u0394 %</th></tr>';
  for (var i = 0; i < roles.length; i++) {
    var d = agentDeltas[roles[i]];
    if (!d.initial || !d.final) continue;
    var cls = d.delta_dollars >= 0 ? 'perf-profit' : 'perf-loss';
    var sign = d.delta_dollars >= 0 ? '+' : '';
    h += '<tr><td style="font-weight:600;">' + esc(agentLabel(roles[i]).toUpperCase()) + '</td>';
    h += '<td style="text-align:right;">' + d.initial.return_pct.toFixed(2) + '%</td>';
    h += '<td style="text-align:right;">' + d.final.return_pct.toFixed(2) + '%</td>';
    h += '<td class="' + cls + '" style="text-align:right;">' + sign + '$' + numFmt(Math.abs(d.delta_dollars)) + '</td>';
    h += '<td class="' + cls + '" style="text-align:right;">' + sign + d.delta_pct.toFixed(2) + '%</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build a comparison table for mean portfolio returns for a single round.
 * Shows the equal-weight mean portfolio performance side by side.
 *
 * @param {object} proposals - perf object with return_pct
 * @param {object} revisions - perf object with return_pct
 * @param {string} label - round label (e.g. "R1", "R2")
 * @param {string} testId - data-testid for the table
 * @returns {string} HTML table string
 */
export function buildMeanPortfolioTable(proposals, revisions, label, testId) {
  if (!proposals || !revisions) return '';
  var delta = round2(revisions.return_pct - proposals.return_pct);
  var cls = delta >= 0 ? 'perf-profit' : 'perf-loss';
  var sign = delta >= 0 ? '+' : '';
  var h = '<table class="data-table" data-testid="' + esc(testId) + '">';
  h += '<tr><th>' + esc(label) + ' Mean Portfolio</th><th>Return</th></tr>';
  h += '<tr><td>Proposals (avg)</td><td style="text-align:right;">';
  h += formatReturnCell(proposals.return_pct) + '</td></tr>';
  h += '<tr><td>Revisions (avg)</td><td style="text-align:right;">';
  h += formatReturnCell(revisions.return_pct) + '</td></tr>';
  h += '<tr><td style="font-weight:600;">Critique Impact</td>';
  h += '<td class="' + cls + '" style="text-align:right;font-weight:600;">';
  h += sign + delta.toFixed(2) + '%</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Format a return percentage with sign and color class.
 *
 * @param {number} pct - Return percentage
 * @returns {string} Formatted return string
 */
function formatReturnCell(pct) {
  var cls = pct >= 0 ? 'perf-profit' : 'perf-loss';
  var sign = pct >= 0 ? '+' : '';
  return '<span class="' + cls + '">' + sign + pct.toFixed(2) + '%</span>';
}

/**
 * Round a number to 2 decimal places.
 *
 * @param {number} n - Number to round
 * @returns {number} Rounded number
 */
function round2(n) {
  return Math.round(n * 100) / 100;
}

/**
 * Build a simple single-column allocation table (fallback).
 * Returns an HTML string with Ticker/JUDGE columns.
 *
 * @param {object} portfolio - {ticker: weight}
 * @returns {string} HTML table string
 */
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
