/**
 * components/table.js
 *
 * Pure HTML builders for data tables: allocations, performance, debate impact, Sharpe, and collapse diagnostics.
 */

import { esc } from '../utils/dom.js';
import { fmtPct, numFmt } from '../utils/format.js';
import { T } from '../utils/labels.js';

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
  let profitCls = perf.profit >= 0 ? 'perf-profit' : 'perf-loss';
  let profitSign = perf.profit >= 0 ? '+' : '';
  let idAttr = id ? ' id="' + esc(id) + '"' : '';
  let cfg = T('agent_perf');
  let h = '<table class="data-table"' + idAttr + '>';
  h += '<tr><th>' + esc(label) + '</th><th>' + esc(cfg.columns[0]) + '</th></tr>';
  h += '<tr><td>' + esc(cfg.rows.initial_capital) + '</td><td>$' + numFmt(perf.initial_capital) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.final_value) + '</td><td class="' + profitCls + '">$' + numFmt(perf.final_value) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.profit_loss) + '</td><td class="' + profitCls + '">' + profitSign + '$' + numFmt(Math.abs(perf.profit)) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.return) + '</td><td class="' + profitCls + '">' + profitSign + perf.return_pct.toFixed(2) + '%</td></tr>';
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
  let seen = agentNames.reduce(function (acc, name) {
    let alloc = agents[name];
    if (alloc !== undefined && alloc !== null) {
      Object.keys(alloc).forEach(function (k) { acc[k] = true; });
    }
    return acc;
  }, {});
  return Object.keys(seen).sort(function (x, y) {
    let maxX = agentNames.reduce(function (mx, name) {
      let ag = agents[name];
      if (ag !== undefined && ag !== null && ag[x] !== undefined && ag[x] > mx) return ag[x];
      return mx;
    }, 0);
    let maxY = agentNames.reduce(function (my, name) {
      let ag = agents[name];
      if (ag !== undefined && ag !== null && ag[y] !== undefined && ag[y] > my) return ag[y];
      return my;
    }, 0);
    return maxY - maxX;
  });
}

/**
 * Build a multi-agent allocation table for a specific round phase.
 *
 * @param {object} agents      - {role: {ticker: weight}}
 * @param {string[]} agentNames - Sorted agent role keys
 * @param {function} agentLabel - Maps role to display label
 * @param {string} [testId]     - Optional data-testid attribute
 * @returns {string} HTML table string
 */
export function buildRoundAllocTable(agents, agentNames, agentLabel, testId) {
  let tickers = collectSortedTickers(agents, agentNames);
  let tid = testId ? ' data-testid="' + esc(testId) + '"' : '';
  let h = '<table class="data-table"' + tid + '>';
  h += '<tr><th></th>';
  for (const agent of agentNames) {
    h += '<th>' + esc(agentLabel(agent).toUpperCase()) + '</th>';
  }
  h += '</tr>';
  for (const ticker of tickers) {
    h += '<tr><td style="font-weight:600;">' + esc(ticker) + '</td>';
    for (const agent of agentNames) {
      let w = agents[agent] ? agents[agent][ticker] : null;
      h += '<td style="font-weight:600;text-align:right;">' + fmtPct(w) + '</td>';
    }
    h += '</tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Format portfolio value from server-provided phase data.
 *
 * @param {object|null} phase - {pv, delta_pct} or null
 * @returns {string} Formatted PV string
 */
function fmtPV(phase) {
  if (!phase) return '\u2014';
  return '$' + numFmt(phase.pv);
}

/**
 * Format delta % from server-provided value. No client-side arithmetic.
 *
 * @param {object|null} phase - {pv, delta_pct} or null
 * @returns {string} HTML span with color class
 */
function fmtDelta(phase) {
  if (!phase) return '\u2014';
  let d = phase.delta_pct;
  let cls = d >= 0 ? 'perf-profit' : 'perf-loss';
  let sign = d >= 0 ? '+' : '';
  return '<span class="' + cls + '">' + sign + d.toFixed(2) + '%</span>';
}

/**
 * Build the header rows for the debate impact table.
 *
 * @returns {string} HTML string with two header rows
 */
function buildImpactHeader() {
  let cfg = T('debate_impact_agents');
  let bdr = ' style="border-left:2px solid #d6c4a1"';
  let groupsExceptLast = cfg.groups.slice(0, -1);
  let h = '<tr><th rowspan="2">' + esc(cfg.sub_columns[0]).replace(esc(cfg.sub_columns[0]), 'Agent') + '</th>';
  for (const group of groupsExceptLast) {
    h += '<th colspan="2"' + bdr + '>' + esc(group) + '</th>';
  }
  h += '<th' + bdr + '>' + esc(cfg.groups[cfg.groups.length - 1]) + '</th></tr>';
  h += '<tr>';
  groupsExceptLast.forEach(function () {
    h += '<th>' + esc(cfg.sub_columns[0]) + '</th><th>' + esc(cfg.sub_columns[1]) + '</th>';
  });
  h += '<th>' + esc(cfg.sub_columns[1]) + '</th></tr>';
  return h;
}

/**
 * Build one agent row for the debate impact table.
 *
 * @param {object} d - Agent delta data
 * @param {string} label - Agent display label
 * @returns {string} HTML tr string
 */
function buildImpactRow(d, label) {
  let h = '<tr><td style="font-weight:600;">' + esc(label) + '</td>';
  let phases = ['r1_proposal', 'r1_revision', 'r1_js', 'r2_revision', 'r2_js'];
  for (const phase of phases) {
    let ph = d[phase];
    h += '<td style="text-align:right;">' + fmtPV(ph) + '</td>';
    h += '<td style="text-align:right;">' + fmtDelta(ph) + '</td>';
  }
  if (d.judge) {
    let jd = d.judge.vs_agent_delta_pct;
    let jcls = jd >= 0 ? 'perf-profit' : 'perf-loss';
    let jsign = jd >= 0 ? '+' : '';
    h += '<td class="' + jcls + '" style="text-align:right;">' + jsign + jd.toFixed(2) + '%</td>';
  } else {
    h += '<td style="text-align:right;">\u2014</td>';
  }
  h += '</tr>';
  return h;
}

/**
 * Build extended per-agent debate impact table with PV and delta columns.
 *
 * @param {object} agentDeltas - {role: {r1_proposal, r1_revision, r1_js, ...}}
 * @param {function} agentLabel - Maps role to display label
 * @returns {string} HTML table string
 */
export function buildDebateImpactTable(agentDeltas, agentLabel) {
  let roles = Object.keys(agentDeltas).sort();
  let h = '<table class="data-table" data-testid="debate-impact-agents">';
  h += buildImpactHeader();
  for (const role of roles) {
    h += buildImpactRow(agentDeltas[role], agentLabel(role).toUpperCase());
  }
  h += '</table>';
  return h;
}

/**
 * Build a comparison table for mean portfolio returns for a single round.
 *
 * @param {object} proposals - perf object with return_pct
 * @param {object} revisions - perf object with return_pct
 * @param {string} label - round label (e.g. "R1", "R2")
 * @param {string} testId - data-testid for the table
 * @param {object} [jsIntervention] - optional JS intervention perf
 * @returns {string} HTML table string
 */
export function buildMeanPortfolioTable(proposals, revisions, label, testId, jsIntervention) {
  if (!proposals || !revisions) return '';
  let delta = round2(revisions.return_pct - proposals.return_pct);
  let cls = delta >= 0 ? 'perf-profit' : 'perf-loss';
  let sign = delta >= 0 ? '+' : '';
  let cfg = T('mean_portfolio');
  let h = '<table class="data-table" data-testid="' + esc(testId) + '">';
  h += '<tr><th>' + esc(label) + ' ' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  h += '<tr><td>' + esc(cfg.rows.proposals) + '</td><td style="text-align:right;">';
  h += formatReturnCell(proposals.return_pct) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.revisions) + '</td><td style="text-align:right;">';
  h += formatReturnCell(revisions.return_pct) + '</td></tr>';
  if (jsIntervention) {
    h += '<tr><td>' + esc(cfg.rows.js_intervention) + '</td><td style="text-align:right;">';
    h += formatReturnCell(jsIntervention.return_pct) + '</td></tr>';
  }
  h += '<tr><td style="font-weight:600;">' + esc(cfg.rows.critique_impact) + '</td>';
  h += '<td class="' + cls + '" style="text-align:right;font-weight:600;">';
  h += sign + delta.toFixed(2) + '%</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Format a return percentage with sign and color class.
 *
 * @param {number} pct - Return percentage
 * @returns {string} Formatted HTML span
 */
function formatReturnCell(pct) {
  let cls = pct >= 0 ? 'perf-profit' : 'perf-loss';
  let sign = pct >= 0 ? '+' : '';
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
 *
 * @param {object} portfolio - {ticker: weight}
 * @returns {string} HTML table string
 */
export function buildSimpleAllocTable(portfolio) {
  let sorted = Object.entries(portfolio)
    .sort(function (a, b) { return b[1] - a[1]; });
  let cfg = T('simple_alloc');
  let h = '<table class="data-table" id="judge-alloc-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  for (const entry of sorted) {
    h += '<tr><td style="font-weight:600;">' + esc(entry[0]) + '</td>';
    h += '<td style="font-weight:600;">' + fmtPct(entry[1]) + '</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Format a Sharpe ratio value for display.
 *
 * @param {number|null} val - Sharpe value or null
 * @returns {string} Formatted string
 */
function fmtSharpe(val) {
  if (val === null || val === undefined) return '\u2014';
  return val.toFixed(2);
}

/**
 * Build Sharpe ratio table with one row per debate phase.
 *
 * @param {object} sharpe - {r1_proposal, r1_revision, r1_js, r2_revision, r2_js}
 * @returns {string} HTML table string
 */
export function buildSharpeTable(sharpe) {
  if (!sharpe) return '';
  let cfg = T('sharpe');
  let phaseKeys = ['r1_proposal', 'r1_revision', 'r1_js', 'r2_revision', 'r2_js'];
  let h = '<table class="data-table" data-testid="debate-impact-sharpe">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  for (const key of phaseKeys) {
    let val = sharpe[key];
    h += '<tr><td>' + esc(cfg.rows[key]) + '</td>';
    h += '<td style="text-align:right;">' + fmtSharpe(val) + '</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build collapse diagnostics agent rows for one round.
 *
 * @param {object} agents - {role: {movement, toward_consensus, collapse_share, dissent}}
 * @param {function} agentLabel - Maps role to display label
 * @returns {string} HTML table rows
 */
function buildCollapseRows(agents, agentLabel) {
  let roles = Object.keys(agents).sort();
  let h = '';
  for (const role of roles) {
    let a = agents[role];
    h += '<tr><td style="font-weight:600;">' + esc(agentLabel(role).toUpperCase()) + '</td>';
    h += '<td style="text-align:right;">' + a.movement.toFixed(4) + '</td>';
    h += '<td style="text-align:right;">' + a.toward_consensus.toFixed(4) + '</td>';
    h += '<td style="text-align:right;">';
    h += a.collapse_share !== null ? a.collapse_share.toFixed(4) : '\u2014';
    h += '</td>';
    h += '<td style="text-align:right;">' + a.dissent.toFixed(4) + '</td></tr>';
  }
  return h;
}

/**
 * Build a collapse diagnostics table for one round.
 *
 * @param {object} roundData - {round, agents, collapse_leader, collapse_index}
 * @param {function} agentLabel - Maps role to display label
 * @returns {string} HTML string
 */
export function buildCollapseTable(roundData, agentLabel) {
  let cfg = T('collapse');
  let h = '<div class="ov-subtitle" style="margin-top:8px;">Round ' + roundData.round + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (const col of cfg.columns) {
    h += '<th>' + esc(col) + '</th>';
  }
  h += '</tr>';
  h += buildCollapseRows(roundData.agents, agentLabel);
  h += '</table>';
  if (roundData.collapse_leader) {
    let ci = roundData.collapse_index;
    let ciCls = '';
    if (ci < 0.3) ciCls = 'perf-profit';
    else if (ci > 0.6) ciCls = 'perf-loss';
    h += '<div style="font-size:0.85em;margin-bottom:8px;">';
    h += esc(cfg.collapse_leader) + ': <strong>' + esc(agentLabel(roundData.collapse_leader).toUpperCase()) + '</strong>';
    h += ' &mdash; ' + esc(cfg.collapse_index) + ': <span class="' + ciCls + '">' + ci.toFixed(4) + '</span>';
    h += '</div>';
  }
  return h;
}

/**
 * Build a single summary metric cell with label and color-coded value.
 *
 * @param {object} cell - {label, value, pct}
 * @returns {string} HTML string
 */
function buildSummaryCell(cell) {
  let h = '<div class="debate-summary-cell">';
  h += '<div class="debate-summary-label">' + esc(cell.label) + '</div>';
  h += '<div class="debate-summary-value">';
  if (cell.value === null || cell.value === undefined) {
    h += '\u2014';
  } else if (cell.pct) {
    let cls = cell.value >= 0 ? 'perf-profit' : 'perf-loss';
    let sign = cell.value >= 0 ? '+' : '';
    h += '<span class="' + cls + '">' + sign + cell.value.toFixed(2) + '%</span>';
  } else {
    h += cell.value.toFixed(2);
  }
  h += '</div></div>';
  return h;
}

/**
 * Build the Debate Performance Summary panel.
 * Horizontal grid of key outcome metrics with color-coded values.
 *
 * @param {object} summary - Server-provided summary metrics
 * @returns {string} HTML string
 */
export function buildDebateSummaryPanel(summary) {
  if (!summary) return '';
  let cfg = T('debate_summary');
  let cells = [
    { label: cfg.cells.mean_proposal_return, value: summary.mean_proposal_return, pct: true },
    { label: cfg.cells.final_debate_return, value: summary.final_debate_return, pct: true },
    { label: cfg.cells.debate_alpha, value: summary.debate_alpha, pct: true },
    { label: cfg.cells.judge_return, value: summary.judge_return, pct: true },
    { label: cfg.cells.agent_vs_judge, value: summary.agent_vs_judge, pct: true },
    { label: cfg.cells.final_sharpe, value: summary.final_sharpe, pct: false },
  ];
  let h = '<div class="debate-summary-grid" data-testid="debate-summary-panel">';
  h += cells.map(function (cell) { return buildSummaryCell(cell); }).join('');
  h += '</div>';
  return h;
}
