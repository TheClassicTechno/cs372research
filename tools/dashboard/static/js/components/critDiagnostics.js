/**
 * components/critDiagnostics.js
 *
 * Pure HTML builder for the CRIT Reasoning Diagnostics section.
 * Renders three tables: pillar score statistics, diagnostic failure rates,
 * and diagnostic counts, grouped by agent and condition.
 */

import { esc } from '../utils/dom.js';
import { fmt } from '../utils/format.js';

/**
 * CSS class for a diagnostic failure percentage.
 * Green = low failure rate, yellow = moderate, red = high.
 */
function failRateClass(pct) {
  if (pct <= 5) return 'perf-profit';
  if (pct <= 20) return 'score-mid';
  if (pct <= 50) return 'score-high';
  return 'perf-loss';
}

/**
 * CSS class for a pillar score value.
 * Green >= 0.85, gold >= 0.75, gray < 0.75.
 */
function pillarClass(val) {
  if (val >= 0.85) return 'perf-profit';
  if (val >= 0.75) return 'score-high';
  return 'score-mid';
}

/**
 * Build the pillar score statistics table (Table 1).
 * Accepts the pillar_stats array from the API.
 * Returns an HTML string.
 */
function buildPillarTable(pillarStats, conditions) {
  if (!Array.isArray(pillarStats) || pillarStats.length === 0) return '';

  let h = '<div class="section-label">Table 1 \u2014 Pillar Score Statistics</div>';
  h += '<table class="data-table crit-diag-table" data-testid="crit-pillar-table">';
  h += '<thead><tr>';
  h += '<th>Agent</th><th>Pillar</th>';
  for (let c = 0; c < conditions.length; c++) {
    h += '<th class="num-col">' + esc(conditions[c]) + '</th>';
  }
  h += '</tr></thead><tbody>';

  // Group by agent, then pillar
  let currentAgent = '';
  for (let i = 0; i < pillarStats.length; i++) {
    let row = pillarStats[i];
    if (row.condition !== conditions[0]) continue;

    // Find matching row for condition B
    let rowB = null;
    for (let j = 0; j < pillarStats.length; j++) {
      if (pillarStats[j].agent === row.agent &&
          pillarStats[j].pillar_key === row.pillar_key &&
          pillarStats[j].condition === conditions[1]) {
        rowB = pillarStats[j];
        break;
      }
    }

    let agentLabel = '';
    if (row.agent !== currentAgent) {
      currentAgent = row.agent;
      agentLabel = row.agent;
    }

    h += '<tr>';
    h += '<td>' + esc(agentLabel) + '</td>';
    h += '<td>' + esc(row.pillar) + '</td>';
    h += '<td class="num-cell ' + pillarClass(row.mean) + '">'
      + fmt(row.mean, 3) + '\u00b1' + fmt(row.stdev, 3) + '</td>';
    if (rowB !== null) {
      h += '<td class="num-cell ' + pillarClass(rowB.mean) + '">'
        + fmt(rowB.mean, 3) + '\u00b1' + fmt(rowB.stdev, 3) + '</td>';
    } else {
      h += '<td class="num-cell">\u2014</td>';
    }
    h += '</tr>';
  }

  h += '</tbody></table>';
  return h;
}

/**
 * Build the diagnostic failure rates table (Table 2).
 * Accepts the flag_stats array from the API.
 * Returns an HTML string.
 */
function buildFlagTable(flagStats, conditions) {
  if (!Array.isArray(flagStats) || flagStats.length === 0) return '';

  let h = '<div class="section-label">Table 2 \u2014 Diagnostic Failure Rates</div>';
  h += '<table class="data-table crit-diag-table" data-testid="crit-flag-table">';
  h += '<thead><tr>';
  h += '<th>Agent</th><th>Diagnostic</th>';
  for (let c = 0; c < conditions.length; c++) {
    h += '<th class="num-col">' + esc(conditions[c]) + '</th>';
  }
  h += '</tr></thead><tbody>';

  let currentAgent = '';
  for (let i = 0; i < flagStats.length; i++) {
    let row = flagStats[i];
    if (row.condition !== conditions[0]) continue;

    let rowB = null;
    for (let j = 0; j < flagStats.length; j++) {
      if (flagStats[j].agent === row.agent &&
          flagStats[j].flag === row.flag &&
          flagStats[j].condition === conditions[1]) {
        rowB = flagStats[j];
        break;
      }
    }

    let agentLabel = '';
    if (row.agent !== currentAgent) {
      currentAgent = row.agent;
      agentLabel = row.agent;
    }

    let flagLabel = row.flag.replace(/_/g, ' ');

    h += '<tr>';
    h += '<td>' + esc(agentLabel) + '</td>';
    h += '<td>' + esc(flagLabel) + '</td>';
    h += '<td class="num-cell ' + failRateClass(row.pct) + '">'
      + fmt(row.pct, 1) + '% (' + esc(String(row.count)) + '/' + esc(String(row.total)) + ')</td>';
    if (rowB !== null) {
      h += '<td class="num-cell ' + failRateClass(rowB.pct) + '">'
        + fmt(rowB.pct, 1) + '% (' + esc(String(rowB.count)) + '/' + esc(String(rowB.total)) + ')</td>';
    } else {
      h += '<td class="num-cell">\u2014</td>';
    }
    h += '</tr>';
  }

  h += '</tbody></table>';
  return h;
}

/**
 * Build the diagnostic counts table (Table 3).
 * Accepts the count_stats array from the API.
 * Returns an HTML string.
 */
function buildCountTable(countStats, conditions) {
  if (!Array.isArray(countStats) || countStats.length === 0) return '';

  let h = '<div class="section-label">Table 3 \u2014 Diagnostic Counts (Mean per Run)</div>';
  h += '<table class="data-table crit-diag-table" data-testid="crit-count-table">';
  h += '<thead><tr>';
  h += '<th>Agent</th><th>Diagnostic</th>';
  for (let c = 0; c < conditions.length; c++) {
    h += '<th class="num-col">' + esc(conditions[c]) + '</th>';
  }
  h += '</tr></thead><tbody>';

  let currentAgent = '';
  for (let i = 0; i < countStats.length; i++) {
    let row = countStats[i];
    if (row.condition !== conditions[0]) continue;

    let rowB = null;
    for (let j = 0; j < countStats.length; j++) {
      if (countStats[j].agent === row.agent &&
          countStats[j].count_key === row.count_key &&
          countStats[j].condition === conditions[1]) {
        rowB = countStats[j];
        break;
      }
    }

    let agentLabel = '';
    if (row.agent !== currentAgent) {
      currentAgent = row.agent;
      agentLabel = row.agent;
    }

    let countLabel = row.count_key.replace(/_/g, ' ');

    h += '<tr>';
    h += '<td>' + esc(agentLabel) + '</td>';
    h += '<td>' + esc(countLabel) + '</td>';
    h += '<td class="num-cell">' + fmt(row.mean, 2)
      + ' <span style="color:#999;">(' + esc(String(row.total_instances)) + ' total)</span></td>';
    if (rowB !== null) {
      h += '<td class="num-cell">' + fmt(rowB.mean, 2)
        + ' <span style="color:#999;">(' + esc(String(rowB.total_instances)) + ' total)</span></td>';
    } else {
      h += '<td class="num-cell">\u2014</td>';
    }
    h += '</tr>';
  }

  h += '</tbody></table>';
  return h;
}

/**
 * Build the full CRIT Reasoning Diagnostics section.
 * Accepts the diagnostics data object from the API.
 * Returns an HTML string with three tables.
 */
export function buildCritDiagnosticsSection(data) {
  if (data === undefined || data === null) return '';
  if (data.pending === true) {
    return '<div class="section-label">Reasoning Diagnostics (CRIT)</div>'
      + '<p class="status-incomplete">' + esc(data.message) + '</p>';
  }
  if (data.error !== undefined) {
    return '<div class="section-label">Reasoning Diagnostics (CRIT)</div>'
      + '<p>' + esc(data.error) + '</p>';
  }

  let conditions = data.conditions;
  if (!Array.isArray(conditions) || conditions.length === 0) return '';

  let h = '<div class="crit-diag-section" data-testid="crit-diagnostics-section">';
  h += '<div class="section-label" style="font-size:1.1em;">Reasoning Diagnostics (CRIT)</div>';
  h += '<div class="crit-diag-subtitle">'
    + esc(String(data.total_records)) + ' agent\u00d7run observations across '
    + esc(String(conditions.length)) + ' conditions</div>';

  h += buildPillarTable(data.pillar_stats, conditions);
  h += buildFlagTable(data.flag_stats, conditions);
  h += buildCountTable(data.count_stats, conditions);

  h += '</div>';
  return h;
}
