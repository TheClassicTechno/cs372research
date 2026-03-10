/**
 * components/ablation.js
 *
 * Pure HTML builders for ablation summary cards and metric tables.
 * Renders experiment-level aggregate statistics from ablation_summary.json.
 */

import { esc } from '../utils/dom.js';
import { fmt, scoreClass } from '../utils/format.js';

/**
 * Build a stat cell with optional score shading.
 * Returns an HTML <td> string.
 */
function statCell(value, decimals) {
  var cls = scoreClass(value);
  var extra = cls ? ' ' + cls : '';
  return '<td class="num-cell' + extra + '">' + fmt(value, decimals) + '</td>';
}

/**
 * Build a small stats row: mean +/- stdev.
 * Returns an HTML string like "0.7200 +/- 0.0600".
 */
function meanStdev(obj, decimals) {
  if (obj === undefined || obj === null) return '\u2014';
  return fmt(obj.mean, decimals) + ' \u00b1 ' + fmt(obj.stdev, decimals);
}

/**
 * Build the rho summary table for one experiment.
 * Accepts the experiment rho object.
 * Returns an HTML table string.
 */
function buildRhoTable(rho) {
  var h = '<div class="section-label">Rho (\u03c1)</div>';
  h += '<table class="data-table">';
  h += '<tr><th>Metric</th><th>Mean</th><th>StDev</th><th>Min</th><th>Max</th></tr>';
  var fr = rho.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>Final Round</td>';
    h += statCell(fr.mean, 4) + statCell(fr.stdev, 4) + statCell(fr.min, 4) + statCell(fr.max, 4);
    h += '</tr>';
  }
  var ar = rho.all_rounds;
  if (ar !== undefined && ar !== null) {
    h += '<tr><td>All Rounds</td>';
    h += statCell(ar.mean, 4) + statCell(ar.stdev, 4);
    h += '<td>\u2014</td><td>\u2014</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build the pillars summary table for one experiment.
 * Accepts the experiment pillars object.
 * Returns an HTML table string.
 */
function buildPillarsTable(pillars) {
  var h = '<div class="section-label">CRIT Pillars (Final Round)</div>';
  h += '<table class="data-table">';
  h += '<tr><th>Pillar</th><th>Mean</th><th>StDev</th></tr>';
  var names = Object.keys(pillars);
  for (var i = 0; i < names.length; i++) {
    var p = pillars[names[i]];
    var fr = p.final_round;
    if (fr === undefined || fr === null) continue;
    h += '<tr><td>' + esc(names[i]) + '</td>';
    h += statCell(fr.mean, 4) + statCell(fr.stdev, 4);
    h += '</tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build JS divergence summary for one experiment.
 * Accepts the experiment js_divergence object.
 * Returns an HTML string.
 */
function buildJsSection(js) {
  var h = '<div class="section-label">JS Divergence</div>';
  h += '<table class="data-table">';
  h += '<tr><th>Metric</th><th>Value</th></tr>';
  var fr = js.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>Final Round Mean \u00b1 StDev</td><td>' + meanStdev(fr, 4) + '</td></tr>';
  }
  var traj = js.trajectory;
  if (traj !== undefined && traj !== null) {
    h += '<tr><td>Mean Delta (first\u2192last)</td><td>' + fmt(traj.mean_delta, 4) + '</td></tr>';
    h += '<tr><td>% Decreased</td><td>' + fmt(traj.pct_decreased, 1) + '%</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build evidence overlap summary for one experiment.
 * Accepts the experiment evidence_overlap object.
 * Returns an HTML string.
 */
function buildEoSection(eo) {
  var h = '<div class="section-label">Evidence Overlap</div>';
  h += '<table class="data-table">';
  h += '<tr><th>Metric</th><th>Value</th></tr>';
  var fr = eo.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>Final Round Mean \u00b1 StDev</td><td>' + meanStdev(fr, 4) + '</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build PID section for one experiment.
 * Accepts the experiment pid object.
 * Returns an HTML string.
 */
/**
 * Format a distribution object as "key: NN.N%, key: NN.N%".
 * Returns an HTML string with labels.
 */
function fmtDistribution(dist) {
  var keys = Object.keys(dist);
  var parts = [];
  for (var i = 0; i < keys.length; i++) {
    parts.push(esc(keys[i]) + ':&nbsp;' + fmt(dist[keys[i]] * 100, 1) + '%');
  }
  return parts.join(' &nbsp;\u2022&nbsp; ');
}

function buildPidSection(pid) {
  var h = '<div class="section-label">PID</div>';
  h += '<table class="data-table">';
  h += '<tr><th>Metric</th><th>Value</th></tr>';
  h += '<tr><td>Beta Final Mean \u00b1 StDev</td><td>' + meanStdev(pid.beta_final, 4) + '</td></tr>';
  var qd = pid.quadrant_distribution;
  if (qd !== undefined && qd !== null) {
    h += '<tr><td>Quadrant Distribution</td><td>' + fmtDistribution(qd) + '</td></tr>';
  }
  var td = pid.tone_distribution;
  if (td !== undefined && td !== null) {
    h += '<tr><td>Tone Distribution</td><td>' + fmtDistribution(td) + '</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build collapse metrics section for one experiment.
 * Accepts the experiment collapse object.
 * Returns an HTML string.
 */
function buildCollapseSection(collapse) {
  var h = '<div class="section-label">Collapse Metrics</div>';
  h += '<table class="data-table">';
  h += '<tr><th>Threshold</th><th>Count</th></tr>';
  h += '<tr><td>JS &lt; 0.05</td><td>' + esc(String(collapse.js_lt_005)) + '</td></tr>';
  h += '<tr><td>JS &lt; 0.07</td><td>' + esc(String(collapse.js_lt_007)) + '</td></tr>';
  h += '<tr><td>JS &lt; 0.10</td><td>' + esc(String(collapse.js_lt_010)) + '</td></tr>';
  h += '<tr><td>High \u03c1 + Low JS</td><td>' + esc(String(collapse.high_rho_low_js));
  h += ' (' + fmt(collapse.pct_high_rho_low_js, 1) + '%)</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Build a breakdown sub-table (per-scenario or per-agent-config).
 * Accepts label, breakdowns object.
 * Returns an HTML card string (collapsed by default).
 */
function buildBreakdownTable(label, breakdowns) {
  if (breakdowns === undefined || breakdowns === null) return '';
  var keys = Object.keys(breakdowns);
  if (keys.length === 0) return '';

  keys.sort(function (a, b) { return rhoSortKey(breakdowns[b]) - rhoSortKey(breakdowns[a]); });

  var h = '<div class="section-label">' + esc(label) + '</div>';
  h += '<table class="data-table">';
  var rhoHeader = label === 'Per Agent Config' ? 'Final \u03c1' : 'Final \u03c1 Mean';
  var jsHeader = label === 'Per Agent Config' ? 'Final JS' : 'Final JS Mean';
  h += '<tr><th>Name</th><th>Runs</th><th>' + rhoHeader + '</th><th>' + jsHeader + '</th></tr>';
  for (var i = 0; i < keys.length; i++) {
    var b = breakdowns[keys[i]];
    var rhoMean = (b.rho && b.rho.final_round) ? fmt(b.rho.final_round.mean, 4) : '\u2014';
    var jsMean = (b.js_divergence && b.js_divergence.final_round) ? fmt(b.js_divergence.final_round.mean, 4) : '\u2014';
    h += '<tr><td>' + esc(keys[i]) + '</td>';
    h += '<td>' + esc(String(b.run_count)) + '</td>';
    h += '<td>' + rhoMean + '</td>';
    h += '<td>' + jsMean + '</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Wrap HTML content in a metrics-col div.
 * Returns an HTML string.
 */
function col(html) {
  return '<div class="metrics-col">' + html + '</div>';
}

/**
 * Build the inner body HTML for an experiment card.
 * Accepts the experiment data object.
 * Returns an HTML string with all metric sections arranged in rows.
 *
 * Row 1: Rho, CRIT Pillars, JS Divergence, Evidence Overlap (side-by-side)
 * Row 2: PID, Collapse Metrics (side-by-side)
 * Row 3: Per Scenario, Per Agent Config (side-by-side)
 */
function buildExperimentBody(data) {
  var body = '<div style="margin-bottom:4px"><strong>Runs:</strong> ' + esc(String(data.run_count));
  body += ' &nbsp; <strong>Model:</strong> ' + esc(data.model) + '</div>';

  // Row 1: Rho, Pillars, JS Divergence, Evidence Overlap
  var row1 = '';
  if (data.rho !== undefined && data.rho !== null) { row1 += col(buildRhoTable(data.rho)); }
  if (data.pillars !== undefined && data.pillars !== null) { row1 += col(buildPillarsTable(data.pillars)); }
  if (data.js_divergence !== undefined && data.js_divergence !== null) { row1 += col(buildJsSection(data.js_divergence)); }
  if (data.evidence_overlap !== undefined && data.evidence_overlap !== null) { row1 += col(buildEoSection(data.evidence_overlap)); }
  if (row1 !== '') { body += '<div class="metrics-row" data-testid="metrics-row-quality">' + row1 + '</div>'; }

  // Row 2: PID, Collapse Metrics
  var row2 = '';
  if (data.pid !== undefined && data.pid !== null) { row2 += col(buildPidSection(data.pid)); }
  if (data.collapse !== undefined && data.collapse !== null) { row2 += col(buildCollapseSection(data.collapse)); }
  if (row2 !== '') { body += '<div class="metrics-row metrics-row-spread" data-testid="metrics-row-pid">' + row2 + '</div>'; }

  // Row 3: Per Scenario, Per Agent Config (side-by-side)
  var row3 = '';
  var scenarioHtml = buildBreakdownTable('Per Scenario', data.per_scenario);
  var agentHtml = buildBreakdownTable('Per Agent Config', data.per_agent_config);
  if (scenarioHtml !== '') { row3 += col(scenarioHtml); }
  if (agentHtml !== '') { row3 += col(agentHtml); }
  if (row3 !== '') { body += '<div class="metrics-row" data-testid="metrics-row-breakdowns">' + row3 + '</div>'; }

  return body;
}

/**
 * Build one collapsible card per experiment.
 * Accepts experiment name and its data object.
 * Returns an HTML string.
 */
export function buildExperimentCard(name, data) {
  var body = buildExperimentBody(data);
  var h = '<div class="card ablation-experiment" data-testid="ablation-experiment">';
  h += '<div class="card-header"><span>' + esc(name) + '</span><span class="arrow">&#9654;</span></div>';
  h += '<div class="card-body">' + body + '</div></div>';
  return h;
}

/**
 * Build a summary comparison table across all experiments.
 * Accepts the full experiments object.
 * Returns an HTML table string.
 */
/**
 * Extract final rho mean from an experiment or breakdown entry.
 * Returns -Infinity when unavailable so entries sort to the bottom.
 */
function rhoSortKey(d) {
  if (d.rho && d.rho.final_round && d.rho.final_round.mean != null) {
    return d.rho.final_round.mean;
  }
  return -Infinity;
}

export function buildAblationOverview(experiments) {
  var names = Object.keys(experiments);
  if (names.length === 0) return '<p>No experiments found.</p>';

  names.sort(function (a, b) { return rhoSortKey(experiments[b]) - rhoSortKey(experiments[a]); });

  var h = '<table class="data-table" data-testid="ablation-overview">';
  h += '<tr><th>Experiment</th><th>Runs</th><th>Model</th>';
  h += '<th>Final \u03c1 Mean</th><th>Final JS Mean</th><th>Collapse %</th></tr>';

  for (var i = 0; i < names.length; i++) {
    var d = experiments[names[i]];
    var rhoMean = (d.rho && d.rho.final_round) ? fmt(d.rho.final_round.mean, 4) : '\u2014';
    var jsMean = (d.js_divergence && d.js_divergence.final_round) ? fmt(d.js_divergence.final_round.mean, 4) : '\u2014';
    var collapsePct = (d.collapse !== undefined && d.collapse !== null) ? fmt(d.collapse.pct_high_rho_low_js, 1) + '%' : '\u2014';
    h += '<tr><td>' + esc(names[i]) + '</td>';
    h += '<td>' + esc(String(d.run_count)) + '</td>';
    h += '<td>' + esc(d.model) + '</td>';
    h += '<td>' + rhoMean + '</td>';
    h += '<td>' + jsMean + '</td>';
    h += '<td>' + collapsePct + '</td></tr>';
  }
  h += '</table>';
  return h;
}
