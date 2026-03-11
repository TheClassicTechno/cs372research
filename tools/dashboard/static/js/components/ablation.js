/**
 * components/ablation.js
 *
 * Pure HTML builders for ablation summary cards and metric tables.
 * Renders experiment-level aggregate statistics from ablation_summary.json.
 */

import { esc } from '../utils/dom.js';
import { fmt, scoreClass } from '../utils/format.js';
import { T } from '../utils/labels.js';

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
  var cfg = T('ablation_rho');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (var c = 0; c < cfg.columns.length; c++) {
    h += '<th>' + esc(cfg.columns[c]) + '</th>';
  }
  h += '</tr>';
  var fr = rho.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>' + esc(cfg.rows.final_round) + '</td>';
    h += statCell(fr.mean, 4) + statCell(fr.stdev, 4) + statCell(fr.min, 4) + statCell(fr.max, 4);
    h += '</tr>';
  }
  var ar = rho.all_rounds;
  if (ar !== undefined && ar !== null) {
    h += '<tr><td>' + esc(cfg.rows.all_rounds) + '</td>';
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
  var cfg = T('ablation_pillars');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (var c = 0; c < cfg.columns.length; c++) {
    h += '<th>' + esc(cfg.columns[c]) + '</th>';
  }
  h += '</tr>';
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
  var cfg = T('ablation_js');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  var fr = js.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>' + esc(cfg.rows.final_round) + '</td><td>' + meanStdev(fr, 4) + '</td></tr>';
  }
  var traj = js.trajectory;
  if (traj !== undefined && traj !== null) {
    h += '<tr><td>' + esc(cfg.rows.mean_delta) + '</td><td>' + fmt(traj.mean_delta, 4) + '</td></tr>';
    h += '<tr><td>' + esc(cfg.rows.pct_decreased) + '</td><td>' + fmt(traj.pct_decreased, 1) + '%</td></tr>';
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
  var cfg = T('ablation_eo');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  var fr = eo.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>' + esc(cfg.rows.final_round) + '</td><td>' + meanStdev(fr, 4) + '</td></tr>';
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
  var cfg = T('ablation_pid');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  h += '<tr><td>' + esc(cfg.rows.beta_final) + '</td><td>' + meanStdev(pid.beta_final, 4) + '</td></tr>';
  var qd = pid.quadrant_distribution;
  if (qd !== undefined && qd !== null) {
    h += '<tr><td>' + esc(cfg.rows.quadrant_distribution) + '</td><td>' + fmtDistribution(qd) + '</td></tr>';
  }
  var td = pid.tone_distribution;
  if (td !== undefined && td !== null) {
    h += '<tr><td>' + esc(cfg.rows.tone_distribution) + '</td><td>' + fmtDistribution(td) + '</td></tr>';
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
  var cfg = T('ablation_collapse');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  h += '<tr><td>' + esc(cfg.rows.js_lt_005) + '</td><td>' + esc(String(collapse.js_lt_005)) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.js_lt_007) + '</td><td>' + esc(String(collapse.js_lt_007)) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.js_lt_010) + '</td><td>' + esc(String(collapse.js_lt_010)) + '</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.high_rho_low_js) + '</td><td>' + esc(String(collapse.high_rho_low_js));
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

  var cfg = T('ablation_breakdown');
  var h = '<div class="section-label">' + esc(label) + '</div>';
  h += '<table class="data-table">';
  var rhoHeader = label === 'Per Agent Config' ? 'Final \u03c1' : 'Final \u03c1 Mean';
  var jsHeader = label === 'Per Agent Config' ? 'Final JS' : 'Final JS Mean';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th><th>' + rhoHeader + '</th><th>' + jsHeader + '</th></tr>';
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
 * Build one config subsection for the debate impact area.
 * Shows per-agent deltas and mean portfolio critique tables in a flex row.
 *
 * @param {string} configKey - sorted role string (e.g. "risk, technical, value")
 * @param {object} cfg - {run_count, agent_deltas, mean_portfolios}
 * @returns {string} HTML string for the config subsection
 */
function buildDebateImpactConfig(configKey, cfg) {
  var h = '<div class="config-card">';
  h += '<div class="section-label"><strong>' + esc(configKey) + '</strong>';
  h += ' <span style="font-weight:400;color:#888;">(' + esc(String(cfg.run_count)) + ' runs)</span>';
  h += '</div>';
  var row = '';
  if (cfg.agent_deltas !== undefined && cfg.agent_deltas !== null) {
    row += col(buildAgentDeltasTable(cfg.agent_deltas));
  }
  if (cfg.mean_portfolios !== undefined && cfg.mean_portfolios !== null) {
    var rounds = Object.keys(cfg.mean_portfolios).sort();
    for (var r = 0; r < rounds.length; r++) {
      var rd = cfg.mean_portfolios[rounds[r]];
      if (rd !== null && typeof rd === 'object' && rd.proposals_return !== undefined) {
        row += col(buildMeanPortfolioSummary(rd, rounds[r].toUpperCase()));
      }
    }
  }
  if (cfg.sharpe !== undefined && cfg.sharpe !== null) {
    row += col(buildAblationSharpeTable(cfg.sharpe));
  }
  if (row !== '') {
    h += '<div class="metrics-row">' + row + '</div>';
  }
  h += '</div>';
  return h;
}

/**
 * Build the debate impact section for an experiment card.
 * Iterates per-agent-config groups and renders each as a subsection.
 *
 * @param {object} impact - {configs: {key: {run_count, agent_deltas, mean_portfolios}}}
 * @returns {string} HTML string for the debate impact section
 */
function buildDebateImpactSection(impact) {
  if (impact === undefined || impact === null) return '';
  var configs = impact.configs;
  if (configs === undefined || configs === null) return '';
  var keys = Object.keys(configs);
  if (keys.length === 0) return '';
  keys.sort();

  var h = '';
  for (var i = 0; i < keys.length; i++) {
    h += buildDebateImpactConfig(keys[i], configs[keys[i]]);
  }
  return '<div data-testid="debate-impact-section">' + h + '</div>';
}

/**
 * Build the per-agent debate delta table for ablation.
 * Shows averaged R1 proposal vs final revision returns per agent.
 *
 * @param {object} agentDeltas - {role: {mean_initial_return, mean_final_return, mean_delta_pct}}
 * @returns {string} HTML table string
 */
function buildAgentDeltasTable(agentDeltas) {
  var roles = Object.keys(agentDeltas);
  if (roles.length === 0) return '';
  var cfg = T('ablation_agent_deltas');
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (var c = 0; c < cfg.columns.length; c++) {
    h += '<th>' + esc(cfg.columns[c]) + '</th>';
  }
  h += '</tr>';
  var sumInit = 0;
  var sumFinal = 0;
  var sumDelta = 0;
  for (var i = 0; i < roles.length; i++) {
    var d = agentDeltas[roles[i]];
    var cls = d.mean_delta_pct >= 0 ? 'perf-profit' : 'perf-loss';
    var sign = d.mean_delta_pct >= 0 ? '+' : '';
    h += '<tr><td>' + esc(roles[i]) + '</td>';
    h += '<td>' + fmt(d.mean_initial_return, 2) + '%</td>';
    h += '<td>' + fmt(d.mean_final_return, 2) + '%</td>';
    h += '<td class="' + cls + '">' + sign + fmt(d.mean_delta_pct, 2) + '%</td></tr>';
    sumInit += d.mean_initial_return;
    sumFinal += d.mean_final_return;
    sumDelta += d.mean_delta_pct;
  }
  h += buildDeltaMeanRow(sumInit, sumFinal, sumDelta, roles.length);
  h += '</table>';
  return h;
}

/**
 * Build the mean summary row for the agent deltas table.
 * Returns an HTML <tr> string with a top border separator.
 */
function buildDeltaMeanRow(sumInit, sumFinal, sumDelta, n) {
  var avg = sumDelta / n;
  var cls = avg >= 0 ? 'perf-profit' : 'perf-loss';
  var sign = avg >= 0 ? '+' : '';
  var sep = 'border-top:2px solid #999;font-weight:600;';
  var h = '<tr><td style="' + sep + '">' + esc(T('ablation_agent_deltas').rows.mean) + '</td>';
  h += '<td style="' + sep + '">' + fmt(sumInit / n, 2) + '%</td>';
  h += '<td style="' + sep + '">' + fmt(sumFinal / n, 2) + '%</td>';
  h += '<td class="' + cls + '" style="' + sep + '">' + sign + fmt(avg, 2) + '%</td></tr>';
  return h;
}

/**
 * Build the mean portfolio critique impact summary for one round.
 * Shows proposals vs revisions returns for a given round.
 *
 * @param {object} mp - {proposals_return, revisions_return, critique_impact}
 * @param {string} roundLabel - Display label (e.g. "R1", "R2")
 * @returns {string} HTML table string
 */
function buildMeanPortfolioSummary(mp, roundLabel) {
  var cls = mp.critique_impact >= 0 ? 'perf-profit' : 'perf-loss';
  var sign = mp.critique_impact >= 0 ? '+' : '';
  var cfg = T('ablation_mean_portfolio');
  var h = '<div class="section-label">' + esc(roundLabel) + ' Mean Portfolio</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  h += '<tr><td>' + esc(cfg.rows.proposals) + '</td><td>' + fmt(mp.proposals_return, 2) + '%</td></tr>';
  h += '<tr><td>' + esc(cfg.rows.revisions) + '</td><td>' + fmt(mp.revisions_return, 2) + '%</td></tr>';
  h += '<tr><td style="font-weight:600;">' + esc(cfg.rows.critique_delta) + '</td>';
  h += '<td class="' + cls + '" style="font-weight:600;">' + sign + fmt(mp.critique_impact, 2) + '%</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Build aggregate Sharpe ratio table for one config group.
 * Shows mean annualized Sharpe per debate phase.
 *
 * @param {object} sharpe - {r1_proposal, r1_revision, r1_js, r2_revision, r2_js}
 * @returns {string} HTML table string
 */
function buildAblationSharpeTable(sharpe) {
  var cfg = T('ablation_sharpe');
  var phaseKeys = ['r1_proposal', 'r1_revision', 'r1_js', 'r2_revision', 'r2_js'];
  var h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table" data-testid="ablation-sharpe">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  for (var i = 0; i < phaseKeys.length; i++) {
    var val = sharpe[phaseKeys[i]];
    h += '<tr><td>' + esc(cfg.rows[phaseKeys[i]]) + '</td>';
    h += '<td style="text-align:right;">';
    h += (val !== null && val !== undefined) ? fmt(val, 4) : '\u2014';
    h += '</td></tr>';
  }
  h += '</table>';
  return h;
}

/**
 * Build the inner body HTML for an experiment card.
 * Accepts the experiment data object and optional debate impact data.
 * Returns an HTML string with all metric sections arranged in rows.
 *
 * Row 0: Debate Impact (agent deltas + mean portfolio)
 * Row 1: Rho, CRIT Pillars, JS Divergence, Evidence Overlap (side-by-side)
 * Row 2: PID, Collapse Metrics (side-by-side)
 * Row 3: Per Scenario, Per Agent Config (side-by-side)
 */
function buildExperimentBody(data, impact) {
  var meta = T('ablation_experiment_meta');
  var body = '<div class="experiment-meta">';
  body += '<table class="ov-htable"><tr>';
  body += '<th>' + esc(meta.columns[0]) + '</th><td>' + esc(String(data.run_count)) + '</td>';
  body += '<th>' + esc(meta.columns[1]) + '</th><td>' + esc(data.model) + '</td>';
  body += '</tr></table></div>';

  // Row 0: Debate Impact (per agent config)
  body += buildDebateImpactSection(impact);

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
  body += buildBreakdownRow(data.per_scenario, data.per_agent_config);

  return body;
}

/**
 * Build the breakdown metrics row (Per Scenario + Per Agent Config).
 * Returns an HTML string or empty string if no data.
 */
function buildBreakdownRow(perScenario, perAgentConfig) {
  var row = '';
  var scenarioHtml = buildBreakdownTable('Per Scenario', perScenario);
  var agentHtml = buildBreakdownTable('Per Agent Config', perAgentConfig);
  if (scenarioHtml !== '') { row += col(scenarioHtml); }
  if (agentHtml !== '') { row += col(agentHtml); }
  if (row !== '') { return '<div class="metrics-row" data-testid="metrics-row-breakdowns">' + row + '</div>'; }
  return '';
}

/**
 * Build one collapsible card per experiment.
 * Accepts experiment name and its data object.
 * Returns an HTML string.
 */
export function buildExperimentCard(name, data, impact) {
  var body = buildExperimentBody(data, impact);
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

  var cfg = T('ablation_overview');
  var h = '<table class="data-table" data-testid="ablation-overview">';
  h += '<tr>';
  for (var c = 0; c < cfg.columns.length; c++) {
    h += '<th>' + esc(cfg.columns[c]) + '</th>';
  }
  h += '</tr>';

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
