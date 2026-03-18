/**
 * components/ablation.js
 *
 * Pure HTML builders for ablation summary cards and metric tables.
 * Renders experiment-level aggregate statistics from ablation_summary.json.
 */

import { esc } from '../utils/dom.js';
import { fmt, fmtPvalue, pvalueClass, scoreClass } from '../utils/format.js';
import { T } from '../utils/labels.js';

/**
 * Build a stat cell with optional score shading.
 * Returns an HTML <td> string.
 */
function statCell(value, decimals) {
  const cls = scoreClass(value);
  const extra = cls ? ' ' + cls : '';
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
  const cfg = T('ablation_rho');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (const col of cfg.columns) {
    h += '<th>' + esc(col) + '</th>';
  }
  h += '</tr>';
  const fr = rho.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>' + esc(cfg.rows.final_round) + '</td>';
    h += statCell(fr.mean, 4) + statCell(fr.stdev, 4) + statCell(fr.min, 4) + statCell(fr.max, 4);
    h += '</tr>';
  }
  const ar = rho.all_rounds;
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
  const cfg = T('ablation_pillars');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (const col of cfg.columns) {
    h += '<th>' + esc(col) + '</th>';
  }
  h += '</tr>';
  const names = Object.keys(pillars);
  for (const name of names) {
    const p = pillars[name];
    const fr = p.final_round;
    if (fr === undefined || fr === null) continue;
    h += '<tr><td>' + esc(name) + '</td>';
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
  const cfg = T('ablation_js');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  const fr = js.final_round;
  if (fr !== undefined && fr !== null) {
    h += '<tr><td>' + esc(cfg.rows.final_round) + '</td><td>' + meanStdev(fr, 4) + '</td></tr>';
  }
  const traj = js.trajectory;
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
  const cfg = T('ablation_eo');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  const fr = eo.final_round;
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
  const parts = Object.keys(dist).map(function (key) {
    return esc(key) + ':&nbsp;' + fmt(dist[key] * 100, 1) + '%';
  });
  return parts.join(' &nbsp;\u2022&nbsp; ');
}

/** Build PID controller metrics table showing beta, quadrant, and tone distributions. */
function buildPidSection(pid) {
  const cfg = T('ablation_pid');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  h += '<tr><td>' + esc(cfg.rows.beta_final) + '</td><td>' + meanStdev(pid.beta_final, 4) + '</td></tr>';
  const qd = pid.quadrant_distribution;
  if (qd !== undefined && qd !== null) {
    h += '<tr><td>' + esc(cfg.rows.quadrant_distribution) + '</td><td>' + fmtDistribution(qd) + '</td></tr>';
  }
  const td = pid.tone_distribution;
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
  const cfg = T('ablation_collapse');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
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
  const keys = Object.keys(breakdowns);
  if (keys.length === 0) return '';

  keys.sort(function (a, b) { return rhoSortKey(breakdowns[b]) - rhoSortKey(breakdowns[a]); });

  const cfg = T('ablation_breakdown');
  let h = '<div class="section-label">' + esc(label) + '</div>';
  h += '<table class="data-table">';
  const rhoHeader = label === 'Per Agent Config' ? 'Final \u03c1' : 'Final \u03c1 Mean';
  const jsHeader = label === 'Per Agent Config' ? 'Final JS' : 'Final JS Mean';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th><th>' + rhoHeader + '</th><th>' + jsHeader + '</th></tr>';
  for (const key of keys) {
    const b = breakdowns[key];
    const rhoMean = (b.rho && b.rho.final_round) ? fmt(b.rho.final_round.mean, 4) : '\u2014';
    const jsMean = (b.js_divergence && b.js_divergence.final_round) ? fmt(b.js_divergence.final_round.mean, 4) : '\u2014';
    h += '<tr><td>' + esc(key) + '</td>';
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
  let h = '<div class="config-card">';
  h += '<div class="section-label"><strong>' + esc(configKey) + '</strong>';
  h += ' <span style="font-weight:400;color:#888;">(' + esc(String(cfg.run_count)) + ' runs)</span>';
  h += '</div>';
  let row = '';
  if (cfg.agent_deltas !== undefined && cfg.agent_deltas !== null) {
    row += col(buildAgentDeltasTable(cfg.agent_deltas));
  }
  if (cfg.mean_portfolios !== undefined && cfg.mean_portfolios !== null) {
    const rounds = Object.keys(cfg.mean_portfolios).sort();
    for (const round of rounds) {
      const rd = cfg.mean_portfolios[round];
      if (rd !== null && typeof rd === 'object' && rd.proposals_return !== undefined) {
        row += col(buildMeanPortfolioSummary(rd, round.toUpperCase()));
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
  const configs = impact.configs;
  if (configs === undefined || configs === null) return '';
  const keys = Object.keys(configs);
  if (keys.length === 0) return '';
  keys.sort();

  const h = keys.map(function (key) {
    return buildDebateImpactConfig(key, configs[key]);
  }).join('');
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
  const roles = Object.keys(agentDeltas);
  if (roles.length === 0) return '';
  const cfg = T('ablation_agent_deltas');
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table">';
  h += '<tr>';
  for (const col of cfg.columns) {
    h += '<th>' + esc(col) + '</th>';
  }
  h += '</tr>';
  for (const role of roles) {
    const d = agentDeltas[role];
    const cls = d.mean_delta_pct >= 0 ? 'perf-profit' : 'perf-loss';
    const sign = d.mean_delta_pct >= 0 ? '+' : '';
    h += '<tr><td>' + esc(role) + '</td>';
    h += '<td>' + fmt(d.mean_initial_return, 2) + '%</td>';
    h += '<td>' + fmt(d.mean_final_return, 2) + '%</td>';
    h += '<td class="' + cls + '">' + sign + fmt(d.mean_delta_pct, 2) + '%</td></tr>';
  }
  const sumInit = roles.reduce(function (s, r) { return s + agentDeltas[r].mean_initial_return; }, 0);
  const sumFinal = roles.reduce(function (s, r) { return s + agentDeltas[r].mean_final_return; }, 0);
  const sumDelta = roles.reduce(function (s, r) { return s + agentDeltas[r].mean_delta_pct; }, 0);
  h += buildDeltaMeanRow(sumInit, sumFinal, sumDelta, roles.length);
  h += '</table>';
  return h;
}

/**
 * Build the mean summary row for the agent deltas table.
 * Returns an HTML <tr> string with a top border separator.
 */
function buildDeltaMeanRow(sumInit, sumFinal, sumDelta, n) {
  const avg = sumDelta / n;
  const cls = avg >= 0 ? 'perf-profit' : 'perf-loss';
  const sign = avg >= 0 ? '+' : '';
  const sep = 'border-top:2px solid #999;font-weight:600;';
  let h = '<tr><td style="' + sep + '">' + esc(T('ablation_agent_deltas').rows.mean) + '</td>';
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
  const cls = mp.critique_impact >= 0 ? 'perf-profit' : 'perf-loss';
  const sign = mp.critique_impact >= 0 ? '+' : '';
  const cfg = T('ablation_mean_portfolio');
  let h = '<div class="section-label">' + esc(roundLabel) + ' Mean Portfolio</div>';
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
  const cfg = T('ablation_sharpe');
  const phaseKeys = ['r1_proposal', 'r1_revision', 'r1_js', 'r2_revision', 'r2_js'];
  let h = '<div class="section-label">' + esc(cfg.title) + '</div>';
  h += '<table class="data-table" data-testid="ablation-sharpe">';
  h += '<tr><th>' + esc(cfg.columns[0]) + '</th><th>' + esc(cfg.columns[1]) + '</th></tr>';
  for (const key of phaseKeys) {
    const val = sharpe[key];
    h += '<tr><td>' + esc(cfg.rows[key]) + '</td>';
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
  const meta = T('ablation_experiment_meta');
  let body = '<div class="experiment-meta">';
  body += '<table class="ov-htable"><tr>';
  body += '<th>' + esc(meta.columns[0]) + '</th><td>' + esc(String(data.run_count)) + '</td>';
  body += '<th>' + esc(meta.columns[1]) + '</th><td>' + esc(data.model) + '</td>';
  body += '</tr></table></div>';

  // CRIT Reasoning Diagnostics placeholder (populated async by ablationView)
  body += '<div data-testid="crit-diagnostics-slot"></div>';

  // Row 0: Debate Impact (per agent config)
  body += buildDebateImpactSection(impact);

  // Paired t-test placeholder (populated async by ablationView)
  body += '<div data-testid="paired-tests-slot"></div>';

  // Financial metrics placeholders (populated async by ablationView)
  body += '<div data-testid="financial-tests-slot"></div>';
  body += '<div data-testid="financial-tests-mr-slot"></div>';

  // Row 1: Rho, Pillars, JS Divergence, Evidence Overlap
  let row1 = '';
  if (data.rho !== undefined && data.rho !== null) { row1 += col(buildRhoTable(data.rho)); }
  if (data.pillars !== undefined && data.pillars !== null) { row1 += col(buildPillarsTable(data.pillars)); }
  if (data.js_divergence !== undefined && data.js_divergence !== null) { row1 += col(buildJsSection(data.js_divergence)); }
  if (data.evidence_overlap !== undefined && data.evidence_overlap !== null) { row1 += col(buildEoSection(data.evidence_overlap)); }
  if (row1 !== '') { body += '<div class="metrics-row metrics-row-quality" data-testid="metrics-row-quality">' + row1 + '</div>'; }

  // Row 2: PID, Collapse Metrics
  let row2 = '';
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
  let row = '';
  const scenarioHtml = buildBreakdownTable('Per Scenario', perScenario);
  const agentHtml = buildBreakdownTable('Per Agent Config', perAgentConfig);
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
  const body = buildExperimentBody(data, impact);
  let h = '<div class="card ablation-experiment" data-testid="ablation-experiment">';
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

/**
 * Build the paired t-test results section for an experiment.
 * Renders summary stats, t-test results, and per-scenario table.
 * Accepts the paired test result object from the API.
 * Returns an HTML string.
 */
export function buildPairedTestsSection(data) {
  if (data === undefined || data === null) return '';
  if (data.pending === true) {
    return '<div class="section-label">Paired t-Test (Collapse Ratio)</div>'
      + '<p class="status-incomplete">' + esc(data.message) + '</p>';
  }
  if (data.error !== undefined) {
    return '<div class="section-label">Paired Statistical Test</div>'
      + '<p>' + esc(data.error) + '</p>';
  }

  const t = data.ttest;
  const s = data.summary;
  let h = '<div class="section-label">Paired t-Test (Collapse Ratio)</div>';

  // Summary row
  h += '<table class="data-table" data-testid="paired-test-summary">';
  h += '<tr><th>N pairs</th><th>t-statistic</th><th>p-value</th>';
  h += '<th>Mean Diff (B\u2212A)</th><th>95% CI</th></tr>';

  const pClass = pvalueClass(t.p_value);
  h += '<tr>';
  h += '<td>' + esc(String(data.n_paired)) + '</td>';
  h += '<td>' + fmt(t.t_statistic, 4) + '</td>';
  h += '<td class="' + pClass + '">' + fmtPvalue(t.p_value) + '</td>';
  h += '<td>' + fmt(t.mean_diff, 4) + '</td>';
  h += '<td>[' + fmt(t.ci_95[0], 4) + ', ' + fmt(t.ci_95[1], 4) + ']</td>';
  h += '</tr></table>';

  // Condition means
  h += '<table class="data-table" data-testid="paired-test-conditions">';
  h += '<tr><th>Condition</th><th>Mean \u00b1 SEM</th>';
  h += '<th>Std Dev</th></tr>';
  h += '<tr><td>(A) ' + esc(data.config_a) + '</td>';
  h += '<td>' + fmt(s.a_mean, 4) + ' \u00b1 ' + fmt(s.a_sem, 4) + '</td>';
  h += '<td>' + fmt(s.a_std, 4) + '</td></tr>';
  h += '<tr><td>(B) ' + esc(data.config_b) + '</td>';
  h += '<td>' + fmt(s.b_mean, 4) + ' \u00b1 ' + fmt(s.b_sem, 4) + '</td>';
  h += '<td>' + fmt(s.b_std, 4) + '</td></tr>';
  h += '</table>';

  // Direction summary
  h += '<p>Intervention preserved divergence in <strong>'
    + esc(String(s.n_b_greater)) + '/' + esc(String(data.n_paired))
    + '</strong> scenarios</p>';

  // Per-scenario table
  h += buildPairedScenarioTable(data.pairs);

  return h;
}

/**
 * Build the per-scenario paired comparison table.
 * Accepts the pairs array [{scenario, a, b}, ...].
 * Returns an HTML table string.
 */
function buildPairedScenarioTable(pairs) {
  if (!Array.isArray(pairs) || pairs.length === 0) return '';

  let h = '<div class="section-label">Per-Scenario Collapse Ratios</div>';
  h += '<table class="data-table" data-testid="paired-test-scenarios">';
  h += '<tr><th>Scenario</th><th>Baseline CR</th>';
  h += '<th>Intervention CR</th><th>\u0394 (B\u2212A)</th></tr>';

  for (const p of pairs) {
    const delta = p.b - p.a;
    const cls = delta >= 0 ? 'perf-profit' : 'perf-loss';
    const sign = delta >= 0 ? '+' : '';
    h += '<tr><td>' + esc(p.scenario) + '</td>';
    h += '<td>' + fmt(p.a, 4) + '</td>';
    h += '<td>' + fmt(p.b, 4) + '</td>';
    h += '<td class="' + cls + '">' + sign + fmt(delta, 4) + '</td></tr>';
  }
  h += '</table>';
  return h;
}

/** Build a summary comparison table ranking all experiments by final rho. */
export function buildAblationOverview(experiments) {
  const names = Object.keys(experiments);
  if (names.length === 0) return '<p>No experiments found.</p>';

  names.sort(function (a, b) { return rhoSortKey(experiments[b]) - rhoSortKey(experiments[a]); });

  const cfg = T('ablation_overview');
  let h = '<table class="data-table" data-testid="ablation-overview">';
  h += '<tr>';
  for (const col of cfg.columns) {
    h += '<th>' + esc(col) + '</th>';
  }
  h += '</tr>';

  for (const name of names) {
    const d = experiments[name];
    const rhoMean = (d.rho && d.rho.final_round) ? fmt(d.rho.final_round.mean, 4) : '\u2014';
    const jsMean = (d.js_divergence && d.js_divergence.final_round) ? fmt(d.js_divergence.final_round.mean, 4) : '\u2014';
    const collapsePct = (d.collapse !== undefined && d.collapse !== null) ? fmt(d.collapse.pct_high_rho_low_js, 1) + '%' : '\u2014';
    h += '<tr><td>' + esc(name) + '</td>';
    h += '<td>' + esc(String(d.run_count)) + '</td>';
    h += '<td>' + esc(d.model) + '</td>';
    h += '<td>' + rhoMean + '</td>';
    h += '<td>' + jsMean + '</td>';
    h += '<td>' + collapsePct + '</td></tr>';
  }
  h += '</table>';
  return h;
}
