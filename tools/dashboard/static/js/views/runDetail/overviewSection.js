import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';

export function buildOverviewPanel(detail, experiment, runId) {
  var m = detail.manifest || {};
  var q = detail.quality || {};

  var configName = '\u2014';
  if (m.config_paths && m.config_paths.length > 0) {
    var cp = m.config_paths[0].replace(/\\/g, '/').split('/');
    configName = cp[cp.length - 1].replace(/\.yaml$/, '').replace(/\.yml$/, '');
  }

  var agentsStr = '\u2014';
  if (m.agent_profiles && typeof m.agent_profiles === 'object') {
    agentsStr = Object.values(m.agent_profiles).join(', ');
  } else if (m.roles && m.roles.length > 0) {
    agentsStr = m.roles.join(', ');
  }

  var tickersStr = '\u2014';
  if (m.ticker_universe && m.ticker_universe.length > 0) {
    tickersStr = m.ticker_universe.filter(function (t) { return t !== '_CASH_'; }).join(', ');
  }

  var roundsStr = (m.actual_rounds != null ? m.actual_rounds : '\u2014') + ' / ' + (m.max_rounds != null ? m.max_rounds : '\u2014');

  var statusCls = (detail.status === 'complete' || detail.status === 'partial') ? ' class="status-ok"' : '';

  var html = '<div class="run-overview">';
  html += '<div class="ov-title">RUN OVERVIEW</div>';

  // Table 1 — run identity
  html += '<table class="ov-htable">';
  html += '<tr><th>Run ID</th><th>Experiment</th><th>Model</th><th>CRIT Model</th><th>Status</th></tr>';
  html += '<tr>';
  html += '<td>' + esc(runId) + '</td>';
  html += '<td>' + esc(experiment) + '</td>';
  html += '<td>' + esc(m.model_name || '\u2014') + '</td>';
  html += '<td>' + esc(m.crit_model_name || '\u2014') + '</td>';
  html += '<td' + statusCls + '>' + esc(detail.status) + '</td>';
  html += '</tr></table>';

  // Table 2 — config and agents (own rows to avoid wrapping)
  html += '<table class="ov-htable" style="table-layout:auto;">';
  html += '<tr><th style="width:1%;white-space:nowrap">Config</th><td>' + esc(configName) + '</td></tr>';
  html += '<tr><th style="width:1%;white-space:nowrap">Agents</th><td>' + esc(agentsStr) + '</td></tr>';
  html += '</table>';

  // Table 3 — metrics
  html += '<table class="ov-htable">';
  html += '<tr><th>Tickers</th><th>Rounds</th><th>Termination</th><th>Final \u03B2</th><th>Final <span style="text-decoration:overline">\u03c1</span></th><th>JS Drop</th></tr>';
  html += '<tr>';
  html += '<td>' + esc(tickersStr) + '</td>';
  html += '<td>' + esc(roundsStr) + '</td>';
  html += '<td>' + esc(m.termination_reason || '\u2014') + '</td>';
  html += '<td>' + fmt(m.final_beta) + '</td>';
  html += '<td>' + fmt(q.final_rho_bar) + '</td>';
  html += '<td>' + fmt(q.js_drop) + '</td>';
  html += '</tr></table>';

  if (q.reasoning_collapse) {
    html += '<div class="ov-warn">REASONING COLLAPSE DETECTED</div>';
  }

  // Judge portfolio placeholder
  html += '<div id="judge-portfolio-section"></div>';

  html += '</div>';
  return html;
}
