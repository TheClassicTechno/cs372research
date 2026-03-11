import { esc } from '../../utils/dom.js';
import { fmt, numFmt } from '../../utils/format.js';
import { flattenConfig } from '../../utils/config.js';

/**
 * Build the RUN OVERVIEW metadata panel.
 * Returns an HTML string with run identity, execution metrics,
 * config parameters, and optional macro environment preview.
 *
 * @param {object} detail - Full run detail from API
 * @param {string} experiment - Experiment name
 * @param {string} runId - Run ID
 * @returns {string} HTML string
 */
export function buildOverviewPanel(detail, experiment, runId) {
  var html = '<div class="run-overview">';
  html += '<div class="ov-title">RUN OVERVIEW</div>';
  html += buildIdentityTable(detail, experiment, runId);
  html += buildMetricsTable(detail);
  html += buildRunDirRow(detail);
  html += buildConfigGrid(detail.debate_config, 'DEBATE CONFIG', 'debate-config-grid', DEBATE_EXCLUDE);
  html += buildConfigGrid(detail.scenario_config, 'SCENARIO CONFIG', 'scenario-config-grid', SCENARIO_EXCLUDE);
  html += buildTickerPerfTable(detail.ticker_performance);
  html += buildMacroPreview(detail.scenario_config);

  var q = detail.quality || {};
  if (q.reasoning_collapse) {
    html += '<div class="ov-warn">REASONING COLLAPSE DETECTED</div>';
  }

  html += '<div id="judge-portfolio-section"></div>';
  html += '</div>';
  return html;
}

/** Keys excluded from debate config grid (shown in identity/metrics tables). */
var DEBATE_EXCLUDE = [
  'agents',
  'debate_setup.llm_model',
  'debate_setup.experiment_name',
  'debate_setup.max_rounds',
  'debate_setup.temperature',
];

/** Keys excluded from scenario config grid. */
var SCENARIO_EXCLUDE = [
  'tickers',
  'macro_context',
  'output_dir',
];

/**
 * Build Section 1 — Core Run Info table.
 *
 * @param {object} detail - Run detail object
 * @param {string} experiment - Experiment name
 * @param {string} runId - Run ID
 * @returns {string} HTML table string
 */
function buildIdentityTable(detail, experiment, runId) {
  var m = detail.manifest || {};
  var statusCls = (detail.status === 'complete' || detail.status === 'partial') ? ' class="status-ok"' : '';
  var h = '<table class="ov-htable">';
  h += '<tr><th>Run ID</th><th>Experiment</th><th>Model</th><th>CRIT Model</th><th>Status</th></tr>';
  h += '<tr>';
  h += '<td>' + esc(runId) + '</td>';
  h += '<td>' + esc(experiment) + '</td>';
  h += '<td>' + esc(m.model_name || '\u2014') + '</td>';
  h += '<td>' + esc(m.crit_model_name || '\u2014') + '</td>';
  h += '<td' + statusCls + '>' + esc(detail.status) + '</td>';
  h += '</tr></table>';
  return h;
}

/**
 * Extract config and agent display strings from manifest.
 *
 * @param {object} m - Manifest object
 * @returns {{configName: string, agentsStr: string}} Display strings
 */
function extractConfigFields(m) {
  var configName = '\u2014';
  if (m.config_paths && m.config_paths.length > 0) {
    var cp = m.config_paths[0].replace(/\\/g, '/').split('/');
    configName = cp[cp.length - 1].replace(/\.yaml$/, '').replace(/\.yml$/, '');
  }
  var agentsStr = '\u2014';
  if (m.agent_profiles && typeof m.agent_profiles === 'object') {
    var vals = Object.entries(m.agent_profiles);
    agentsStr = vals.map(function (entry) {
      return typeof entry[1] === 'string' ? entry[1] : entry[0];
    }).join(', ');
  } else if (m.roles && m.roles.length > 0) {
    agentsStr = m.roles.join(', ');
  }
  return { configName: configName, agentsStr: agentsStr };
}

/**
 * Build Section 2 — Execution Metadata table.
 *
 * @param {object} detail - Run detail object
 * @returns {string} HTML table string
 */
function buildMetricsTable(detail) {
  var m = detail.manifest || {};
  var q = detail.quality || {};
  var fields = extractConfigFields(m);
  var roundsStr = (m.actual_rounds != null ? m.actual_rounds : '\u2014') + ' / ' + (m.max_rounds != null ? m.max_rounds : '\u2014');
  var tempStr = '\u2014';
  var dc = detail.debate_config;
  if (dc && dc.debate_setup && dc.debate_setup.temperature != null) {
    tempStr = String(dc.debate_setup.temperature);
  }
  var h = '<table class="ov-htable">';
  h += '<tr><th>Config</th><th>Agents</th><th>Rounds</th><th>Termination</th>';
  h += '<th>Temperature</th><th>Initial \u03B2</th>';
  h += '<th>Final \u03B2</th><th>Final <span style="text-decoration:overline">\u03c1</span></th><th>JS Drop</th></tr>';
  h += '<tr>';
  h += '<td>' + esc(fields.configName) + '</td>';
  h += '<td>' + esc(fields.agentsStr) + '</td>';
  h += '<td>' + esc(roundsStr) + '</td>';
  h += '<td>' + esc(m.termination_reason || '\u2014') + '</td>';
  h += '<td>' + esc(tempStr) + '</td>';
  h += '<td>' + fmt(m.initial_beta) + '</td>';
  h += '<td>' + fmt(m.final_beta) + '</td>';
  h += '<td>' + fmt(q.final_rho_bar) + '</td>';
  h += '<td>' + fmt(q.js_drop) + '</td>';
  h += '</tr></table>';
  return h;
}

/**
 * Build Section 3 — Run Directory row.
 *
 * @param {object} detail - Run detail object
 * @returns {string} HTML table string
 */
function buildRunDirRow(detail) {
  if (typeof detail.run_dir !== 'string') return '';
  var h = '<table class="ov-htable" style="table-layout:auto;">';
  h += '<tr><th style="width:1%;white-space:nowrap">Run Dir</th>';
  h += '<td style="font-family:monospace">' + esc(detail.run_dir) + '</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Filter and cap config items, then render as grid cells.
 *
 * @param {Array} flat - Flattened config items
 * @param {Array} excludeKeys - Keys to exclude
 * @returns {Array} Filtered items (max 100)
 */
function filterConfigItems(flat, excludeKeys) {
  var excludeSet = {};
  for (var e = 0; e < excludeKeys.length; e++) {
    excludeSet[excludeKeys[e]] = true;
  }
  var filtered = [];
  for (var i = 0; i < flat.length; i++) {
    if (!excludeSet[flat[i].key]) {
      filtered.push(flat[i]);
    }
  }
  if (filtered.length > 100) {
    filtered = filtered.slice(0, 100);
  }
  return filtered;
}

/**
 * Render grouped config items as HTML grid cells.
 *
 * @param {Array} grouped - Items from collapseNamespaces()
 * @returns {string} HTML string of grid cells
 */
function renderConfigCells(grouped) {
  var h = '';
  for (var g = 0; g < grouped.length; g++) {
    var item = grouped[g];
    if (item.children) {
      h += '<div class="ov-config-cell">';
      h += '<span class="ov-config-key">' + esc(item.key) + '</span>';
      for (var c = 0; c < item.children.length; c++) {
        var child = item.children[c];
        var val = child.value.length > 120 ? child.value.slice(0, 120) + '\u2026' : child.value;
        h += '<span class="ov-config-val">' + esc(child.key) + ': ' + esc(val) + '</span>';
      }
      h += '</div>';
    } else {
      var val = item.value.length > 120 ? item.value.slice(0, 120) + '\u2026' : item.value;
      h += '<div class="ov-config-cell">';
      h += '<span class="ov-config-key">' + esc(item.key) + '</span>';
      h += '<span class="ov-config-val">' + esc(val) + '</span>';
      h += '</div>';
    }
  }
  return h;
}

/**
 * Build Section 4 — Config Parameter Grid.
 *
 * @param {object} config - Raw config object
 * @param {string} title - Section title
 * @param {string} testId - data-testid attribute
 * @param {Array} excludeKeys - Keys to exclude
 * @returns {string} HTML string
 */
function buildConfigGrid(config, title, testId, excludeKeys) {
  if (!config || typeof config !== 'object') return '';
  var flat = flattenConfig(config);
  if (!flat.length) return '';
  var filtered = filterConfigItems(flat, excludeKeys);
  if (!filtered.length) return '';
  var grouped = collapseNamespaces(filtered);
  var h = '<div class="ov-subtitle">' + esc(title) + '</div>';
  h += '<div class="ov-config-grid" data-testid="' + esc(testId) + '">';
  h += renderConfigCells(grouped);
  h += '</div>';
  return h;
}

/**
 * Collapse shared prefixes: when >=3 sibling keys share a common prefix,
 * group them into a single cell with indented sub-keys.
 *
 * @param {Array} items - Flattened config items
 * @returns {Array} Grouped items
 */
function collapseNamespaces(items) {
  var groups = {};
  var order = [];
  for (var i = 0; i < items.length; i++) {
    var dotIdx = items[i].key.lastIndexOf('.');
    var parent = dotIdx >= 0 ? items[i].key.slice(0, dotIdx) : '';
    var child = dotIdx >= 0 ? items[i].key.slice(dotIdx + 1) : items[i].key;
    if (!groups[parent]) {
      groups[parent] = [];
      order.push(parent);
    }
    groups[parent].push({ key: child, value: items[i].value, fullKey: items[i].key });
  }
  var result = [];
  for (var o = 0; o < order.length; o++) {
    var prefix = order[o];
    var members = groups[prefix];
    if (prefix && members.length >= 3) {
      result.push({ key: prefix, children: members });
    } else {
      for (var m = 0; m < members.length; m++) {
        result.push({ key: members[m].fullKey, value: members[m].value });
      }
    }
  }
  return result;
}

/**
 * Build Section 5 — Macro Environment Preview.
 *
 * @param {object} scenarioConfig - Scenario config object
 * @returns {string} HTML string
 */
function buildMacroPreview(scenarioConfig) {
  if (!scenarioConfig || !scenarioConfig.macro_context) return '';
  var text = String(scenarioConfig.macro_context);
  if (text.length > 500) {
    text = text.slice(0, 500) + '\u2026';
  }
  var h = '<div class="ov-subtitle">MACRO ENVIRONMENT</div>';
  h += '<div class="ov-macro-box">' + esc(text) + '</div>';
  return h;
}

/**
 * Build Ticker Performance Table.
 *
 * @param {Array} tickerPerf - Array of {ticker, open, close, pct_change}
 * @returns {string} HTML table string
 */
function buildTickerPerfTable(tickerPerf) {
  if (!tickerPerf || tickerPerf.length === 0) return '';
  var h = '<div class="ov-subtitle">TICKER PERFORMANCE</div>';
  h += '<table class="data-table" data-testid="ticker-perf-table">';
  h += '<tr><th>Ticker</th><th>Open</th><th>Close</th><th>\u0394%</th></tr>';
  for (var i = 0; i < tickerPerf.length; i++) {
    var t = tickerPerf[i];
    var cls = t.pct_change >= 0 ? 'perf-profit' : 'perf-loss';
    var sign = t.pct_change >= 0 ? '+' : '';
    h += '<tr><td style="font-weight:600;">' + esc(t.ticker) + '</td>';
    h += '<td style="text-align:right;">$' + numFmt(t.open) + '</td>';
    h += '<td style="text-align:right;">$' + numFmt(t.close) + '</td>';
    h += '<td class="' + cls + '" style="text-align:right;">';
    h += sign + t.pct_change.toFixed(2) + '%</td></tr>';
  }
  h += '</table>';
  return h;
}
