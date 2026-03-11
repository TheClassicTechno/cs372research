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
  html += buildRunSummary(detail, experiment, runId);
  html += buildKeyMetrics(detail);
  html += buildExperimentConfig(detail);
  html += buildDebateConfigGroups(detail.debate_config);
  html += buildScenarioConfigGroups(detail.scenario_config);
  html += buildTickerPerfTable(detail.ticker_performance);
  html += buildMacroCollapsible(detail.scenario_config);

  var q = detail.quality || {};
  if (q.reasoning_collapse) {
    html += '<div class="ov-warn">REASONING COLLAPSE DETECTED</div>';
  }

  html += '<div id="judge-portfolio-section"></div>';
  html += '</div>';
  return html;
}

/** Keys excluded from debate config grid (shown in summary/metrics). */
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

/** Semantic group definitions for debate config parameters. */
var DEBATE_GROUP_DEFS = [
  { name: 'Agent Setup', prefixes: ['agent'] },
  { name: 'Broker', prefixes: ['broker'] },
  { name: 'Dataset', prefixes: ['dataset'] },
  { name: 'Judge', prefixes: ['judge'] },
  { name: 'PID Settings', prefixes: ['pid'] },
  { name: 'Runtime', prefixes: ['runtime', 'debate_setup'] },
];

/** Semantic group definitions for scenario config parameters. */
var SCENARIO_GROUP_DEFS = [
  { name: 'Allocation Constraints', prefixes: ['allocation', 'constraint'] },
];

/**
 * Build Section 1 — Run Summary table with two header/data row pairs.
 *
 * @param {object} detail - Run detail object
 * @param {string} experiment - Experiment name
 * @param {string} runId - Run ID
 * @returns {string} HTML table string
 */
function buildRunSummary(detail, experiment, runId) {
  var m = detail.manifest || {};
  var statusCls = '';
  if (detail.status === 'complete' || detail.status === 'partial') {
    statusCls = ' class="status-ok"';
  } else if (detail.status === 'running') {
    statusCls = ' class="status-running"';
  } else if (detail.status === 'failed') {
    statusCls = ' class="status-failed"';
  }

  var h = '<table class="ov-htable">';
  h += '<tr><th>Run ID</th><th>Experiment</th><th>Status</th></tr>';
  h += '<tr>';
  h += '<td>' + esc(runId) + '</td>';
  h += '<td>' + esc(experiment) + '</td>';
  h += '<td' + statusCls + '>' + esc(detail.status) + '</td>';
  h += '</tr>';
  h += '<tr><th>Model</th><th>CRIT Model</th><th>Run Dir</th></tr>';
  h += '<tr>';
  h += '<td>' + esc(m.model_name || '\u2014') + '</td>';
  h += '<td>' + esc(m.crit_model_name || '\u2014') + '</td>';
  h += '<td style="font-family:monospace">' + esc(typeof detail.run_dir === 'string' ? detail.run_dir : '\u2014') + '</td>';
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
 * Build Section 2 — Key Metrics row with 5 metric cards.
 *
 * @param {object} detail - Run detail object
 * @returns {string} HTML string
 */
function buildKeyMetrics(detail) {
  var m = detail.manifest || {};
  var q = detail.quality || {};
  var roundsStr = (m.actual_rounds != null ? m.actual_rounds : '\u2014')
    + ' / ' + (m.max_rounds != null ? m.max_rounds : '\u2014');

  var metrics = [
    { label: 'Initial \u03B2', value: fmt(m.initial_beta) },
    { label: 'Final \u03B2', value: fmt(m.final_beta) },
    { label: 'Final \u03c1\u0304', value: fmt(q.final_rho_bar) },
    { label: 'JS Drop', value: fmt(q.js_drop) },
    { label: 'Rounds', value: esc(roundsStr) },
  ];

  var h = '<div class="ov-metrics-row">';
  for (var i = 0; i < metrics.length; i++) {
    h += '<div class="ov-metric-card">';
    h += '<div class="ov-metric-value">' + metrics[i].value + '</div>';
    h += '<div class="ov-metric-label">' + metrics[i].label + '</div>';
    h += '</div>';
  }
  h += '</div>';
  return h;
}

/**
 * Build Section 3 — Experiment Config table (3 rows x 2 key-value pairs).
 *
 * @param {object} detail - Run detail object
 * @returns {string} HTML table string
 */
function buildExperimentConfig(detail) {
  var m = detail.manifest || {};
  var fields = extractConfigFields(m);
  var roundsStr = (m.actual_rounds != null ? m.actual_rounds : '\u2014')
    + ' / ' + (m.max_rounds != null ? m.max_rounds : '\u2014');
  var tempStr = '\u2014';
  var dc = detail.debate_config;
  if (dc && dc.debate_setup && dc.debate_setup.temperature != null) {
    tempStr = String(dc.debate_setup.temperature);
  }

  var h = '<div class="ov-subtitle">EXPERIMENT CONFIG</div>';
  h += '<table class="ov-htable">';
  h += '<tr><th>Config</th><td>' + esc(fields.configName) + '</td>';
  h += '<th>Agents</th><td>' + esc(fields.agentsStr) + '</td></tr>';
  h += '<tr><th>Rounds</th><td>' + esc(roundsStr) + '</td>';
  h += '<th>Termination</th><td>' + esc(m.termination_reason || '\u2014') + '</td></tr>';
  h += '<tr><th>Temperature</th><td>' + esc(tempStr) + '</td>';
  h += '<th>Initial \u03B2</th><td>' + fmt(m.initial_beta) + '</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Group flat config items into named groups by key prefix.
 * Unmatched items are placed in an "Other" group.
 *
 * @param {Array} flatItems - Array of {key, value} objects
 * @param {Array} groupDefs - Array of {name, prefixes} definitions
 * @returns {Array} Array of {name, items} groups (non-empty only)
 */
function groupConfigItems(flatItems, groupDefs) {
  var groups = {};
  var groupOrder = [];
  for (var i = 0; i < groupDefs.length; i++) {
    groups[groupDefs[i].name] = [];
    groupOrder.push(groupDefs[i].name);
  }
  groups['Other'] = [];

  for (var j = 0; j < flatItems.length; j++) {
    var item = flatItems[j];
    var matched = false;
    for (var k = 0; k < groupDefs.length; k++) {
      var prefixes = groupDefs[k].prefixes;
      for (var p = 0; p < prefixes.length; p++) {
        if (item.key === prefixes[p] || item.key.indexOf(prefixes[p] + '.') === 0) {
          groups[groupDefs[k].name].push(item);
          matched = true;
          break;
        }
      }
      if (matched) break;
    }
    if (!matched) {
      groups['Other'].push(item);
    }
  }

  var result = [];
  for (var g = 0; g < groupOrder.length; g++) {
    if (groups[groupOrder[g]].length > 0) {
      result.push({ name: groupOrder[g], items: groups[groupOrder[g]] });
    }
  }
  if (groups['Other'].length > 0) {
    result.push({ name: 'Other', items: groups['Other'] });
  }
  return result;
}

/**
 * Build a config section with subgroup cards from a flat config.
 *
 * @param {object} config - Raw config object
 * @param {string} title - Section title
 * @param {string} testId - data-testid attribute value
 * @param {Array} excludeKeys - Keys to exclude
 * @param {Array} groupDefs - Group definitions for groupConfigItems
 * @returns {string} HTML string
 */
function buildConfigGroupSection(config, title, testId, excludeKeys, groupDefs) {
  if (!config || typeof config !== 'object') return '';
  var flat = flattenConfig(config);
  if (flat.length === 0) return '';

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
  if (filtered.length === 0) return '';

  var grouped = groupConfigItems(filtered, groupDefs);
  var h = '<div class="ov-subtitle">' + esc(title) + '</div>';
  h += '<div class="ov-config-groups" data-testid="' + esc(testId) + '">';
  for (var g = 0; g < grouped.length; g++) {
    h += renderGroupCard(grouped[g]);
  }
  h += '</div>';
  return h;
}

/**
 * Render a single config group card with key-value rows.
 *
 * @param {object} group - {name, items} group object
 * @returns {string} HTML string for one group card
 */
function renderGroupCard(group) {
  var h = '<div class="ov-config-group">';
  h += '<div class="ov-config-group-title">' + esc(group.name) + '</div>';
  h += '<div class="ov-config-group-body">';
  for (var i = 0; i < group.items.length; i++) {
    var item = group.items[i];
    var val = item.value;
    var truncated = val.length > 60 ? val.slice(0, 60) + '\u2026' : val;
    var titleAttr = val.length > 60 ? ' title="' + esc(val) + '"' : '';
    h += '<div class="ov-kv-row">';
    h += '<span class="ov-config-key">' + esc(item.key) + '</span>';
    h += '<span class="ov-config-val"' + titleAttr + '>' + esc(truncated) + '</span>';
    h += '</div>';
  }
  h += '</div></div>';
  return h;
}

/**
 * Build Section 4 — Debate Config with semantic subgroup cards.
 *
 * @param {object} debateConfig - Debate config object
 * @returns {string} HTML string
 */
function buildDebateConfigGroups(debateConfig) {
  return buildConfigGroupSection(
    debateConfig, 'DEBATE CONFIG', 'debate-config-grid',
    DEBATE_EXCLUDE, DEBATE_GROUP_DEFS
  );
}

/**
 * Build Section 4b — Scenario Config with semantic subgroup cards.
 *
 * @param {object} scenarioConfig - Scenario config object
 * @returns {string} HTML string
 */
function buildScenarioConfigGroups(scenarioConfig) {
  return buildConfigGroupSection(
    scenarioConfig, 'SCENARIO CONFIG', 'scenario-config-grid',
    SCENARIO_EXCLUDE, SCENARIO_GROUP_DEFS
  );
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
  h += '<div class="ov-capped-table">';
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
  h += '</table></div>';
  return h;
}

/**
 * Build collapsible Macro Environment card with full text.
 *
 * @param {object} scenarioConfig - Scenario config object
 * @returns {string} HTML string
 */
function buildMacroCollapsible(scenarioConfig) {
  if (!scenarioConfig || !scenarioConfig.macro_context) return '';
  var text = String(scenarioConfig.macro_context);
  var h = '<div class="card">';
  h += '<div class="card-header">MACRO ENVIRONMENT <span class="arrow">\u25B6</span></div>';
  h += '<div class="card-body">';
  h += '<pre class="content ov-scroll-box">' + esc(text) + '</pre>';
  h += '</div></div>';
  return h;
}
