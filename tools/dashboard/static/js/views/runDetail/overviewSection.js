import { esc } from '../../utils/dom.js';
import { fmt, numFmt } from '../../utils/format.js';
import { flattenConfig } from '../../utils/config.js';
import { buildCard } from '../../components/card.js';

/**
 * Build the RUN OVERVIEW metadata panel.
 * Compact summary: identity, config/agents, key metrics.
 *
 * @param {object} detail - Full run detail from API
 * @param {string} experiment - Experiment name
 * @param {string} runId - Run ID
 * @returns {string} HTML string
 */
export function buildOverviewPanel(detail, experiment, runId) {
  var m = detail.manifest || {};
  var q = detail.quality || {};

  var html = '<div class="run-overview">';
  html += '<div class="ov-title">RUN OVERVIEW</div>';

  // --- Table 1: Run identity ---
  var statusCls = '';
  if (detail.status === 'complete' || detail.status === 'partial') {
    statusCls = ' class="status-ok"';
  } else if (detail.status === 'running') {
    statusCls = ' class="status-running"';
  } else if (detail.status === 'failed') {
    statusCls = ' class="status-failed"';
  }

  html += '<table class="ov-htable">';
  html += '<tr><th>Run ID</th><th>Experiment</th><th>Model</th><th>CRIT Model</th><th>Status</th></tr>';
  html += '<tr>';
  html += '<td>' + esc(runId) + '</td>';
  html += '<td>' + esc(experiment) + '</td>';
  html += '<td>' + esc(m.model_name || '\u2014') + '</td>';
  html += '<td>' + esc(m.crit_model_name || '\u2014') + '</td>';
  html += '<td' + statusCls + '>' + esc(detail.status) + '</td>';
  html += '</tr></table>';

  // --- Table 2: Config and agents ---
  var fields = extractConfigFields(m);
  html += '<table class="ov-htable" style="table-layout:auto;">';
  html += '<tr><th style="width:1%;white-space:nowrap">Config</th><td>' + esc(fields.configName) + '</td></tr>';
  html += '<tr><th style="width:1%;white-space:nowrap">Agents</th><td>' + esc(fields.agentsStr) + '</td></tr>';
  html += '</table>';

  // --- Table 3: Key metrics ---
  var tickersStr = '\u2014';
  if (m.ticker_universe && m.ticker_universe.length > 0) {
    tickersStr = m.ticker_universe.filter(function (t) { return t !== '_CASH_'; }).join(', ');
  }
  var roundsStr = (m.actual_rounds != null ? m.actual_rounds : '\u2014')
    + ' / ' + (m.max_rounds != null ? m.max_rounds : '\u2014');

  html += '<table class="ov-htable">';
  html += '<tr><th>Tickers</th><th>Rounds</th><th>Termination</th><th>Initial \u03B2</th><th>Final \u03B2</th><th>Final <span style="text-decoration:overline">\u03c1</span></th><th>JS Drop</th></tr>';
  html += '<tr>';
  html += '<td>' + esc(tickersStr) + '</td>';
  html += '<td>' + esc(roundsStr) + '</td>';
  html += '<td>' + esc(m.termination_reason || '\u2014') + '</td>';
  html += '<td>' + fmt(m.initial_beta) + '</td>';
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

/**
 * Build collapsible config detail cards (debate config, scenario config,
 * ticker performance, macro environment).
 * These appear below the overview as separate collapsible sections.
 *
 * @param {object} detail - Full run detail from API
 * @returns {string} HTML string of collapsible cards
 */
export function buildConfigCards(detail) {
  var html = '';

  // Debate config groups card
  var debateHtml = buildConfigGroupsHtml(
    detail.debate_config, 'debate-config-grid',
    DEBATE_EXCLUDE, DEBATE_GROUP_DEFS
  );
  if (debateHtml) {
    html += buildCard('Debate Config', debateHtml);
  }

  // Scenario config groups card
  var scenarioHtml = buildConfigGroupsHtml(
    detail.scenario_config, 'scenario-config-grid',
    SCENARIO_EXCLUDE, SCENARIO_GROUP_DEFS
  );
  if (scenarioHtml) {
    html += buildCard('Scenario Config', scenarioHtml);
  }

  // Ticker performance card
  var tickerHtml = buildTickerPerfHtml(detail.ticker_performance);
  if (tickerHtml) {
    html += buildCard('Ticker Performance', tickerHtml);
  }

  // Macro environment card
  var macroHtml = buildMacroHtml(detail.scenario_config);
  if (macroHtml) {
    html += buildCard('Macro Environment', macroHtml);
  }

  return html;
}

// ---- Private helpers ----

/** Keys excluded from debate config grid (shown in summary/metrics). */
var DEBATE_EXCLUDE = [
  'agents',
  'debate_setup.llm_model',
  'debate_setup.experiment_name',
  'debate_setup.max_rounds',
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
 * Build config groups HTML content (for use inside a card body).
 *
 * @param {object} config - Raw config object
 * @param {string} testId - data-testid attribute value
 * @param {Array} excludeKeys - Keys to exclude
 * @param {Array} groupDefs - Group definitions
 * @returns {string} HTML string or empty string
 */
function buildConfigGroupsHtml(config, testId, excludeKeys, groupDefs) {
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
  var h = '<div class="ov-config-groups" data-testid="' + esc(testId) + '">';
  for (var g = 0; g < grouped.length; g++) {
    h += renderGroupCard(grouped[g]);
  }
  h += '</div>';
  return h;
}

/**
 * Group flat config items into named groups by key prefix.
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
 * Build ticker performance HTML content (for use inside a card body).
 *
 * @param {Array} tickerPerf - Array of {ticker, open, close, pct_change}
 * @returns {string} HTML string or empty string
 */
function buildTickerPerfHtml(tickerPerf) {
  if (!tickerPerf || tickerPerf.length === 0) return '';
  var h = '<table class="data-table" data-testid="ticker-perf-table">';
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

/**
 * Build macro environment HTML content (for use inside a card body).
 *
 * @param {object} scenarioConfig - Scenario config object
 * @returns {string} HTML string or empty string
 */
function buildMacroHtml(scenarioConfig) {
  if (!scenarioConfig || !scenarioConfig.macro_context) return '';
  return '<pre class="content ov-scroll-box">' + esc(String(scenarioConfig.macro_context)) + '</pre>';
}
