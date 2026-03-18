/**
 * views/runDetail/overviewSection.js
 *
 * Builds the run overview panel and collapsible config detail cards.
 */
import { esc } from '../../utils/dom.js';
import { fmt, numFmt } from '../../utils/format.js';
import { flattenConfig } from '../../utils/config.js';
import { buildCard } from '../../components/card.js';
import { T } from '../../utils/labels.js';

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
  let m = detail.manifest || {};
  let q = detail.quality || {};

  let sec = T('sections');
  let html = '<div class="run-overview">';
  html += '<div class="ov-title">' + esc(sec.run_overview) + '</div>';

  html += buildIdentityTable(detail, runId, experiment);
  html += buildConfigTable(m);
  html += buildOverviewMetricsTable(m, q);

  if (q.reasoning_collapse) {
    html += '<div class="ov-warn">REASONING COLLAPSE DETECTED</div>';
  }

  // Judge portfolio placeholder
  html += '<div id="judge-portfolio-section"></div>';

  html += '</div>';
  return html;
}

/**
 * Build the run identity table (run ID, experiment, models, status).
 *
 * @param {object} detail - Full run detail from API
 * @param {string} runId - Run ID
 * @param {string} experiment - Experiment name
 * @returns {string} HTML table string
 */
function buildIdentityTable(detail, runId, experiment) {
  let m = detail.manifest || {};
  let statusCls = '';
  if (detail.status === 'complete' || detail.status === 'partial') {
    statusCls = ' class="status-ok"';
  } else if (detail.status === 'running') {
    statusCls = ' class="status-running"';
  } else if (detail.status === 'failed') {
    statusCls = ' class="status-failed"';
  }

  let idCfg = T('overview_identity');
  let h = '<table class="ov-htable">';
  h += '<tr>' + idCfg.columns.map(function (col) { return '<th>' + esc(col) + '</th>'; }).join('') + '</tr>';
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
 * Build the config and agents summary table.
 *
 * @param {object} manifest - Run manifest object
 * @returns {string} HTML table string
 */
function buildConfigTable(manifest) {
  let cfgLabels = T('overview_config');
  let fields = extractConfigFields(manifest);
  let h = '<table class="ov-htable" style="table-layout:auto;">';
  h += '<tr><th style="width:1%;white-space:nowrap">' + esc(cfgLabels.rows.config) + '</th><td>' + esc(fields.configName) + '</td></tr>';
  h += '<tr><th style="width:1%;white-space:nowrap">' + esc(cfgLabels.rows.agents) + '</th><td>' + esc(fields.agentsStr) + '</td></tr>';
  h += '</table>';
  return h;
}

/**
 * Build the key metrics table (tickers, rounds, termination, beta, quality).
 *
 * @param {object} manifest - Run manifest object
 * @param {object} quality - Run quality metrics object
 * @returns {string} HTML table string
 */
function buildOverviewMetricsTable(manifest, quality) {
  let tickersStr = '\u2014';
  if (manifest.ticker_universe && manifest.ticker_universe.length > 0) {
    tickersStr = manifest.ticker_universe.filter(function (t) { return t !== '_CASH_'; }).join(', ');
  }
  let roundsStr = (manifest.actual_rounds != null ? manifest.actual_rounds : '\u2014')
    + ' / ' + (manifest.max_rounds != null ? manifest.max_rounds : '\u2014');

  let metCfg = T('overview_metrics');
  let h = '<table class="ov-htable">';
  h += '<tr>' + metCfg.columns.map(function (col) { return '<th>' + col + '</th>'; }).join('') + '</tr>';
  h += '<tr>';
  h += '<td>' + esc(tickersStr) + '</td>';
  h += '<td>' + esc(roundsStr) + '</td>';
  h += '<td>' + esc(manifest.termination_reason || '\u2014') + '</td>';
  h += '<td>' + fmt(manifest.initial_beta) + '</td>';
  h += '<td>' + fmt(manifest.final_beta) + '</td>';
  h += '<td>' + fmt(quality.final_rho_bar) + '</td>';
  h += '<td>' + fmt(quality.js_drop) + '</td>';
  h += '</tr></table>';
  return h;
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
  let cards = T('cards');
  let html = '';

  // Debate config groups card
  let debateHtml = buildConfigGroupsHtml(
    detail.debate_config, 'debate-config-grid',
    DEBATE_EXCLUDE, DEBATE_GROUP_DEFS
  );
  if (debateHtml) {
    html += buildCard(cards.debate_config, debateHtml);
  }

  // Scenario config groups card
  let scenarioHtml = buildConfigGroupsHtml(
    detail.scenario_config, 'scenario-config-grid',
    SCENARIO_EXCLUDE, SCENARIO_GROUP_DEFS
  );
  if (scenarioHtml) {
    html += buildCard(cards.scenario_config, scenarioHtml);
  }

  // Ticker performance card
  let tickerHtml = buildTickerPerfHtml(detail.ticker_performance);
  if (tickerHtml) {
    html += buildCard(cards.ticker_performance, tickerHtml);
  }

  // Macro environment card
  let macroHtml = buildMacroHtml(detail.scenario_config);
  if (macroHtml) {
    html += buildCard(cards.macro_environment, macroHtml);
  }

  return html;
}

// ---- Private helpers ----

/** Keys excluded from debate config grid (shown in summary/metrics). */
const DEBATE_EXCLUDE = [
  'agents',
  'debate_setup.llm_model',
  'debate_setup.experiment_name',
  'debate_setup.max_rounds',
];

/** Keys excluded from scenario config grid. */
const SCENARIO_EXCLUDE = [
  'tickers',
  'macro_context',
  'output_dir',
];

/** Semantic group definitions for debate config parameters. */
const DEBATE_GROUP_DEFS = [
  { name: 'Agent Setup', prefixes: ['agent'] },
  { name: 'Broker', prefixes: ['broker'] },
  { name: 'Dataset', prefixes: ['dataset'] },
  { name: 'Judge', prefixes: ['judge'] },
  { name: 'PID Settings', prefixes: ['pid'] },
  { name: 'Runtime', prefixes: ['runtime', 'debate_setup'] },
];

/** Semantic group definitions for scenario config parameters. */
const SCENARIO_GROUP_DEFS = [
  { name: 'Allocation Constraints', prefixes: ['allocation', 'constraint'] },
];

/**
 * Extract config and agent display strings from manifest.
 *
 * @param {object} m - Manifest object
 * @returns {{configName: string, agentsStr: string}} Display strings
 */
function extractConfigFields(m) {
  let configName = '\u2014';
  if (m.config_paths && m.config_paths.length > 0) {
    let cp = m.config_paths[0].replace(/\\/g, '/').split('/');
    configName = cp[cp.length - 1].replace(/\.yaml$/, '').replace(/\.yml$/, '');
  }
  let agentsStr = '\u2014';
  if (m.agent_profiles && typeof m.agent_profiles === 'object') {
    let vals = Object.entries(m.agent_profiles);
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
  let flat = flattenConfig(config);
  if (flat.length === 0) return '';

  let excludeSet = {};
  excludeKeys.forEach(function (k) { excludeSet[k] = true; });
  let filtered = flat.filter(function (item) { return !excludeSet[item.key]; });
  if (filtered.length === 0) return '';

  let grouped = groupConfigItems(filtered, groupDefs);
  let h = '<div class="ov-config-groups" data-testid="' + esc(testId) + '">';
  grouped.forEach(function (grp) { h += renderGroupCard(grp); });
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
  let groups = {};
  let groupOrder = groupDefs.map(function (gd) { return gd.name; });
  groupOrder.forEach(function (name) { groups[name] = []; });
  groups['Other'] = [];

  flatItems.forEach(function (item) {
    let matched = groupDefs.some(function (gd) {
      return gd.prefixes.some(function (prefix) {
        if (item.key === prefix || item.key.indexOf(prefix + '.') === 0) {
          groups[gd.name].push(item);
          return true;
        }
        return false;
      });
    });
    if (!matched) {
      groups['Other'].push(item);
    }
  });

  let result = groupOrder.filter(function (name) { return groups[name].length > 0; }).map(function (name) {
    return { name: name, items: groups[name] };
  });
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
  let h = '<div class="ov-config-group">';
  h += '<div class="ov-config-group-title">' + esc(group.name) + '</div>';
  h += '<div class="ov-config-group-body">';
  for (const item of group.items) {
    let val = item.value;
    let truncated = val.length > 60 ? val.slice(0, 60) + '\u2026' : val;
    let titleAttr = val.length > 60 ? ' title="' + esc(val) + '"' : '';
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
  let tpCfg = T('ticker_perf');
  let h = '<table class="data-table" data-testid="ticker-perf-table">';
  h += '<tr>' + tpCfg.columns.map(function (col) { return '<th>' + esc(col) + '</th>'; }).join('') + '</tr>';
  for (const t of tickerPerf) {
    let cls = t.pct_change >= 0 ? 'perf-profit' : 'perf-loss';
    let sign = t.pct_change >= 0 ? '+' : '';
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
