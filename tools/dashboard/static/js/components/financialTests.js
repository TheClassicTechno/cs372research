/**
 * components/financialTests.js
 *
 * Pure HTML builder for the financial metrics paired t-test comparison.
 * Renders a three-part layout: key improvements summary, grouped metrics
 * table, and CI tooltips for statistical detail.
 */

import { esc } from '../utils/dom.js';
import { fmtPvalue, pvalueClass } from '../utils/format.js';

/**
 * Metric display configuration: label, group, and formatting type.
 * Returns {label, group, type} for a given metric key.
 */
var METRIC_CONFIG = {
  'daily_metrics_excess_return_pct': { label: 'Excess Return vs S&P500', group: 'performance', type: 'pct' },
  'daily_metrics_total_return_pct':  { label: 'Return %',               group: 'performance', type: 'pct' },
  'daily_metrics_annualized_sharpe': { label: 'Sharpe',                 group: 'performance', type: 'ratio' },
  'daily_metrics_annualized_sortino':{ label: 'Sortino',                group: 'performance', type: 'ratio' },
  'daily_metrics_calmar_ratio':      { label: 'Calmar Ratio',           group: 'performance', type: 'ratio' },
  'daily_metrics_annualized_volatility': { label: 'Volatility',         group: 'risk',        type: 'pct' },
  'daily_metrics_max_drawdown_pct':  { label: 'Max Drawdown',           group: 'risk',        type: 'pct' },
  'total_trades':                    { label: 'Num Positions',           group: 'portfolio',   type: 'int' },
  'final_cash':                      { label: 'Uninvested Cash',         group: 'portfolio',   type: 'cash' },
};

/** Ordered metric keys for the table display. */
var METRIC_ORDER = [
  'daily_metrics_excess_return_pct',
  'daily_metrics_total_return_pct',
  'daily_metrics_annualized_sharpe',
  'daily_metrics_annualized_sortino',
  'daily_metrics_calmar_ratio',
  'daily_metrics_annualized_volatility',
  'daily_metrics_max_drawdown_pct',
  'total_trades',
  'final_cash',
];

/** Group labels for subheader rows. */
var GROUP_LABELS = {
  'performance': 'Performance',
  'risk': 'Risk',
  'portfolio': 'Portfolio Behavior',
};

/**
 * Look up config for a metric key.
 * Returns the config object or a generated fallback.
 */
function getMetricConfig(metric) {
  if (METRIC_CONFIG[metric] !== undefined) return METRIC_CONFIG[metric];
  return {
    label: metric.replace('daily_metrics_', '').replace(/_/g, ' '),
    group: 'other',
    type: 'ratio',
  };
}

/**
 * Format a single numeric value for display (no SEM).
 * Returns a formatted string based on metric type.
 */
function fmtValue(val, type) {
  if (val == null) return '\u2014';
  if (type === 'pct') return Number(val).toFixed(2) + '%';
  if (type === 'ratio') return Number(val).toFixed(3);
  if (type === 'int') return Math.round(val).toLocaleString();
  if (type === 'cash') return '$' + Math.round(val).toLocaleString();
  return Number(val).toFixed(3);
}

/**
 * Format a delta value with sign prefix.
 * Returns a string like "+0.91%" or "\u22120.133".
 */
function fmtFinDelta(val, type) {
  if (val == null) return '\u2014';
  var sign = val >= 0 ? '+' : '\u2212';
  var abs = Math.abs(val);
  if (type === 'pct') return sign + abs.toFixed(2) + '%';
  if (type === 'ratio') return sign + abs.toFixed(3);
  if (type === 'int') return sign + Math.round(abs).toLocaleString();
  if (type === 'cash') return sign + '$' + Math.round(abs).toLocaleString();
  return sign + abs.toFixed(3);
}

/**
 * Determine if a positive delta is "good" for this metric.
 * For drawdown and volatility, lower is better.
 */
function isPositiveGood(metric) {
  if (metric === 'daily_metrics_annualized_volatility') return false;
  if (metric === 'daily_metrics_max_drawdown_pct') return false;
  return true;
}

/**
 * Return a CSS class for the delta value based on improvement direction.
 * Returns 'perf-profit' for improvement, '' otherwise.
 */
function deltaClass(metric, diff) {
  if (diff === 0 || diff == null) return '';
  var positive = diff > 0;
  if (!isPositiveGood(metric)) positive = !positive;
  return positive ? 'perf-profit' : '';
}

/**
 * Format a CI range as a tooltip string.
 * Returns e.g. "95% CI: [+0.12%, +1.70%]".
 */
function fmtCiTooltip(type, ci) {
  var lo = ci[0];
  var hi = ci[1];
  return '95% CI: [' + fmtFinDelta(lo, type) + ', ' + fmtFinDelta(hi, type) + ']';
}

/** Key metrics to highlight in the summary box. */
var SUMMARY_METRICS = [
  'daily_metrics_excess_return_pct',
  'daily_metrics_annualized_sharpe',
  'daily_metrics_annualized_sortino',
  'daily_metrics_calmar_ratio',
];

/**
 * Build the key improvements summary box.
 * Shows only metrics with positive improvement.
 * Returns an HTML string.
 */
function buildSummaryBox(metricsMap) {
  var items = [];
  for (var i = 0; i < SUMMARY_METRICS.length; i++) {
    var key = SUMMARY_METRICS[i];
    var m = metricsMap[key];
    if (m === undefined) continue;
    var cfg = getMetricConfig(key);
    var diff = m.mean_diff;
    if (diff == null) continue;
    var improved = isPositiveGood(key) ? diff > 0 : diff < 0;
    if (!improved) continue;
    items.push({ label: cfg.label, value: fmtFinDelta(diff, cfg.type) });
  }

  if (items.length === 0) return '';

  var h = '<div class="fin-summary-box" data-testid="financial-summary-box">';
  h += '<div class="fin-summary-title">Key Improvements from Intervention</div>';
  h += '<div class="fin-summary-items">';
  for (var j = 0; j < items.length; j++) {
    h += '<div class="fin-summary-item">';
    h += '<div class="fin-summary-label">' + esc(items[j].label) + '</div>';
    h += '<div class="fin-summary-value perf-profit">' + esc(items[j].value) + '</div>';
    h += '</div>';
  }
  h += '</div></div>';
  return h;
}

/**
 * Build the grouped metrics table.
 * Returns an HTML string with subheader rows for each metric group.
 */
function buildMetricsTable(data, metricsMap) {
  var h = '<table class="data-table fin-table" data-testid="financial-tests-table">';
  h += '<thead><tr>';
  h += '<th>Metric</th>';
  h += '<th class="num-col">Baseline</th>';
  h += '<th class="num-col">Intervention</th>';
  h += '<th class="num-col">\u0394 Difference</th>';
  h += '<th class="num-col">p-value</th>';
  h += '</tr></thead>';
  h += '<tbody>';

  var currentGroup = '';

  for (var i = 0; i < METRIC_ORDER.length; i++) {
    var key = METRIC_ORDER[i];
    var m = metricsMap[key];
    if (m === undefined) continue;

    var cfg = getMetricConfig(key);

    if (cfg.group !== currentGroup) {
      currentGroup = cfg.group;
      var groupLabel = GROUP_LABELS[currentGroup];
      if (groupLabel === undefined) groupLabel = currentGroup;
      h += '<tr class="fin-group-header"><td colspan="5">' + esc(groupLabel) + '</td></tr>';
    }

    var pClass = pvalueClass(m.p_value);
    var dClass = deltaClass(key, m.mean_diff);
    var ciTip = fmtCiTooltip(cfg.type, m.ci_95);

    h += '<tr>';
    h += '<td class="fin-metric-name">' + esc(cfg.label) + '</td>';
    h += '<td class="num-cell">' + fmtValue(m.a_mean, cfg.type) + '</td>';
    h += '<td class="num-cell">' + fmtValue(m.b_mean, cfg.type) + '</td>';
    h += '<td class="num-cell ' + dClass + '" title="' + esc(ciTip) + '">'
      + fmtFinDelta(m.mean_diff, cfg.type) + '</td>';
    h += '<td class="num-cell ' + pClass + '">' + fmtPvalue(m.p_value) + '</td>';
    h += '</tr>';
  }

  h += '</tbody></table>';
  h += '<div class="fin-ci-hint">Hover over \u0394 values to see 95% confidence intervals</div>';
  return h;
}

/**
 * Build the financial metrics paired t-test section for an experiment.
 * Renders a three-part layout: title, summary box, and grouped table.
 * Accepts the financial test result object from the API.
 * Returns an HTML string.
 */
export function buildFinancialTestsSection(data) {
  if (data === undefined || data === null) return '';
  if (data.pending === true) {
    return '<div class="section-label">Financial Metrics Comparison</div>'
      + '<p class="status-incomplete">' + esc(data.message) + '</p>';
  }
  if (data.error !== undefined) {
    return '<div class="section-label">Financial Metrics Comparison</div>'
      + '<p>' + esc(data.error) + '</p>';
  }

  var metrics = data.metrics;
  if (!Array.isArray(metrics) || metrics.length === 0) return '';

  var metricsMap = {};
  for (var i = 0; i < metrics.length; i++) {
    metricsMap[metrics[i].metric] = metrics[i];
  }

  var sourceLabel = data.source === 'mean_revisions'
    ? 'Mean Agent Revisions' : 'Judge Portfolio';

  var h = '<div class="fin-section" data-testid="financial-tests-section">';

  h += '<div class="section-label">Financial Performance Impact \u2014 '
    + esc(sourceLabel) + ' (Paired t-test, N\u2009=\u2009'
    + esc(String(data.n_paired)) + ')</div>';
  h += '<div class="fin-subtitle">Effect of intervention ('
    + esc(data.config_b) + ') vs baseline (' + esc(data.config_a) + ')</div>';

  h += buildSummaryBox(metricsMap);
  h += buildMetricsTable(data, metricsMap);

  h += '</div>';
  return h;
}
