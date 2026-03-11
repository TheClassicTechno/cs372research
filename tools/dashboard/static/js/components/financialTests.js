/**
 * components/financialTests.js
 *
 * Pure HTML builder for the financial metrics paired t-test comparison
 * table.  Renders per-metric mean +/- SEM, 95% CI, and p-values
 * matching the output format of scripts/compare_multiple_configs.py.
 */

import { esc } from '../utils/dom.js';
import { fmt, fmtPvalue, pvalueClass } from '../utils/format.js';

/**
 * Map a flattened metric key to a display label.
 * Matches the formatting from scripts/compare_multiple_configs.py.
 * Returns a human-readable string.
 */
function financialMetricLabel(metric) {
  const labels = {
    'daily_metrics_excess_return_pct': 'Excess Return % (v SP500)',
    'daily_metrics_annualized_sharpe': 'Sharpe',
    'daily_metrics_annualized_volatility': 'Volatility',
    'daily_metrics_max_drawdown_pct': 'Max Drawdown %',
    'daily_metrics_total_return_pct': 'Return %',
    'daily_metrics_annualized_sortino': 'Sortino',
    'daily_metrics_calmar_ratio': 'Calmar Ratio',
    'total_trades': 'Num Positions',
    'final_cash': 'Uninvested Cash',
  };
  if (labels[metric] !== undefined) return labels[metric];
  return metric.replace('daily_metrics_', '').replace(/_/g, ' ');
}

/**
 * Format a metric value for display based on the metric key.
 * Returns a formatted string.
 */
function fmtMetricVal(metric, mean, sem) {
  if (metric === 'final_cash') {
    return fmt(mean / 1000, 2) + 'k \u00b1 ' + fmt(sem / 1000, 2) + 'k';
  }
  if (metric.indexOf('pct') !== -1 || metric.indexOf('return') !== -1) {
    return fmt(mean, 2) + '% \u00b1 ' + fmt(sem, 2) + '%';
  }
  if (metric === 'total_trades') {
    return fmt(mean, 1) + ' \u00b1 ' + fmt(sem, 1);
  }
  return fmt(mean, 4) + ' \u00b1 ' + fmt(sem, 4);
}

/**
 * Format a CI range for display based on the metric key.
 * Returns a formatted string like "[+1.23%, +4.56%]".
 */
function fmtCiRange(metric, ci) {
  const lo = ci[0];
  const hi = ci[1];
  const sign = function (v) { return v >= 0 ? '+' : ''; };
  if (metric === 'final_cash') {
    return '[' + sign(lo) + fmt(lo / 1000, 2) + 'k, ' + sign(hi) + fmt(hi / 1000, 2) + 'k]';
  }
  if (metric.indexOf('pct') !== -1 || metric.indexOf('return') !== -1) {
    return '[' + sign(lo) + fmt(lo, 2) + '%, ' + sign(hi) + fmt(hi, 2) + '%]';
  }
  if (metric === 'total_trades') {
    return '[' + sign(lo) + fmt(lo, 1) + ', ' + sign(hi) + fmt(hi, 1) + ']';
  }
  return '[' + sign(lo) + fmt(lo, 4) + ', ' + sign(hi) + fmt(hi, 4) + ']';
}

/**
 * Build the financial metrics paired t-test section for an experiment.
 * Renders a comparison table matching compare_multiple_configs.py format.
 * Accepts the financial test result object from the API.
 * Returns an HTML string.
 */
export function buildFinancialTestsSection(data) {
  if (data === undefined || data === null) return '';
  if (data.error !== undefined) {
    return '<div class="section-label">Financial Metrics Comparison</div>'
      + '<p>' + esc(data.error) + '</p>';
  }

  const metrics = data.metrics;
  if (!Array.isArray(metrics) || metrics.length === 0) return '';

  let h = '<div class="section-label">Financial Metrics Comparison (Paired t-Test, N='
    + esc(String(data.n_paired)) + ')</div>';

  h += '<table class="data-table" data-testid="financial-tests-table">';
  h += '<tr><th>Metric</th>';
  h += '<th>(A) ' + esc(data.config_a) + '</th>';
  h += '<th>(B) ' + esc(data.config_b) + '</th>';
  h += '<th>95% CI (Diff B\u2212A)</th>';
  h += '<th>p-value</th></tr>';

  for (let i = 0; i < metrics.length; i++) {
    const m = metrics[i];
    const pClass = pvalueClass(m.p_value);
    h += '<tr>';
    h += '<td>' + esc(financialMetricLabel(m.metric)) + '</td>';
    h += '<td style="text-align:right;">' + fmtMetricVal(m.metric, m.a_mean, m.a_sem) + '</td>';
    h += '<td style="text-align:right;">' + fmtMetricVal(m.metric, m.b_mean, m.b_sem) + '</td>';
    h += '<td style="text-align:right;">' + fmtCiRange(m.metric, m.ci_95) + '</td>';
    h += '<td class="' + pClass + '" style="text-align:right;">' + fmtPvalue(m.p_value) + '</td>';
    h += '</tr>';
  }
  h += '</table>';
  return h;
}
