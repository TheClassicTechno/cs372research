/**
 * components/financialSignificance.js
 *
 * Pure HTML builder for the cross-ablation financial significance summary
 * table.  Experiments as columns, key financial metrics as rows, with
 * p-value colour coding.
 */

import { esc } from '../utils/dom.js';
import { fmtPvalue, pvalueClass } from '../utils/format.js';

/**
 * Metric display labels and formatting type.
 * Mirrors the config in financialTests.js for consistency.
 */
let METRIC_LABELS = {
  'daily_metrics_excess_return_pct':       { label: 'Excess Return',  type: 'pct' },
  'daily_metrics_total_return_pct':        { label: 'Return %',       type: 'pct' },
  'daily_metrics_annualized_sharpe':       { label: 'Sharpe',         type: 'ratio' },
  'daily_metrics_annualized_sortino':      { label: 'Sortino',        type: 'ratio' },
  'daily_metrics_calmar_ratio':            { label: 'Calmar Ratio',   type: 'ratio' },
  'daily_metrics_annualized_volatility':   { label: 'Volatility',     type: 'pct' },
  'daily_metrics_max_drawdown_pct':        { label: 'Max Drawdown',   type: 'pct' },
  'total_trades':                          { label: 'Num Positions',  type: 'int' },
  'final_cash':                            { label: 'Uninvested Cash', type: 'cash' },
};

/**
 * Format a delta value with sign prefix for compact display.
 */
function fmtSigDelta(val, type) {
  if (val == null) return '\u2014';
  let sign = val >= 0 ? '+' : '\u2212';
  let abs = Math.abs(val);
  if (type === 'pct') return sign + abs.toFixed(2) + '%';
  if (type === 'ratio') return sign + abs.toFixed(3);
  if (type === 'int') return sign + Math.round(abs).toLocaleString();
  if (type === 'cash') return sign + '$' + Math.round(abs).toLocaleString();
  return sign + abs.toFixed(3);
}

/**
 * Return a short display name for an experiment.
 * Strips the "vskarich_" prefix and replaces underscores with spaces.
 */
function shortName(exp) {
  return exp.replace(/^vskarich_/, '').replace(/_/g, ' ');
}

/**
 * Look up label config for a metric key.
 */
function getLabel(metric) {
  if (METRIC_LABELS[metric] !== undefined) return METRIC_LABELS[metric];
  return {
    label: metric.replace('daily_metrics_', '').replace(/_/g, ' '),
    type: 'ratio',
  };
}

/**
 * Build the cross-ablation financial significance table.
 *
 * Accepts the API response object with ``experiments`` and ``metrics``
 * arrays.  Returns an HTML string.
 */
export function buildFinancialSignificanceTable(data) {
  if (data == null) return '';
  let experiments = data.experiments;
  let metrics = data.metrics;
  if (!Array.isArray(experiments) || experiments.length === 0) return '';
  if (!Array.isArray(metrics) || metrics.length === 0) return '';

  let h = '<div class="fin-section" data-testid="financial-significance-table">';
  h += '<div class="section-label">Cross-Ablation Financial Significance (Judge Portfolio)</div>';

  h += '<table class="data-table fin-table">';

  // Header row: Metric + one column per experiment
  h += '<thead><tr><th>Metric</th>';
  for (let i = 0; i < experiments.length; i++) {
    h += '<th class="num-col">' + esc(shortName(experiments[i])) + '</th>';
  }
  h += '</tr></thead>';

  // Body: one row per metric
  h += '<tbody>';
  for (let m = 0; m < metrics.length; m++) {
    let entry = metrics[m];
    let cfg = getLabel(entry.metric);

    h += '<tr>';
    h += '<td class="fin-metric-name">' + esc(cfg.label) + '</td>';

    for (let e = 0; e < experiments.length; e++) {
      let r = entry.results[experiments[e]];
      if (r == null) {
        h += '<td class="num-cell">\u2014</td>';
        continue;
      }
      let pCls = pvalueClass(r.p_value);
      let bold = r.p_value < 0.05 ? ' style="font-weight:bold;"' : '';
      let cell = fmtSigDelta(r.mean_diff, cfg.type) + ' p=' + fmtPvalue(r.p_value);
      if (r.p_value < 0.05) cell += ' *';
      h += '<td class="num-cell ' + pCls + '"' + bold + '>' + esc(cell) + '</td>';
    }

    h += '</tr>';
  }
  h += '</tbody></table></div>';
  return h;
}
