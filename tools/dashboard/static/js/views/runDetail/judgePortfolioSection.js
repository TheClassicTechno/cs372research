import { fetchPortfolio, fetchPerformance } from '../../api/runs.js';
import { buildSimpleAllocTable } from '../../components/table.js';
import { esc } from '../../utils/dom.js';
import { fmtPct, numFmt } from '../../utils/format.js';
import { appState } from '../../state.js';

export function loadJudgePortfolio(experiment, runId, finalPortfolio, manifest, token) {
  var div = document.getElementById('judge-portfolio-section');
  if (!div) return;
  if (!finalPortfolio || Object.keys(finalPortfolio).length === 0) {
    div.innerHTML = '';
    return;
  }

  var h = '<div class="ov-title" style="margin-top:16px;">FINAL ALLOCATIONS</div>';
  h += '<div id="judge-portfolio-layout" style="display:flex;gap:24px;align-items:flex-start;">';
  h += '<div id="judge-alloc-wrap"><span style="color:#666;font-size:0.85em;">Loading allocations...</span></div>';
  h += '<div id="perf-metrics"><span style="color:#666;font-size:0.85em;">Loading performance...</span></div>';
  h += '</div>';
  div.innerHTML = h;

  var m = manifest || {};
  var profileMap = (m.agent_profiles && typeof m.agent_profiles === 'object') ? m.agent_profiles : null;
  function agentLabel(role) {
    if (profileMap && profileMap[role]) return profileMap[role];
    return role;
  }

  fetchPortfolio(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('judge-alloc-wrap');
      if (!wrap) return;
      if (!data || data.length === 0) {
        wrap.innerHTML = buildSimpleAllocTable(finalPortfolio);
        return;
      }
      var last = data[data.length - 1];
      var agents = last.allocations || {};
      var agentNames = Object.keys(agents).sort();

      var tickers = Object.keys(finalPortfolio).sort(function (a, b) {
        return (finalPortfolio[b] || 0) - (finalPortfolio[a] || 0);
      });

      var th = '<table class="data-table" id="judge-alloc-table">';
      th += '<tr><th></th>';
      for (var a = 0; a < agentNames.length; a++) {
        th += '<th>' + esc(agentLabel(agentNames[a]).toUpperCase()) + '</th>';
      }
      th += '<th>JUDGE</th></tr>';

      for (var t = 0; t < tickers.length; t++) {
        var ticker = tickers[t];
        th += '<tr><td style="font-weight:600;">' + esc(ticker) + '</td>';
        for (var a = 0; a < agentNames.length; a++) {
          var w = agents[agentNames[a]] ? agents[agentNames[a]][ticker] : null;
          th += '<td>' + fmtPct(w) + '</td>';
        }
        th += '<td style="font-weight:600;">' + fmtPct(finalPortfolio[ticker]) + '</td>';
        th += '</tr>';
      }
      th += '</table>';
      wrap.innerHTML = th;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('judge-alloc-wrap');
      if (wrap) wrap.innerHTML = buildSimpleAllocTable(finalPortfolio);
    });

  fetchPerformance(experiment, runId)
    .then(function (perf) {
      if (appState.viewToken !== token) return;
      var perfDiv = document.getElementById('perf-metrics');
      if (!perfDiv) return;

      if (perf.error) {
        perfDiv.innerHTML = '<span style="color:#666;font-size:0.85em;">' + esc(perf.error) + '</span>';
        return;
      }

      var profitCls = perf.profit >= 0 ? 'perf-profit' : 'perf-loss';
      var profitSign = perf.profit >= 0 ? '+' : '';
      var returnPct = perf.return_pct;

      var ph = '<table class="data-table" id="perf-table">';
      ph += '<tr><th>Metric</th><th>Value</th></tr>';
      ph += '<tr><td>Initial Capital</td><td>$' + numFmt(perf.initial_capital) + '</td></tr>';
      ph += '<tr><td>Final Value</td><td class="' + profitCls + '">$' + numFmt(perf.final_value) + '</td></tr>';
      ph += '<tr><td>Profit/Loss</td><td class="' + profitCls + '">' + profitSign + '$' + numFmt(Math.abs(perf.profit)) + '</td></tr>';
      ph += '<tr><td>Return</td><td class="' + profitCls + '">' + profitSign + returnPct.toFixed(2) + '%</td></tr>';
      ph += '</table>';
      perfDiv.innerHTML = ph;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var perfDiv = document.getElementById('perf-metrics');
      if (perfDiv) perfDiv.innerHTML = '<span style="color:#666;font-size:0.85em;">Performance data unavailable</span>';
    });
}
