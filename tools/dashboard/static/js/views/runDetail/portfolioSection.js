import { fetchPortfolio } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

export function loadPortfolioSection(experiment, runId, token) {
  var div = document.getElementById('portfolio-section');
  fetchPortfolio(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }

      var lastConsensus = data[data.length - 1].consensus || {};
      var tickers = Object.keys(lastConsensus).sort(function (a, b) {
        return (lastConsensus[b] || 0) - (lastConsensus[a] || 0);
      });

      var ptCfg = T('portfolio_trajectory');
      var h = '<div class="section-label">' + esc(ptCfg.title) + '</div>';
      h += '<table class="data-table"><tr><th>' + esc(ptCfg.columns[0]) + '</th>';
      for (var i = 0; i < data.length; i++) {
        h += '<th>R' + data[i].round + '</th>';
      }
      h += '</tr>';
      for (var t = 0; t < tickers.length; t++) {
        h += '<tr><td>' + esc(tickers[t]) + '</td>';
        for (var i = 0; i < data.length; i++) {
          var w = data[i].consensus[tickers[t]];
          h += '<td>' + (w != null ? fmt(w) : '\u2014') + '</td>';
        }
        h += '</tr>';
      }
      h += '</table>';

      div.innerHTML = buildCard(T('cards').portfolio_trajectory, h, false);
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
