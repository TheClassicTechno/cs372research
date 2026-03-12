import { fetchPortfolio } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

export function loadPortfolioSection(experiment, runId, token) {
  let div = document.getElementById('portfolio-section');
  fetchPortfolio(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }

      let lastConsensus = data[data.length - 1].consensus || {};
      let tickers = Object.keys(lastConsensus).sort(function (a, b) {
        return (lastConsensus[b] || 0) - (lastConsensus[a] || 0);
      });

      let ptCfg = T('portfolio_trajectory');
      let h = '<div class="section-label">' + esc(ptCfg.title) + '</div>';
      h += '<table class="data-table"><tr><th>' + esc(ptCfg.columns[0]) + '</th>';
      for (let i = 0; i < data.length; i++) {
        h += '<th>R' + data[i].round + '</th>';
      }
      h += '</tr>';
      for (let t = 0; t < tickers.length; t++) {
        h += '<tr><td>' + esc(tickers[t]) + '</td>';
        for (let i = 0; i < data.length; i++) {
          let w = data[i].consensus[tickers[t]];
          h += '<td>' + (w != null ? fmt(w) : '\u2014') + '</td>';
        }
        h += '</tr>';
      }
      h += '</table>';

      div.innerHTML = buildCard(T('cards').portfolio_trajectory, h, false);
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
