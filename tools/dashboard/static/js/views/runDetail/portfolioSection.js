/**
 * views/runDetail/portfolioSection.js
 *
 * Loads and renders the consensus portfolio weight trajectory across rounds.
 */
import { fetchPortfolio } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

/** Fetches portfolio data and renders the per-ticker weight trajectory table. */
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
      h += data.map(function (d) { return '<th>R' + d.round + '</th>'; }).join('');
      h += '</tr>';
      tickers.forEach(function (ticker) {
        h += '<tr><td>' + esc(ticker) + '</td>';
        data.forEach(function (d) {
          let w = d.consensus[ticker];
          h += '<td>' + (w != null ? fmt(w) : '\u2014') + '</td>';
        });
        h += '</tr>';
      });
      h += '</table>';

      div.innerHTML = '<div data-testid="portfolio-content">' + buildCard(T('cards').portfolio_trajectory, h, false) + '</div>';
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
