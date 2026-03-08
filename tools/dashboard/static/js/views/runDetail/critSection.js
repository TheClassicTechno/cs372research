import { fetchCRIT } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildCRITChart } from '../../components/charts.js';
import { appState } from '../../state.js';

export function loadCRITSection(experiment, runId, token) {
  var div = document.getElementById('crit-section');
  fetchCRIT(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }
      var h = '<div class="section-label">CRIT SCORE TRAJECTORY</div>';
      h += '<div class="chart-container">' + buildCRITChart(data) + '</div>';
      div.innerHTML = buildCard('CRIT Scores', h, true);
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
