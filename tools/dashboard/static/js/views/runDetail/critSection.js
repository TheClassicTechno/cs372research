/**
 * views/runDetail/critSection.js
 *
 * Loads and renders the CRIT score trajectory chart for a run.
 */
import { fetchCRIT } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildCRITChart } from '../../components/charts.js';
import { makeAgentLabel } from '../../utils/agentLabel.js';
import { appState } from '../../state.js';

/** Fetches CRIT data and renders the score trajectory chart into the crit-section div. */
export function loadCRITSection(experiment, runId, token) {
  let div = document.getElementById('crit-section');
  fetchCRIT(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }
      let agentLabel = makeAgentLabel(appState.manifest);
      let h = '<div class="section-label">CRIT SCORE TRAJECTORY</div>';
      h += '<div class="chart-container">' + buildCRITChart(data, agentLabel) + '</div>';
      div.innerHTML = '<div data-testid="crit-section-content">' + buildCard('CRIT Scores', h, true) + '</div>';
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
