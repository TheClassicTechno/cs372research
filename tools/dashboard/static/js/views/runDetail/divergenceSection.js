import { fetchDivergence } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';

export function loadDivergenceSection(experiment, runId, token) {
  var div = document.getElementById('divergence-section');
  fetchDivergence(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }

      var h = '';
      for (var i = 0; i < data.length; i++) {
        var d = data[i];
        h += '<div class="section-label">Round ' + d.round + '</div>';
        h += '<table class="data-table">';
        h += '<tr><th>Metric</th><th>Proposed</th><th>Revised</th></tr>';

        var proposed = d.proposed || {};
        var revised = d.revised || {};

        h += '<tr><td>JS Divergence</td>';
        h += '<td>' + fmt(proposed.js_divergence) + '</td>';
        h += '<td>' + fmt(revised.js_divergence) + '</td></tr>';

        h += '<tr><td>Evidence Overlap</td>';
        h += '<td>' + fmt(proposed.evidence_overlap) + '</td>';
        h += '<td>' + fmt(revised.evidence_overlap) + '</td></tr>';

        h += '</table>';
      }

      div.innerHTML = buildCard('Divergence Overview', h, true);
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
