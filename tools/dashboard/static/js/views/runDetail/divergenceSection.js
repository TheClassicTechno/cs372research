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

      // Group flat entries by round: { roundNum: { propose: {...}, revise: {...} } }
      var byRound = {};
      var roundOrder = [];
      for (var i = 0; i < data.length; i++) {
        var d = data[i];
        var rn = d.round;
        if (!byRound[rn]) {
          byRound[rn] = {};
          roundOrder.push(rn);
        }
        byRound[rn][d.phase] = d;
      }

      var h = '';
      for (var r = 0; r < roundOrder.length; r++) {
        var rn = roundOrder[r];
        var phases = byRound[rn];
        var propose = phases.propose || {};
        var revise = phases.revise || {};

        h += '<div class="section-label">Round ' + rn + '</div>';
        h += '<table class="data-table">';
        h += '<tr><th>Metric</th><th>Proposed</th><th>Revised</th></tr>';

        h += '<tr><td>JS Divergence</td>';
        h += '<td>' + fmt(propose.js_divergence) + '</td>';
        h += '<td>' + fmt(revise.js_divergence) + '</td></tr>';

        h += '<tr><td>Evidence Overlap</td>';
        h += '<td>' + fmt(propose.mean_overlap) + '</td>';
        h += '<td>' + fmt(revise.mean_overlap) + '</td></tr>';

        h += '</table>';
      }

      div.innerHTML = buildCard('Divergence Overview', h, true);
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
