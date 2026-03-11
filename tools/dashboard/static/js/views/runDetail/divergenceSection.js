import { fetchDivergence } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

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

      var divCfg = T('divergence');
      var h = '';
      for (var r = 0; r < roundOrder.length; r++) {
        var rn = roundOrder[r];
        var phases = byRound[rn];
        var propose = phases.propose || {};
        var revise = phases.revise || {};

        h += '<div class="section-label">Round ' + rn + '</div>';
        h += '<table class="data-table">';
        h += '<tr>';
        for (var dc = 0; dc < divCfg.columns.length; dc++) {
          h += '<th>' + divCfg.columns[dc] + '</th>';
        }
        h += '</tr>';

        h += '<tr><td>' + divCfg.rows.proposed + '</td>';
        h += '<td>' + fmt(propose.js_divergence) + '</td>';
        h += '<td>' + fmt(propose.mean_overlap) + '</td></tr>';

        h += '<tr><td>' + divCfg.rows.revised + '</td>';
        h += '<td>' + fmt(revise.js_divergence) + '</td>';
        h += '<td>' + fmt(revise.mean_overlap) + '</td></tr>';

        // Retry phases (retry_001, retry_002, ...)
        var phaseKeys = Object.keys(phases).sort();
        for (var p = 0; p < phaseKeys.length; p++) {
          var pk = phaseKeys[p];
          if (pk.indexOf('retry_') !== 0) continue;
          var retryData = phases[pk];
          var retryNum = pk.replace('retry_', '').replace(/^0+/, '') || '1';
          h += '<tr><td>Retry ' + retryNum + '</td>';
          h += '<td>' + fmt(retryData.js_divergence) + '</td>';
          h += '<td>' + fmt(retryData.mean_overlap) + '</td></tr>';
        }

        h += '</table>';
      }

      div.innerHTML = buildCard(T('cards').divergence_overview, h, true);
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
