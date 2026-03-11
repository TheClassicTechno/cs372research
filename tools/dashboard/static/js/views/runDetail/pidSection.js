import { fetchPID } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildPIDChart } from '../../components/charts.js';
import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

export function loadPIDSection(experiment, runId, token) {
  var div = document.getElementById('pid-section');
  fetchPID(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) {
        div.innerHTML = '';
        return;
      }

      var pidCfg = T('pid_dynamics');
      var h = '<div class="section-label">' + esc(pidCfg.title) + '</div>';
      h += '<table class="data-table">';
      h += '<tr>';
      for (var pc = 0; pc < pidCfg.columns.length; pc++) {
        h += '<th>' + pidCfg.columns[pc] + '</th>';
      }
      h += '</tr>';
      for (var i = 0; i < data.length; i++) {
        var d = data[i];
        h += '<tr>';
        h += '<td>' + d.round + '</td>';
        h += '<td>' + esc(d.quadrant || '\u2014') + '</td>';
        h += '<td>' + esc(d.tone_bucket || '\u2014') + '</td>';
        h += '<td>' + fmt(d.beta_in) + '</td>';
        h += '<td>' + fmt(d.beta_new) + '</td>';
        h += '<td>' + fmt(d.rho_bar) + '</td>';
        h += '</tr>';
      }
      h += '</table>';

      h += '<div class="section-label">' + esc(T('sections').pid_trajectory) + '</div>';
      h += '<div class="chart-container">' + buildPIDChart(data) + '</div>';

      div.innerHTML = buildCard(T('cards').pid_dynamics, h, true);
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
