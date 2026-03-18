/**
 * views/runDetail/pidSection.js
 *
 * Loads and renders the PID dynamics table and beta/rho trajectory chart.
 */
import { fetchPID } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildPIDChart } from '../../components/charts.js';
import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

/** Fetches PID data and renders the dynamics table and trajectory chart into the pid-section div. */
export function loadPIDSection(experiment, runId, token) {
  let div = document.getElementById('pid-section');
  fetchPID(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) {
        div.innerHTML = '';
        return;
      }

      let pidCfg = T('pid_dynamics');
      let h = '<div class="section-label">' + esc(pidCfg.title) + '</div>';
      h += '<table class="data-table">';
      h += '<tr>' + pidCfg.columns.map(function (col) { return '<th>' + col + '</th>'; }).join('') + '</tr>';
      data.forEach(function (d) {
        h += '<tr>';
        h += '<td>' + d.round + '</td>';
        h += '<td>' + esc(d.quadrant || '\u2014') + '</td>';
        h += '<td>' + esc(d.tone_bucket || '\u2014') + '</td>';
        h += '<td>' + fmt(d.beta_in) + '</td>';
        h += '<td>' + fmt(d.beta_new) + '</td>';
        h += '<td>' + fmt(d.rho_bar) + '</td>';
        h += '</tr>';
      });
      h += '</table>';

      h += '<div class="section-label">' + esc(T('sections').pid_trajectory) + '</div>';
      h += '<div class="chart-container">' + buildPIDChart(data) + '</div>';

      div.innerHTML = '<div data-testid="pid-section-content">' + buildCard(T('cards').pid_dynamics, h, true) + '</div>';
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
