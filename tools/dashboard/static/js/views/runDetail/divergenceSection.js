/**
 * views/runDetail/divergenceSection.js
 *
 * Loads and renders the per-round JS divergence and overlap statistics table.
 */
import { fetchDivergence } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { fmt } from '../../utils/format.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

/** Fetches divergence data and renders per-round JS divergence tables into the divergence-section div. */
export function loadDivergenceSection(experiment, runId, token) {
  let div = document.getElementById('divergence-section');
  fetchDivergence(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }

      // Group flat entries by round: { roundNum: { propose: {...}, revise: {...} } }
      let byRound = {};
      let roundOrder = [];
      data.forEach(function (d) {
        const rn = d.round;
        if (!byRound[rn]) {
          byRound[rn] = {};
          roundOrder.push(rn);
        }
        byRound[rn][d.phase] = d;
      });

      let divCfg = T('divergence');
      let h = roundOrder.reduce(function (acc, rn) {
        let phases = byRound[rn];
        let propose = phases.propose || {};
        let revise = phases.revise || {};

        acc += '<div class="section-label">Round ' + rn + '</div>';
        acc += '<table class="data-table">';
        acc += '<tr>' + divCfg.columns.map(function (col) { return '<th>' + col + '</th>'; }).join('') + '</tr>';

        acc += '<tr><td>' + divCfg.rows.proposed + '</td>';
        acc += '<td>' + fmt(propose.js_divergence) + '</td>';
        acc += '<td>' + fmt(propose.mean_overlap) + '</td></tr>';

        acc += '<tr><td>' + divCfg.rows.revised + '</td>';
        acc += '<td>' + fmt(revise.js_divergence) + '</td>';
        acc += '<td>' + fmt(revise.mean_overlap) + '</td></tr>';

        // Retry phases (retry_001, retry_002, ...)
        let retryRows = Object.keys(phases).sort().filter(function (pk) { return pk.indexOf('retry_') === 0; }).map(function (pk) {
          let retryData = phases[pk];
          let retryNum = pk.replace('retry_', '').replace(/^0+/, '') || '1';
          return '<tr><td>Retry ' + retryNum + '</td>'
            + '<td>' + fmt(retryData.js_divergence) + '</td>'
            + '<td>' + fmt(retryData.mean_overlap) + '</td></tr>';
        }).join('');
        acc += retryRows;

        acc += '</table>';
        return acc;
      }, '');

      div.innerHTML = '<div data-testid="divergence-content">' + buildCard(T('cards').divergence_overview, h, true) + '</div>';
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
