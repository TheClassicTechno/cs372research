/**
 * views/runDetail/pidStatsSection.js
 *
 * Loads and renders per-round per-agent CRIT pillar scores and rho_i statistics.
 */
import { fetchCRIT } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { esc } from '../../utils/dom.js';
import { fmt, scoreClass } from '../../utils/format.js';
import { makeAgentLabel } from '../../utils/agentLabel.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

/** Fetches CRIT data and renders per-agent pillar breakdowns with rho deltas for each round. */
export function loadPIDStatsSection(experiment, runId, token) {
  let div = document.getElementById('pid-stats-section');
  fetchCRIT(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }
      let agentLabel = makeAgentLabel(appState.manifest);

      let roles = {};
      data.forEach(function (d) {
        if (d.rho_i) Object.keys(d.rho_i).forEach(function (r) { roles[r] = true; });
      });
      let roleList = Object.keys(roles).sort();
      let psCfg = T('pid_stats');
      let pillars = ['LV', 'ES', 'AC', 'CA'];
      let pillarNames = { 'LV': psCfg.columns[2], 'ES': psCfg.columns[3], 'AC': psCfg.columns[4], 'CA': psCfg.columns[5] };

      let h = '';
      let prevRhoBar = null;
      data.forEach(function (d) {

        let rhoLabel = '\u03c1\u0304 ' + fmt(d.rho_bar, 3);
        if (prevRhoBar != null && d.rho_bar != null) {
          let delta = d.rho_bar - prevRhoBar;
          let sign = delta >= 0 ? '+' : '';
          let cls = delta >= 0 ? 'delta-up' : 'delta-down';
          rhoLabel += '  <span class="' + cls + '">(' + sign + delta.toFixed(3) + ')</span>';
        }
        prevRhoBar = d.rho_bar;

        h += '<div class="section-label">Round ' + d.round + ' \u2014 ' + rhoLabel + '</div>';
        h += '<table class="data-table">';
        h += '<tr><th>' + esc(psCfg.columns[0]) + '</th><th class="num-col">' + psCfg.columns[1] + '</th>';
        h += pillars.map(function (pl) { return '<th class="num-col">' + pillarNames[pl] + '</th>'; }).join('');
        h += '</tr>';
        roleList.forEach(function (role) {
          let rho_i = d.rho_i ? d.rho_i[role] : null;
          let agentPillars = (d.pillars && d.pillars[role]) ? d.pillars[role] : {};
          h += '<tr><td>' + esc(agentLabel(role)) + '</td>';
          h += '<td class="num-cell rho-col ' + scoreClass(rho_i) + '">' + fmt(rho_i, 3) + '</td>';
          h += pillars.map(function (pl) {
            let pv = agentPillars[pl];
            return '<td class="num-cell ' + scoreClass(pv) + '">' + fmt(pv, 3) + '</td>';
          }).join('');
          h += '</tr>';
        });
        h += '</table>';
      });

      div.innerHTML = '<div data-testid="pid-stats-content">' + buildCard(T('cards').pid_stats_overview, h, true) + '</div>';
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
