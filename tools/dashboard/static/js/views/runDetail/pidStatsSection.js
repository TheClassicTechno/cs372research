import { fetchCRIT } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { esc } from '../../utils/dom.js';
import { fmt, scoreClass } from '../../utils/format.js';
import { makeAgentLabel } from '../../utils/agentLabel.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

export function loadPIDStatsSection(experiment, runId, token) {
  var div = document.getElementById('pid-stats-section');
  fetchCRIT(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      if (!data || data.length === 0) { div.innerHTML = ''; return; }
      var agentLabel = makeAgentLabel(appState.manifest);

      var roles = {};
      data.forEach(function (d) {
        if (d.rho_i) Object.keys(d.rho_i).forEach(function (r) { roles[r] = true; });
      });
      var roleList = Object.keys(roles).sort();
      var psCfg = T('pid_stats');
      var pillars = ['LV', 'ES', 'AC', 'CA'];
      var pillarNames = { 'LV': psCfg.columns[2], 'ES': psCfg.columns[3], 'AC': psCfg.columns[4], 'CA': psCfg.columns[5] };

      var h = '';
      var prevRhoBar = null;
      for (var i = 0; i < data.length; i++) {
        var d = data[i];

        var rhoLabel = '\u03c1\u0304 ' + fmt(d.rho_bar, 3);
        if (prevRhoBar != null && d.rho_bar != null) {
          var delta = d.rho_bar - prevRhoBar;
          var sign = delta >= 0 ? '+' : '';
          var cls = delta >= 0 ? 'delta-up' : 'delta-down';
          rhoLabel += '  <span class="' + cls + '">(' + sign + delta.toFixed(3) + ')</span>';
        }
        prevRhoBar = d.rho_bar;

        h += '<div class="section-label">Round ' + d.round + ' \u2014 ' + rhoLabel + '</div>';
        h += '<table class="data-table">';
        h += '<tr><th>' + esc(psCfg.columns[0]) + '</th><th class="num-col">' + psCfg.columns[1] + '</th>';
        for (var p = 0; p < pillars.length; p++) h += '<th class="num-col">' + pillarNames[pillars[p]] + '</th>';
        h += '</tr>';
        for (var r = 0; r < roleList.length; r++) {
          var role = roleList[r];
          var rho_i = d.rho_i ? d.rho_i[role] : null;
          var agentPillars = (d.pillars && d.pillars[role]) ? d.pillars[role] : {};
          h += '<tr><td>' + esc(agentLabel(role)) + '</td>';
          h += '<td class="num-cell rho-col ' + scoreClass(rho_i) + '">' + fmt(rho_i, 3) + '</td>';
          for (var p = 0; p < pillars.length; p++) {
            var pv = agentPillars[pillars[p]];
            h += '<td class="num-cell ' + scoreClass(pv) + '">' + fmt(pv, 3) + '</td>';
          }
          h += '</tr>';
        }
        h += '</table>';
      }

      div.innerHTML = buildCard(T('cards').pid_stats_overview, h, true);
      div.querySelector('.card').classList.add('open');
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}
