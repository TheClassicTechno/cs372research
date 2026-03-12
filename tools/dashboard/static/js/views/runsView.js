import { fetchExperiments, fetchRuns } from '../api/runs.js';
import { runsViewState, appState } from '../state.js';
import { esc } from '../utils/dom.js';
import { fmt, numFmt, fmtDuration } from '../utils/format.js';

export function renderRunsView(token) {
  let appDiv = document.getElementById('app');
  appDiv.innerHTML = '<div class="controls">' +
    '<label for="exp-select">Experiment:</label>' +
    '<select id="exp-select"></select>' +
    '<label for="runs-search">Search:</label>' +
    '<input id="runs-search" type="text" placeholder="Filter runs...">' +
    '<span class="status-text" id="runs-status">Loading...</span>' +
    '</div>' +
    '<div id="runs-table"></div>';

  let expSelect = document.getElementById('exp-select');
  let searchInput = document.getElementById('runs-search');
  let tableDiv = document.getElementById('runs-table');
  let statusSpan = document.getElementById('runs-status');

  fetchExperiments()
    .then(function (exps) {
      if (appState.viewToken !== token) return;
      expSelect.innerHTML = '';
      let targetExp = runsViewState.lastExperiment || '';
      if (!targetExp) { try { targetExp = sessionStorage.getItem('dashExp') || ''; } catch { /* storage unavailable */ } }
      let foundTarget = false;
      exps.forEach(function (e) {
        let opt = document.createElement('option');
        opt.value = e.experiment;
        opt.textContent = e.experiment + ' (' + e.run_count + ' runs)';
        if (e.experiment === targetExp) { opt.selected = true; foundTarget = true; }
        expSelect.appendChild(opt);
      });
      if (exps.length > 0) loadRuns(foundTarget ? targetExp : exps[0].experiment);
      else statusSpan.textContent = 'No experiments found';
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      statusSpan.textContent = 'Failed to load experiments';
    });

  expSelect.addEventListener('change', function () {
    searchInput.value = '';
    runsViewState.lastExperiment = expSelect.value;
    try { sessionStorage.setItem('dashExp', expSelect.value); } catch { /* storage unavailable */ }
    loadRuns(expSelect.value);
  });

  searchInput.addEventListener('input', function () {
    renderRunsTable(runsViewState.allRuns, runsViewState.experiment, searchInput.value);
  });

  function loadRuns(experiment) {
    statusSpan.textContent = 'Loading runs...';
    fetchRuns(experiment)
      .then(function (runs) {
        if (appState.viewToken !== token) return;
        runsViewState.allRuns = runs;
        runsViewState.experiment = experiment;
        renderRunsTable(runs, experiment, searchInput.value);
        statusSpan.textContent = runs.length + ' runs';
      })
      .catch(function () {
        if (appState.viewToken !== token) return;
        statusSpan.textContent = 'Failed to load runs';
      });
  }

  function getRunSearchText(r) {
    let parts = [r.run_id || '', r.status || '', r.model_name || ''];
    if (r.agent_profiles && typeof r.agent_profiles === 'object') {
      parts.push(Object.entries(r.agent_profiles).map(function (e) {
        return typeof e[1] === 'string' ? e[1] : e[0];
      }).join(' '));
    } else if (r.roles) {
      parts.push(r.roles.join(' '));
    }
    if (r.config_paths && r.config_paths.length > 0) {
      parts.push(r.config_paths[0]);
    }
    return parts.join(' ').toLowerCase();
  }

  function renderRunsTable(runs, experiment, searchTerm) {
    let filtered = runs;
    if (searchTerm) {
      let q = searchTerm.toLowerCase();
      filtered = runs.filter(function (r) {
        return getRunSearchText(r).indexOf(q) !== -1;
      });
    }

    if (filtered.length === 0) {
      tableDiv.innerHTML = '<p style="color:#999;font-size:0.9em;">No runs found.</p>';
      return;
    }
    let bestIdx = -1;
    let bestRho = -1;
    for (let i = 0; i < filtered.length; i++) {
      let rho = filtered[i].final_rho_bar;
      if (rho != null && rho > bestRho) {
        bestRho = rho;
        bestIdx = i;
      }
    }

    let h = '<table class="data-table">';
    h += '<tr><th>run_id</th><th>status</th><th>agents</th><th>config</th><th style="width:1%">rounds</th>';
    h += '<th style="width:1%">final <span style="text-decoration:overline">\u03c1</span></th><th style="width:1%">js_drop</th>';
    h += '<th>model</th><th>portfolio</th><th style="width:1%">duration</th></tr>';
    for (let i = 0; i < filtered.length; i++) {
      let r = filtered[i];
      let isBest = (i === bestIdx) ? ' best-run' : '';
      let statusClass = r.status === 'incomplete' ? ' status-incomplete' : (r.status === 'partial' ? ' status-partial' : '');
      let configName = '\u2014';
      if (r.config_paths && r.config_paths.length > 0) {
        let cp = r.config_paths[0];
        let parts = cp.replace(/\\/g, '/').split('/');
        let fname = parts[parts.length - 1];
        configName = fname.replace(/\.yaml$/, '').replace(/\.yml$/, '');
      }
      let agentsList = '\u2014';
      if (r.agent_profiles && typeof r.agent_profiles === 'object') {
        agentsList = Object.entries(r.agent_profiles).map(function (e) {
          return esc(typeof e[1] === 'string' ? e[1] : e[0]);
        }).join('<br>');
      } else if (r.roles && r.roles.length > 0) {
        agentsList = r.roles.map(function (a) { return esc(a); }).join('<br>');
      }
      let perfCell = '\u2014';
      if (r.portfolio_final_value != null) {
        let perfCls = r.portfolio_final_value >= 100000 ? 'perf-profit' : 'perf-loss';
        perfCell = '<span class="' + perfCls + '">$' + numFmt(r.portfolio_final_value) + '</span>';
      }
      h += '<tr class="clickable' + isBest + '" data-action="open-run" data-experiment="' + esc(experiment) + '" data-run-id="' + esc(r.run_id) + '">';
      h += '<td>' + esc(r.run_id) + '</td>';
      h += '<td class="' + statusClass + '">' + esc(r.status) + '</td>';
      h += '<td style="white-space:normal;">' + agentsList + '</td>';
      h += '<td>' + esc(configName) + '</td>';
      h += '<td>' + (r.actual_rounds != null ? r.actual_rounds : '\u2014') + '</td>';
      h += '<td>' + fmt(r.final_rho_bar) + '</td>';
      h += '<td>' + fmt(r.js_drop) + '</td>';
      h += '<td>' + esc(r.model_name || '\u2014') + '</td>';
      h += '<td>' + perfCell + '</td>';
      h += '<td>' + fmtDuration(r.elapsed_s) + '</td>';
      h += '</tr>';
    }
    h += '</table>';
    tableDiv.innerHTML = h;
  }
}

export function handleAction(action, el) {
  if (action === 'open-run') {
    let experiment = el.dataset.experiment;
    let runId = el.dataset.runId;
    window.location.hash = '#run/' + experiment + '/' + runId;
  }
}
