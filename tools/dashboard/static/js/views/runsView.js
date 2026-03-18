/**
 * views/runsView.js
 *
 * Renders the experiment/runs listing view with search filtering and navigation.
 */
import { fetchExperiments, fetchRuns } from '../api/runs.js';
import { runsViewState, appState } from '../state.js';
import { esc } from '../utils/dom.js';
import { fmt, numFmt, fmtDuration } from '../utils/format.js';

/**
 * Builds the static controls HTML for the runs view header.
 * @returns {string} HTML string containing the experiment selector, search input, and status span.
 */
function buildControlsHtml() {
  return '<div class="controls" data-testid="runs-view-content">' +
    '<label for="exp-select">Experiment:</label>' +
    '<select id="exp-select"></select>' +
    '<label for="runs-search">Search:</label>' +
    '<input id="runs-search" type="text" placeholder="Filter runs...">' +
    '<span class="status-text" id="runs-status">Loading...</span>' +
    '</div>' +
    '<div id="runs-table"></div>';
}

/**
 * Wires up event listeners on the experiment dropdown and search input.
 * @param {HTMLSelectElement} expSelect - The experiment dropdown element.
 * @param {HTMLInputElement} searchInput - The search/filter input element.
 * @param {function(string): void} loadRunsFn - Callback to load runs for a given experiment.
 */
function setupEventListeners(expSelect, searchInput, loadRunsFn) {
  expSelect.addEventListener('change', function () {
    searchInput.value = '';
    runsViewState.lastExperiment = expSelect.value;
    try { sessionStorage.setItem('dashExp', expSelect.value); } catch { /* storage unavailable */ }
    loadRunsFn(expSelect.value);
  });

  searchInput.addEventListener('input', function () {
    renderRunsTable(runsViewState.allRuns, runsViewState.experiment, searchInput.value,
      document.getElementById('runs-table'));
  });
}

/**
 * Populates the experiment dropdown with options and selects the previously-active experiment.
 * @param {HTMLSelectElement} expSelect - The experiment dropdown element.
 * @param {Array<{experiment: string, run_count: number}>} exps - List of experiment metadata objects.
 * @returns {{targetExp: string, foundTarget: boolean}} The resolved target experiment and whether it was found.
 */
function populateExperimentDropdown(expSelect, exps) {
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
  return { targetExp: targetExp, foundTarget: foundTarget };
}

/**
 * Builds a lowercase searchable string from a run's key fields for filtering.
 * @param {Object} r - A run object with optional run_id, status, model_name, agent_profiles, roles, config_paths.
 * @returns {string} Lowercase concatenation of searchable fields.
 */
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

/**
 * Builds the HTML string for a single run table row.
 * @param {Object} run - The run data object.
 * @param {number} i - The row index in the filtered list.
 * @param {number} bestIdx - The index of the best-performing run (highest final_rho_bar).
 * @param {string} experiment - The experiment name for the data-experiment attribute.
 * @returns {string} HTML string for one <tr> element.
 */
function buildRunRow(run, i, bestIdx, experiment) {
  let isBest = (i === bestIdx) ? ' best-run' : '';
  let statusClass = run.status === 'incomplete' ? ' status-incomplete' : (run.status === 'partial' ? ' status-partial' : '');
  let configName = '\u2014';
  if (run.config_paths && run.config_paths.length > 0) {
    let cp = run.config_paths[0];
    let parts = cp.replace(/\\/g, '/').split('/');
    let fname = parts[parts.length - 1];
    configName = fname.replace(/\.yaml$/, '').replace(/\.yml$/, '');
  }
  let agentsList = '\u2014';
  if (run.agent_profiles && typeof run.agent_profiles === 'object') {
    agentsList = Object.entries(run.agent_profiles).map(function (e) {
      return esc(typeof e[1] === 'string' ? e[1] : e[0]);
    }).join('<br>');
  } else if (run.roles && run.roles.length > 0) {
    agentsList = run.roles.map(function (a) { return esc(a); }).join('<br>');
  }
  let perfCell = '\u2014';
  if (run.portfolio_final_value != null) {
    let perfCls = run.portfolio_final_value >= 100000 ? 'perf-profit' : 'perf-loss';
    perfCell = '<span class="' + perfCls + '">$' + numFmt(run.portfolio_final_value) + '</span>';
  }
  let h = '<tr class="clickable' + isBest + '" data-action="open-run" data-experiment="' + esc(experiment) + '" data-run-id="' + esc(run.run_id) + '">';
  h += '<td>' + esc(run.run_id) + '</td>';
  h += '<td class="' + statusClass + '">' + esc(run.status) + '</td>';
  h += '<td style="white-space:normal;">' + agentsList + '</td>';
  h += '<td>' + esc(configName) + '</td>';
  h += '<td>' + (run.actual_rounds != null ? run.actual_rounds : '\u2014') + '</td>';
  h += '<td>' + fmt(run.final_rho_bar) + '</td>';
  h += '<td>' + fmt(run.js_drop) + '</td>';
  h += '<td>' + esc(run.model_name || '\u2014') + '</td>';
  h += '<td>' + perfCell + '</td>';
  h += '<td>' + fmtDuration(run.elapsed_s) + '</td>';
  h += '</tr>';
  return h;
}

/**
 * Filters runs by search term and renders the results as an HTML table.
 * @param {Array<Object>} runs - The full list of run objects.
 * @param {string} experiment - The current experiment name.
 * @param {string} searchTerm - The current search/filter string.
 * @param {HTMLElement} tableDiv - The DOM element to render the table into.
 */
function renderRunsTable(runs, experiment, searchTerm, tableDiv) {
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
  let bestIdx = filtered.reduce(function (best, r, idx) {
    let rho = r.final_rho_bar;
    return (rho != null && rho > (best.rho === undefined ? -1 : best.rho)) ? { idx: idx, rho: rho } : best;
  }, { idx: -1, rho: -1 }).idx;

  let h = '<table class="data-table" data-testid="runs-table-content">';
  h += '<tr><th>run_id</th><th>status</th><th>agents</th><th>config</th><th style="width:1%">rounds</th>';
  h += '<th style="width:1%">final <span style="text-decoration:overline">\u03c1</span></th><th style="width:1%">js_drop</th>';
  h += '<th>model</th><th>portfolio</th><th style="width:1%">duration</th></tr>';
  filtered.forEach(function (r, i) {
    h += buildRunRow(r, i, bestIdx, experiment);
  });
  h += '</table>';
  tableDiv.innerHTML = h;
}

/** Initializes the runs list view, loading experiments and wiring up search/select controls. */
export function renderRunsView(token) {
  let appDiv = document.getElementById('app');
  appDiv.innerHTML = buildControlsHtml();

  let expSelect = document.getElementById('exp-select');
  let searchInput = document.getElementById('runs-search');
  let statusSpan = document.getElementById('runs-status');

  /** Fetches runs for the given experiment and renders them into the table. */
  function loadRuns(experiment) {
    statusSpan.textContent = 'Loading runs...';
    fetchRuns(experiment)
      .then(function (runs) {
        if (appState.viewToken !== token) return;
        runsViewState.allRuns = runs;
        runsViewState.experiment = experiment;
        renderRunsTable(runs, experiment, searchInput.value, document.getElementById('runs-table'));
        statusSpan.textContent = runs.length + ' runs';
      })
      .catch(function () {
        if (appState.viewToken !== token) return;
        statusSpan.textContent = 'Failed to load runs';
      });
  }

  fetchExperiments()
    .then(function (exps) {
      if (appState.viewToken !== token) return;
      let result = populateExperimentDropdown(expSelect, exps);
      if (exps.length > 0) loadRuns(result.foundTarget ? result.targetExp : exps[0].experiment);
      else statusSpan.textContent = 'No experiments found';
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      statusSpan.textContent = 'Failed to load experiments';
    });

  setupEventListeners(expSelect, searchInput, loadRuns);
}

/** Handles delegated click actions, navigating to a run detail view on 'open-run'. */
export function handleAction(action, el) {
  if (action === 'open-run') {
    let experiment = el.dataset.experiment;
    let runId = el.dataset.runId;
    window.location.hash = '#run/' + experiment + '/' + runId;
  }
}
