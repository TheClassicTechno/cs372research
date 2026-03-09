/**
 * views/ablationView.js
 *
 * Orchestrates the Ablation tab: fetches summary data, renders experiment
 * cards and overview table, handles the Regenerate action.
 */

import { fetchAblation, regenerateAblation } from '../api/runs.js';
import { buildAblationOverview, buildExperimentCard } from '../components/ablation.js';
import { ablationState, appState } from '../state.js';

/**
 * Render the ablation content from fetched data into the DOM.
 * Accepts the view token and the data object.
 */
function renderContent(token, data) {
  if (appState.viewToken !== token) return;
  var container = document.getElementById('ablation-content');
  if (container === null) {
    throw new Error('ablation-content container not found');
  }

  if (data.error === 'not_generated') {
    container.innerHTML = '<p>No ablation summary found. Click <strong>Regenerate</strong> to generate one.</p>';
    return;
  }

  var names = Object.keys(data);
  if (names.length === 0) {
    container.innerHTML = '<p>Ablation summary is empty.</p>';
    return;
  }

  var html = buildAblationOverview(data);
  for (var i = 0; i < names.length; i++) {
    html += buildExperimentCard(names[i], data[names[i]]);
  }
  container.innerHTML = html;
}

/**
 * Render the ablation view skeleton and load data.
 * Accepts the view token for stale-fetch prevention.
 */
export function renderAblationView(token) {
  var appDiv = document.getElementById('app');
  var html = '<div class="controls">';
  html += '<button data-action="regenerate-ablation" data-testid="regenerate-ablation" type="button">Regenerate</button>';
  html += '<span class="status-text" id="ablation-status"></span>';
  html += '</div>';
  html += '<div id="ablation-content" data-testid="ablation-content"><p class="loading">Loading\u2026</p></div>';
  appDiv.innerHTML = html;

  fetchAblation()
    .then(function (data) {
      if (appState.viewToken !== token) return;
      ablationState.data = data;
      renderContent(token, data);
    })
    .catch(function (err) {
      if (appState.viewToken !== token) return;
      var container = document.getElementById('ablation-content');
      if (container !== null) {
        container.innerHTML = '<p>Error loading ablation data: ' + String(err) + '</p>';
      }
    });
}

/**
 * Execute the regenerate workflow: call API, reload data, update DOM.
 * Uses the current viewToken for stale-fetch guards.
 */
function doRegenerate() {
  var status = document.getElementById('ablation-status');
  if (status !== null) { status.textContent = 'Regenerating\u2026'; }
  var token = appState.viewToken;
  regenerateAblation()
    .then(function (result) {
      if (appState.viewToken !== token) return;
      if (result.status === 'error') {
        if (status !== null) { status.textContent = 'Error: ' + (result.detail || 'unknown'); }
        return;
      }
      if (status !== null) { status.textContent = 'Done. Reloading\u2026'; }
      return fetchAblation();
    })
    .then(function (data) {
      if (data === undefined) return;
      if (appState.viewToken !== token) return;
      ablationState.data = data;
      renderContent(token, data);
      if (status !== null) { status.textContent = ''; }
    })
    .catch(function (err) {
      if (appState.viewToken !== token) return;
      if (status !== null) { status.textContent = 'Error: ' + String(err); }
    });
}

/**
 * Handle delegated actions for the ablation view.
 * Accepts the action name and the source element.
 */
export function handleAction(action, el) {
  if (action === 'regenerate-ablation') {
    doRegenerate();
  }
}

/** Teardown — no timers or polling to clean up. */
export function teardown() {
  // no-op
}
