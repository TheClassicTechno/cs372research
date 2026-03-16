/**
 * views/ablationView.js
 *
 * Orchestrates the Ablation tab: fetches summary data, renders experiment
 * cards and overview table, handles the Regenerate action.
 */

import { fetchAblation, regenerateAblation, fetchAblationDebateImpact, fetchPairedTests, fetchFinancialTests, fetchFinancialTestsMeanRev, fetchCritDiagnostics, fetchFinancialSignificance } from '../api/runs.js';
import { buildAblationOverview, buildExperimentCard, buildPairedTestsSection } from '../components/ablation.js';
import { buildCritDiagnosticsSection } from '../components/critDiagnostics.js';
import { buildFinancialSignificanceTable } from '../components/financialSignificance.js';
import { buildFinancialTestsSection } from '../components/financialTests.js';
import { ablationState, appState } from '../state.js';

/**
 * Render the ablation content from fetched data into the DOM.
 * Accepts the view token and the data object.
 */
function renderContent(token, data, impactMap) {
  if (appState.viewToken !== token) return;
  const container = document.getElementById('ablation-content');
  if (container === null) {
    throw new Error('ablation-content container not found');
  }

  if (data.error === 'not_generated') {
    container.innerHTML = '<p>No ablation summary found. Click <strong>Regenerate</strong> to generate one.</p>';
    return;
  }

  const names = Object.keys(data);
  if (names.length === 0) {
    container.innerHTML = '<p>Ablation summary is empty.</p>';
    return;
  }

  const impacts = impactMap !== undefined && impactMap !== null ? impactMap : {};
  let html = buildAblationOverview(data);
  html += '<div data-testid="financial-significance-slot"></div>';
  for (let i = 0; i < names.length; i++) {
    html += buildExperimentCard(names[i], data[names[i]], impacts[names[i]]);
  }
  container.innerHTML = html;

  loadFinancialSignificanceSummary(token);
  loadCritDiagnostics(token, names);
  loadPairedTests(token, names);
  loadFinancialTests(token, names);
  loadFinancialTestsMeanRev(token, names);
}

/**
 * Fetch and inject the cross-ablation financial significance summary table.
 * Fills the single slot after the overview table.
 * Accepts the view token.
 */
function loadFinancialSignificanceSummary(token) {
  fetchFinancialSignificance()
    .then(function (result) {
      if (appState.viewToken !== token) return;
      let slot = document.querySelector('[data-testid="financial-significance-slot"]');
      if (slot !== null) {
        slot.innerHTML = buildFinancialSignificanceTable(result);
      }
    })
    .catch(function () {
      // Non-critical — leave slot empty if endpoint unavailable
    });
}

/**
 * Fetch and inject CRIT reasoning diagnostics for each experiment.
 * Finds the placeholder slot inside each experiment card and fills it.
 * Accepts the view token and the list of experiment names.
 */
function loadCritDiagnostics(token, names) {
  const slots = document.querySelectorAll('[data-testid="crit-diagnostics-slot"]');
  for (let i = 0; i < names.length; i++) {
    (function (expName, slot) {
      fetchCritDiagnostics(expName)
        .then(function (result) {
          if (appState.viewToken !== token) return;
          slot.innerHTML = buildCritDiagnosticsSection(result);
        })
        .catch(function () {
          // Non-critical — leave slot empty if endpoint unavailable
        });
    })(names[i], slots[i]);
  }
}

/**
 * Fetch and inject paired t-test results for each experiment.
 * Finds the placeholder slot inside each experiment card and fills it.
 * Accepts the view token and the list of experiment names.
 */
function loadPairedTests(token, names) {
  const slots = document.querySelectorAll('[data-testid="paired-tests-slot"]');
  for (let i = 0; i < names.length; i++) {
    (function (expName, slot) {
      fetchPairedTests(expName)
        .then(function (result) {
          if (appState.viewToken !== token) return;
          slot.innerHTML = buildPairedTestsSection(result);
        })
        .catch(function () {
          // Non-critical — leave slot empty if endpoint unavailable
        });
    })(names[i], slots[i]);
  }
}

/**
 * Fetch and inject financial metrics paired tests for each experiment.
 * Finds the placeholder slot inside each experiment card and fills it.
 * Accepts the view token and the list of experiment names.
 */
function loadFinancialTests(token, names) {
  const slots = document.querySelectorAll('[data-testid="financial-tests-slot"]');
  for (let i = 0; i < names.length; i++) {
    (function (expName, slot) {
      fetchFinancialTests(expName)
        .then(function (result) {
          if (appState.viewToken !== token) return;
          slot.innerHTML = buildFinancialTestsSection(result);
        })
        .catch(function () {
          // Non-critical — leave slot empty if endpoint unavailable
        });
    })(names[i], slots[i]);
  }
}

/**
 * Fetch and inject mean-agent-revision financial tests for each experiment.
 * Finds the placeholder slot inside each experiment card and fills it.
 * Accepts the view token and the list of experiment names.
 */
function loadFinancialTestsMeanRev(token, names) {
  const slots = document.querySelectorAll('[data-testid="financial-tests-mr-slot"]');
  for (let i = 0; i < names.length; i++) {
    (function (expName, slot) {
      fetchFinancialTestsMeanRev(expName)
        .then(function (result) {
          if (appState.viewToken !== token) return;
          slot.innerHTML = buildFinancialTestsSection(result);
        })
        .catch(function () {
          // Non-critical — leave slot empty if endpoint unavailable
        });
    })(names[i], slots[i]);
  }
}

/**
 * Render the ablation view skeleton and load data.
 * Accepts the view token for stale-fetch prevention.
 */
export function renderAblationView(token) {
  const appDiv = document.getElementById('app');
  let html = '<div class="controls">';
  html += '<button data-action="regenerate-ablation" data-testid="regenerate-ablation" type="button">Regenerate</button>';
  html += '<span class="status-text" id="ablation-status"></span>';
  html += '</div>';
  html += '<div id="ablation-content" data-testid="ablation-content"><p class="loading">Loading\u2026</p></div>';
  appDiv.innerHTML = html;

  Promise.all([fetchAblation(), fetchAblationDebateImpact()])
    .then(function (results) {
      if (appState.viewToken !== token) return;
      ablationState.data = results[0];
      renderContent(token, results[0], results[1]);
    })
    .catch(function (err) {
      if (appState.viewToken !== token) return;
      const container = document.getElementById('ablation-content');
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
  const status = document.getElementById('ablation-status');
  if (status !== null) { status.textContent = 'Regenerating\u2026'; }
  const token = appState.viewToken;
  regenerateAblation()
    .then(function (result) {
      if (appState.viewToken !== token) return;
      if (result.status === 'error') {
        if (status !== null) { status.textContent = 'Error: ' + (result.detail || 'unknown'); }
        return;
      }
      if (status !== null) { status.textContent = 'Done. Reloading\u2026'; }
      return Promise.all([fetchAblation(), fetchAblationDebateImpact()]);
    })
    .then(function (results) {
      if (results === undefined) return;
      if (appState.viewToken !== token) return;
      ablationState.data = results[0];
      renderContent(token, results[0], results[1]);
      if (status !== null) { status.textContent = ''; }
      // Note: renderContent already calls loadPairedTests
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
export function handleAction(action, _el) {
  if (action === 'regenerate-ablation') {
    doRegenerate();
  }
}

/** Teardown — no timers or polling to clean up. */
export function teardown() {
  // no-op
}
