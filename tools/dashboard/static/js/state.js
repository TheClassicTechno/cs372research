/**
 * state.js
 *
 * Centralized mutable application state and state-mutation helpers.
 */

export const liveState = {
  renderedIds: {},
  clearedIds: {},
  runId: null,
  _interval: null
};

export const runsViewState = {
  allRuns: [],
  experiment: '',
  lastExperiment: ''
};

export const ablationState = {
  data: null
};

export const appState = {
  viewToken: null,
  manifest: null
};

/** Reset live-view state when leaving the live page. */
export function resetLiveState() {
  liveState.renderedIds = {};
  liveState.clearedIds = {};
  liveState.runId = null;
}

/** Generate a fresh view token to guard stale async writes. */
export function newViewToken() {
  appState.viewToken = Symbol();
  return appState.viewToken;
}

/** Store the current run's manifest for use by lazy-loaded sections. */
export function setManifest(manifest) {
  appState.manifest = manifest;
}
