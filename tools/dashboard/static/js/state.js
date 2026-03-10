export var liveState = {
  renderedIds: {},
  clearedIds: {},
  runId: null,
  _interval: null
};

export var runsViewState = {
  allRuns: [],
  experiment: '',
  lastExperiment: ''
};

export var ablationState = {
  data: null
};

export var appState = {
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
