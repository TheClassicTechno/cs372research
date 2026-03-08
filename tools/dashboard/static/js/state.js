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

export var appState = {
  viewToken: null
};

export function resetLiveState() {
  liveState.renderedIds = {};
  liveState.clearedIds = {};
  liveState.runId = null;
}

export function newViewToken() {
  appState.viewToken = Symbol();
  return appState.viewToken;
}
