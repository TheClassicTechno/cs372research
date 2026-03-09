import { newViewToken, runsViewState } from './state.js';
import * as ablationView from './views/ablationView.js';
import * as liveView from './views/liveView.js';
import * as runsView from './views/runsView.js';
import * as runDetailView from './views/runDetail/index.js';

var activeView = null;

export function getActiveView() { return activeView; }

function getRoute() {
  var h = window.location.hash || '#live';
  if (h.indexOf('#run/') === 0) {
    var parts = h.substring(5).split('/');
    return { view: 'run-detail', experiment: parts[0], runId: parts.slice(1).join('/') };
  }
  if (h === '#runs') return { view: 'runs' };
  if (h === '#ablation') return { view: 'ablation' };
  if (h === '#live') return { view: 'live' };
  return { view: 'live' };
}

function updateNav() {
  var r = getRoute();
  var links = document.querySelectorAll('#nav a');
  for (var i = 0; i < links.length; i++) {
    var v = links[i].getAttribute('data-view');
    links[i].className = (v === r.view || (r.view === 'run-detail' && v === 'runs')) ? 'active' : '';
  }
}

export function route() {
  // Teardown previous view
  if (activeView && activeView.teardown) {
    activeView.teardown();
  }

  var token = newViewToken();
  updateNav();
  var r = getRoute();

  if (r.view === 'live') {
    activeView = liveView;
    liveView.renderLiveDebateView(token);
  } else if (r.view === 'runs') {
    activeView = runsView;
    runsView.renderRunsView(token);
  } else if (r.view === 'ablation') {
    activeView = ablationView;
    ablationView.renderAblationView(token);
  } else if (r.view === 'run-detail') {
    runsViewState.lastExperiment = r.experiment;
    try { sessionStorage.setItem('dashExp', r.experiment); } catch (e) { }
    activeView = runDetailView;
    runDetailView.renderRunDetailView(r.experiment, r.runId, token);
  }
}
