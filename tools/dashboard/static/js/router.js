import { newViewToken, runsViewState } from './state.js';
import * as ablationView from './views/ablationView.js';
import * as runsView from './views/runsView.js';
import * as runDetailView from './views/runDetail/index.js';

let activeView = null;

export function getActiveView() { return activeView; }

function getRoute() {
  let h = window.location.hash || '#runs';
  if (h.indexOf('#run/') === 0) {
    let parts = h.substring(5).split('/');
    return { view: 'run-detail', experiment: parts[0], runId: parts.slice(1).join('/') };
  }
  if (h === '#runs') return { view: 'runs' };
  if (h === '#ablation') return { view: 'ablation' };
  return { view: 'runs' };
}

function updateNav() {
  let r = getRoute();
  let links = document.querySelectorAll('#nav a');
  for (let i = 0; i < links.length; i++) {
    let v = links[i].getAttribute('data-view');
    links[i].className = (v === r.view || (r.view === 'run-detail' && v === 'runs')) ? 'active' : '';
  }
}

export function route() {
  // Teardown previous view
  if (activeView && activeView.teardown) {
    activeView.teardown();
  }

  let token = newViewToken();
  updateNav();
  let r = getRoute();

  if (r.view === 'runs') {
    activeView = runsView;
    runsView.renderRunsView(token);
  } else if (r.view === 'ablation') {
    activeView = ablationView;
    ablationView.renderAblationView(token);
  } else if (r.view === 'run-detail') {
    runsViewState.lastExperiment = r.experiment;
    try { sessionStorage.setItem('dashExp', r.experiment); } catch { /* storage unavailable */ }
    activeView = runDetailView;
    runDetailView.renderRunDetailView(r.experiment, r.runId, token);
  }
}
