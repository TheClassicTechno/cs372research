import { fetchLiveEvents, clearLiveEvents } from '../api/live.js';
import { buildEventCard, buildRunHeader } from '../components/card.js';
import { liveState, resetLiveState, appState } from '../state.js';

export function renderLiveDebateView(token) {
  var appDiv = document.getElementById('app');
  var html = '<div class="controls">';
  html += '<button id="live-clear-btn" type="button">Clear</button>';
  html += '<span class="status-text" id="live-status">Connecting...</span>';
  html += '</div>';
  html += '<div id="live-entries"></div>';
  appDiv.innerHTML = html;

  var entriesDiv = document.getElementById('live-entries');
  var statusSpan = document.getElementById('live-status');

  resetLiveState();

  document.getElementById('live-clear-btn').addEventListener('click', function () {
    clearLiveEvents();
    entriesDiv.innerHTML = '';
    liveState.renderedIds = {};
    liveState.clearedIds = {};
    statusSpan.textContent = '0 events | cleared';
  });

  function poll() {
    if (appState.viewToken !== token) return;
    fetchLiveEvents()
      .then(function (data) {
        if (appState.viewToken !== token) return;
        var events = data.events || [];

        if (data.run_id && data.run_id !== liveState.runId) {
          liveState.runId = data.run_id;
          liveState.renderedIds = {};
          liveState.clearedIds = {};
          entriesDiv.insertAdjacentHTML('afterbegin', buildRunHeader(data));
        }

        if (!data.run_id && !entriesDiv.innerHTML) {
          entriesDiv.innerHTML = '<p style="color:#999;font-size:0.9em;">No events yet. Waiting for debate...</p>';
          statusSpan.textContent = '0 events';
          return;
        }

        var newEvents = [];
        for (var i = 0; i < events.length; i++) {
          if (!liveState.renderedIds[events[i].id] && !liveState.clearedIds[events[i].id]) {
            newEvents.push(events[i]);
          }
        }

        if (newEvents.length > 0) {
          var ph = entriesDiv.querySelector('p');
          if (ph) ph.remove();

          var header = entriesDiv.querySelector('div[style]');
          var insertAfter = header || null;

          for (var i = 0; i < newEvents.length; i++) {
            var ev = newEvents[i];
            liveState.renderedIds[ev.id] = true;
            if (insertAfter) {
              insertAfter.insertAdjacentHTML('afterend', buildEventCard(ev));
            } else {
              entriesDiv.insertAdjacentHTML('afterbegin', buildEventCard(ev));
            }
          }
        }

        var statusLabel = data.status === 'complete' ? 'complete' : 'live';
        statusSpan.textContent = events.length + ' events | ' + statusLabel;
      })
      .catch(function () {
        if (appState.viewToken !== token) return;
        statusSpan.textContent = 'Connection error';
      });
  }

  poll();
  if (liveState._interval) clearInterval(liveState._interval);
  liveState._interval = setInterval(poll, 1500);
}

export function teardown() {
  if (liveState._interval) {
    clearInterval(liveState._interval);
    liveState._interval = null;
  }
}
