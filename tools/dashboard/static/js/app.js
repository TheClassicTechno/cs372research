import { route, getActiveView } from './router.js';
import { initLabels } from './utils/labels.js';

fetch('/static/table_labels.json')
  .then(function (r) { return r.json(); })
  .then(function (data) {
    initLabels(data);

    // Event delegation at #app
    document.getElementById('app').addEventListener('click', function (e) {
      // 1. Card toggle — universal
      var header = e.target.closest('.card-header');
      if (header) {
        var card = header.closest('.card');
        if (card) card.classList.toggle('open');
        return;
      }

      // 2. Dir toggle in file tree
      var dirToggle = e.target.closest('.dir-toggle');
      if (dirToggle) {
        dirToggle.parentElement.classList.toggle('collapsed');
        return;
      }

      // 3. data-action dispatch — forward to active view
      var actionEl = e.target.closest('[data-action]');
      if (actionEl) {
        var action = actionEl.dataset.action;

        // dir toggle handled above already via class, but also via data-action
        if (action === 'toggle-dir') {
          actionEl.parentElement.classList.toggle('collapsed');
          return;
        }

        var view = getActiveView();
        if (view && view.handleAction) {
          view.handleAction(action, actionEl);
        }
      }
    });

    // Hash routing
    window.addEventListener('hashchange', route);
    route();
  });
