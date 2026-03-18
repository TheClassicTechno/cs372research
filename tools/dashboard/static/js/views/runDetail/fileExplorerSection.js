/**
 * views/runDetail/fileExplorerSection.js
 *
 * Renders a file tree browser and displays individual file contents for a run.
 */
import { fetchTree, fetchFile } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildTree } from '../../components/fileTree.js';
import { esc } from '../../utils/dom.js';
import { appState } from '../../state.js';

/** Fetches the run's file tree and renders a navigable file explorer widget. */
export function loadFileExplorer(experiment, runId, token) {
  let div = document.getElementById('file-explorer-section');
  fetchTree(experiment, runId)
    .then(function (tree) {
      if (appState.viewToken !== token) return;
      if (!tree || tree.length === 0) { div.innerHTML = ''; return; }
      let h = '<div class="file-tree">' + buildTree(tree, experiment, runId) + '</div>';
      h += '<div id="file-content-display"></div>';
      div.innerHTML = '<div data-testid="file-explorer-content">' + buildCard('File Explorer', h, false) + '</div>';
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}

/** Fetches and displays the contents of a single file in the file-content-display area. */
export function loadFileContent(experiment, runId, filePath) {
  let display = document.getElementById('file-content-display');
  if (!display) return;
  let fullPath = 'logging/runs/' + experiment + '/' + runId + '/' + filePath;
  display.innerHTML = '<span class="loading">Loading ' + esc(fullPath) + '...</span>';
  fetchFile(experiment, runId, filePath)
    .then(function (data) {
      let content = typeof data.content === 'string' ? data.content : JSON.stringify(data.content, null, 2);
      let fullPath = 'logging/runs/' + experiment + '/' + runId + '/' + filePath;
      let h = '<div class="section-label">' + esc(fullPath) + '</div>';
      if (data.truncated) {
        h += '<p style="color:#960;font-size:0.8em;">[truncated \u2014 showing first 50,000 chars]</p>';
      }
      h += '<pre class="content">' + esc(content) + '</pre>';
      display.innerHTML = h;
    })
    .catch(function () {
      display.innerHTML = '<span style="color:#900;">Failed to load file.</span>';
    });
}
