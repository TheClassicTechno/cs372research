import { fetchTree, fetchFile } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildTree } from '../../components/fileTree.js';
import { esc } from '../../utils/dom.js';
import { appState } from '../../state.js';

export function loadFileExplorer(experiment, runId, token) {
  var div = document.getElementById('file-explorer-section');
  fetchTree(experiment, runId)
    .then(function (tree) {
      if (appState.viewToken !== token) return;
      if (!tree || tree.length === 0) { div.innerHTML = ''; return; }
      var h = '<div class="file-tree">' + buildTree(tree, experiment, runId) + '</div>';
      h += '<div id="file-content-display"></div>';
      div.innerHTML = buildCard('File Explorer', h, false);
    })
    .catch(function () { if (appState.viewToken === token) div.innerHTML = ''; });
}

export function loadFileContent(experiment, runId, filePath) {
  var display = document.getElementById('file-content-display');
  if (!display) return;
  var fullPath = 'logging/runs/' + experiment + '/' + runId + '/' + filePath;
  display.innerHTML = '<span class="loading">Loading ' + esc(fullPath) + '...</span>';
  fetchFile(experiment, runId, filePath)
    .then(function (data) {
      var content = typeof data.content === 'string' ? data.content : JSON.stringify(data.content, null, 2);
      var fullPath = 'logging/runs/' + experiment + '/' + runId + '/' + filePath;
      var h = '<div class="section-label">' + esc(fullPath) + '</div>';
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
