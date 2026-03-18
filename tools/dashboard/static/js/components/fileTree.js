/**
 * components/fileTree.js
 *
 * Recursive HTML builder for the run file-explorer tree widget.
 */

import { esc } from '../utils/dom.js';

/** Recursively build a nested <ul> tree of directories and clickable file links. */
export function buildTree(items, experiment, runId, _isRoot) {
  if (_isRoot === undefined) _isRoot = true;
  let tag = _isRoot ? '<ul data-testid="file-tree">' : '<ul>';
  let inner = items.map(function (item) {
    if (item.type === 'dir') {
      let li = '<li><span class="dir-toggle" data-action="toggle-dir">[+] ' + esc(item.name) + '/</span>';
      if (item.children) li += buildTree(item.children, experiment, runId, false);
      li += '</li>';
      return li;
    }
    let li = '<li><span class="file-link" data-action="load-file" data-experiment="' + esc(experiment) + '" data-run-id="' + esc(runId) + '" data-path="' + esc(item.path) + '">' + esc(item.name) + '</span>';
    if (item.size_bytes != null) li += ' <span style="color:#999">(' + item.size_bytes + 'b)</span>';
    li += '</li>';
    return li;
  }).join('');
  return tag + inner + '</ul>';
}
