/**
 * labels.js — Table label configuration loader.
 * Provides T(tableId) lookup for display config from table_labels.json.
 * Must be initialized once via initLabels() before first use.
 */
let _labels = null;

/** Initialize the labels config. Called once from app.js. */
export function initLabels(data) { _labels = data; }

/** Look up table display config by ID. Throws on missing ID. */
export function T(tableId) {
  if (_labels === null) throw new Error('Labels not initialized');
  let cfg = _labels[tableId];
  if (cfg === undefined) throw new Error('Unknown table label ID: ' + tableId);
  return cfg;
}
