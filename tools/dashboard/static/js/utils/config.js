/**
 * config.js
 *
 * Pure utility for flattening nested config objects into dot-notation key-value pairs.
 */

/**
 * Flatten a nested config object into dot-notation key-value pairs.
 * Arrays are joined as comma-separated strings.
 * Nested objects are recursively flattened with dot-separated keys.
 *
 * @param {object} obj - The config object to flatten
 * @param {string} [prefix] - Key prefix for recursion
 * @returns {Array<{key: string, value: string}>} Sorted array of {key, value} pairs
 */
export function flattenConfig(obj, prefix) {
  if (obj === null || obj === undefined || typeof obj !== 'object') return [];
  var result = [];
  var keys = Object.keys(obj).sort();
  for (var i = 0; i < keys.length; i++) {
    var k = keys[i];
    var fullKey = prefix !== undefined ? prefix + '.' + k : k;
    var v = obj[k];
    if (v === null || v === undefined) {
      result.push({ key: fullKey, value: '\u2014' });
    } else if (Array.isArray(v)) {
      result.push({ key: fullKey, value: v.join(', ') });
    } else if (typeof v === 'object') {
      var nested = flattenConfig(v, fullKey);
      for (var j = 0; j < nested.length; j++) {
        result.push(nested[j]);
      }
    } else {
      result.push({ key: fullKey, value: String(v) });
    }
  }
  return result;
}
