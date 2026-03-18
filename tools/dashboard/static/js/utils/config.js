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
  return Object.keys(obj).sort().flatMap(function (k) {
    let fullKey = prefix !== undefined ? prefix + '.' + k : k;
    let v = obj[k];
    if (v === null || v === undefined) {
      return [{ key: fullKey, value: '\u2014' }];
    } else if (Array.isArray(v)) {
      return [{ key: fullKey, value: v.join(', ') }];
    } else if (typeof v === 'object') {
      return flattenConfig(v, fullKey);
    } else {
      return [{ key: fullKey, value: String(v) }];
    }
  });
}
