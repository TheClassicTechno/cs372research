/**
 * api/client.js
 *
 * Low-level HTTP client providing a shared JSON fetch wrapper for all API calls.
 */

/** Fetch a URL and parse the response as JSON. */
export function fetchJSON(url) {
  return fetch(url).then(function (r) { return r.json(); });
}
