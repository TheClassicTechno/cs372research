export function fetchJSON(url) {
  return fetch(url).then(function (r) { return r.json(); });
}
