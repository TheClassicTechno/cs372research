/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    {
      name: "utils-no-imports",
      comment: "utils/ may import sibling utils but nothing else from the project",
      severity: "error",
      from: { path: "^static/js/utils/" },
      to: { path: "^static/js/", pathNot: "^static/js/utils/" },
    },
    {
      name: "components-no-api",
      comment: "components/ must not import api/",
      severity: "error",
      from: { path: "^static/js/components/" },
      to: { path: "^static/js/api/" },
    },
    {
      name: "components-no-state",
      comment: "components/ must not import state.js",
      severity: "error",
      from: { path: "^static/js/components/" },
      to: { path: "^static/js/state\\.js$" },
    },
    {
      name: "components-no-views",
      comment: "components/ must not import views/",
      severity: "error",
      from: { path: "^static/js/components/" },
      to: { path: "^static/js/views/" },
    },
    {
      name: "api-no-components",
      comment: "api/ must not import components/",
      severity: "error",
      from: { path: "^static/js/api/" },
      to: { path: "^static/js/components/" },
    },
    {
      name: "api-no-views",
      comment: "api/ must not import views/",
      severity: "error",
      from: { path: "^static/js/api/" },
      to: { path: "^static/js/views/" },
    },
    {
      name: "api-no-state",
      comment: "api/ must not import state.js",
      severity: "error",
      from: { path: "^static/js/api/" },
      to: { path: "^static/js/state\\.js$" },
    },
    {
      name: "views-no-app",
      comment: "views/ must not import app.js",
      severity: "error",
      from: { path: "^static/js/views/" },
      to: { path: "^static/js/app\\.js$" },
    },
    {
      name: "no-circular-deps",
      comment: "No circular dependencies allowed",
      severity: "error",
      from: {},
      to: { circular: true },
    },
  ],
  options: {
    doNotFollow: {
      path: "node_modules",
    },
  },
};
