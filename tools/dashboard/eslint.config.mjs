import security from "eslint-plugin-security";

export default [
  {
    files: ["static/js/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        document: "readonly",
        window: "readonly",
        fetch: "readonly",
        console: "readonly",
        sessionStorage: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        Symbol: "readonly",
        JSON: "readonly",
        encodeURIComponent: "readonly",
        parseInt: "readonly",
      },
    },
    plugins: {
      security,
    },
    rules: {
      // --- Safety ---
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      "no-implicit-globals": "error",

      // --- Code quality ---
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-console": "warn",
      "no-debugger": "error",
      "no-alert": "error",
      "no-var": "warn",
      "eqeqeq": ["warn", "smart"],

      // --- Size limits ---
      "max-lines": ["warn", { max: 400, skipBlankLines: true, skipComments: true }],
      "max-lines-per-function": ["warn", { max: 60, skipBlankLines: true, skipComments: true }],

      // --- Security plugin ---
      "security/detect-eval-with-expression": "error",
      "security/detect-non-literal-regexp": "warn",
      "security/detect-object-injection": "off",
    },
  },

  // --- Layer: utils — must be pure, no DOM, no fetch ---
  {
    files: ["static/js/utils/**/*.js"],
    rules: {
      "no-restricted-globals": ["error",
        { name: "document", message: "utils must not access DOM" },
        { name: "fetch", message: "utils must not perform network requests" },
      ],
    },
  },

  // --- Layer: components — no fetch, no state imports ---
  {
    files: ["static/js/components/**/*.js"],
    rules: {
      "no-restricted-globals": ["error",
        { name: "fetch", message: "components must not perform network requests" },
      ],
    },
  },

  // --- Layer: api — no DOM access ---
  {
    files: ["static/js/api/**/*.js"],
    rules: {
      "no-restricted-globals": ["error",
        { name: "document", message: "api layer must not access DOM" },
      ],
    },
  },
];
