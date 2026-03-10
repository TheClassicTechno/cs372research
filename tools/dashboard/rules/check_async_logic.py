# def detect_async_failure_modes(text: str, rel: str) -> None:
#     """
#     Detect common async failure patterns in AI-generated JavaScript.
#
#     This checker looks for:
#     - unawaited async calls
#     - async functions without try/catch or throw
#     - DOM mutation inside async functions
#     - async render functions
#     - Promise chains without error handling
#     - fetch without error handling
#     - async event handlers without error capture
#     - orphaned setTimeout / setInterval async calls
#     """
#
#     lines = text.splitlines()
#
#     # ------------------------------------------------
#     # 1. Unhandled Promise calls (missing await/catch)
#     # ------------------------------------------------
#     call_pattern = r"\b([A-Za-z_$][\w$]*)\([^)]*\)\s*;"
#     for match in re.finditer(call_pattern, text):
#
#         call = match.group(0)
#
#         if any(x in call for x in ["await", ".then(", ".catch(", "console.", "setTimeout", "setInterval"]):
#             continue
#
#         # ignore common sync functions
#         if match.group(1) in {"render", "log", "warn", "error"}:
#             continue
#
#         line = text[:match.start()].count("\n") + 1
#
#         warnings.append(
#             f"{rel}:{line}: possible unhandled async call (missing await/.catch)"
#         )
#
#     # ------------------------------------------------
#     # 2. Promise chains without catch
#     # ------------------------------------------------
#     promise_pattern = r"\.then\("
#     for match in re.finditer(promise_pattern, text):
#
#         line = text[:match.start()].count("\n") + 1
#
#         snippet = text[match.start():match.start() + 200]
#
#         if ".catch(" not in snippet:
#             warnings.append(
#                 f"{rel}:{line}: Promise chain without .catch()"
#             )
#
#     # ------------------------------------------------
#     # 3. Async function without failure path
#     # ------------------------------------------------
#     async_fn_pattern = r"async\s+function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"
#
#     for match in re.finditer(async_fn_pattern, text):
#
#         fn_name = match.group(1)
#         start = match.end()
#
#         brace = 1
#         i = start
#
#         while i < len(text) and brace > 0:
#             if text[i] == "{":
#                 brace += 1
#             elif text[i] == "}":
#                 brace -= 1
#             i += 1
#
#         body = text[start:i]
#
#         if "throw" not in body and "catch" not in body:
#
#             line = text[:match.start()].count("\n") + 1
#
#             warnings.append(
#                 f"{rel}:{line}: async function '{fn_name}' has no error handling"
#             )
#
#     # ------------------------------------------------
#     # 4. Async render functions
#     # ------------------------------------------------
#     render_pattern = r"async\s+function\s+(render|display|update|draw)[A-Za-z_$]*"
#
#     for match in re.finditer(render_pattern, text):
#
#         line = text[:match.start()].count("\n") + 1
#
#         violations.append(
#             f"{rel}:{line}: async render function detected"
#         )
#
#     # ------------------------------------------------
#     # 5. DOM mutation inside async function
#     # ------------------------------------------------
#     async_fn_pattern = r"async\s+function\s+[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{"
#
#     for match in re.finditer(async_fn_pattern, text):
#
#         start = match.end()
#
#         brace = 1
#         i = start
#
#         while i < len(text) and brace > 0:
#             if text[i] == "{":
#                 brace += 1
#             elif text[i] == "}":
#                 brace -= 1
#             i += 1
#
#         body = text[start:i]
#
#         if any(dom in body for dom in [
#             ".innerHTML",
#             ".appendChild",
#             ".replaceChildren",
#             ".insertAdjacentHTML",
#             ".textContent",
#         ]):
#
#             line = text[:match.start()].count("\n") + 1
#
#             warnings.append(
#                 f"{rel}:{line}: DOM mutation inside async function"
#             )
#
#     # ------------------------------------------------
#     # 6. Fetch without error handling
#     # ------------------------------------------------
#     if "fetch(" in text:
#
#         if "response.ok" not in text and "res.ok" not in text:
#
#             warnings.append(
#                 f"{rel}: fetch() used without checking response.ok"
#             )
#
#     # ------------------------------------------------
#     # 7. Async event listeners
#     # ------------------------------------------------
#     event_pattern = r"addEventListener\([^,]+,\s*async\s*\("
#
#     for match in re.finditer(event_pattern, text):
#
#         line = text[:match.start()].count("\n") + 1
#
#         warnings.append(
#             f"{rel}:{line}: async event handler may swallow errors"
#         )
#
#     # ------------------------------------------------
#     # 8. setTimeout/setInterval with async
#     # ------------------------------------------------
#     timer_pattern = r"(setTimeout|setInterval)\(\s*async"
#
#     for match in re.finditer(timer_pattern, text):
#
#         line = text[:match.start()].count("\n") + 1
#
#         warnings.append(
#             f"{rel}:{line}: async function used inside timer"
#         )
#
#     # ------------------------------------------------
#     # 9. Floating async arrow functions
#     # ------------------------------------------------
#     floating_async = r"async\s*\([^)]*\)\s*=>\s*\{"
#
#     for match in re.finditer(floating_async, text):
#
#         line = text[:match.start()].count("\n") + 1
#
#         warnings.append(
#             f"{rel}:{line}: async arrow function may produce unhandled Promise"
#         )