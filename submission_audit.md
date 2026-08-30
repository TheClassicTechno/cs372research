# Submission Audit — checklist.md §3

Written 2026-08-29. Every entry below was verified against the repository or the
compiled PDF, not from memory.

## Target

| Field | Value | Source |
|---|---|---|
| Workshop | Verify-Agents ("Who Verifies the Agents?"), NeurIPS 2026 | `neuripspaperwhatochange.txt` |
| Page limit | 4–9 content pages; references + appendix excluded | same |
| Template | NeurIPS 2026, `dblblindworkshop` option | `neurips_2026.sty:90` |
| Review | Double-blind | same |
| Archival | Non-archival | same |
| Deadline | **NOT VERIFIED** — no web access this session; confirm on OpenReview | — |

## Compiled PDF state

- Main content **9 pages** (appendix begins p10). Verified by `pdftotext -layout`.
- 45 pages total: content 1–9, appendix 10–37, references 38, checklist 39–45.
- 0 LaTeX errors, 0 undefined references, no `??`, no TODO/FIXME.
- Anonymized: 0 hits for author names, institution, or project repository URLs in
  the compiled text. The single `github.com` string is the LangGraph citation in
  the bibliography (`refs.bib:172`), a third-party asset.
- Ordering note: content → appendix → references → checklist. NeurIPS specifies
  content → references → appendix → checklist. The reviewer FAQ explicitly says not
  to penalize this ordering.

## Repository map

| Artifact | Location |
|---|---|
| Main `.tex` | `neuripsworkshoppaper.tex` (mirrored to `paper.tex`) |
| Bibliography | `refs.bib` — bib ↔ citations match exactly both directions |
| Checklist | `checklist.tex` |
| Financial results | `results/scenario_runs/` — 6 configs × 35 scenarios, 210 `summary.json` |
| Scenario→run join | root-level `results_tracking_*.csv` (load-bearing; see Known issues) |
| Seed-variance runs | `results/seed_variance/` — 30 runs, 210 files |
| Table verification | `analysis/verify_paper_tables.py` → `_report.txt`, `.csv` |
| Noise-floor probe | `analysis/seed_variance_runs.sh`, `analysis/seed_variance.py` |
| Figure generation | `analysis/make_effect_figure.py` → `effect_vs_noise.pdf` |
| Debate configs | `config/debate/baselines/*.yaml` |

## Known methodological issues (all disclosed in §5.1 of the paper)

1. **Two run configurations are not re-executable.** Recorded hashes
   `81d25963ccf7bea5` and `a503355d691a8fbc` match no surviving YAML. Verified by
   hashing all 9831 git blobs and all 438 working-tree YAMLs. Recoverable only from
   the `final/config.yaml` snapshot inside the original run directories, which are
   not on this machine.
2. **The in-loop RCA retry is absent from version history.** `ConsistencyJudge`
   (`eval/consistency.py`) is never imported from `multi_agent/` in any of 511
   commits on any ref; no config ever enabled it. The intervention demonstrably ran
   (all 35 scenarios differ from the unregulated condition) but existed only as an
   uncommitted working copy.
3. **Per-round CRIT logs are absent.** `logging/` is gitignored by design
   (`.gitignore`), so the corpus behind the n=210 correlations lives on whichever
   machine ran the March 8–10 2026 batch. Run directories are named in the tracking
   CSVs (e.g. `logging/runs/default/run_2026-03-10_17-34-53`).
4. **No cross-provider re-runs possible here.** Only `OPENAI_API_KEY` is set.

## Still owed by the authors

- Deadline confirmation, author entry, reviewer nomination, license selection.
- Answer **No** to open code/data access, consistent with `checklist.tex`.
