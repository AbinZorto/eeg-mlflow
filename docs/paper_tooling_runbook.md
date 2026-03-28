# Paper Tooling Runbook

Reference for building the manuscript, figures, tables, and literature base.

## Manuscript Workspace

- paper root: `paper/`
- main TeX entrypoint: `paper/main.tex`
- generated figures: `paper/figures/`
- generated tables: `paper/tables/`
- machine-readable facts: `paper/data/manuscript_facts.json`
- raw literature search outputs: `paper/literature/raw/`
- legacy external sources used during drafting:
  - bibliography source: `/Users/abin/Paper/analysis.bib`
  - outdated same-data manuscript draft: `/Users/abin/Paper/paper2-body.tex`

## Literature Search Tools

### `paper-scripts/search_citations.py`

- purpose: search CORE, OpenAlex, and Semantic Scholar for paper candidates
- current behavior:
  - auto-loads `repo/.env`
  - uses CORE and OpenAlex directly
  - attempts Semantic Scholar with API key first, then retries without the key if a `403` is returned
  - may still hit Semantic Scholar `429` limits on broad queries
- recommended usage:

```bash
python3 paper-scripts/search_citations.py \
  "major depressive disorder EEG biomarker treatment response" \
  --output paper/literature/raw/mdd_eeg_treatment_response.json \
  --limit 6
```

- query guidance:
  - prefer narrow, clinically anchored queries
  - avoid short broad queries like `depression EEG biomarker`
  - search in thematic batches while writing instead of only once at the beginning

### `paper-scripts/format_bibtex.py`

- purpose: convert JSON search outputs into BibTeX entries
- usage:

```bash
python3 paper-scripts/format_bibtex.py \
  --input paper/literature/raw/mdd_eeg_treatment_response.json \
  --output paper/references_raw.bib
```

- important rule: do not append to the same BibTeX file from multiple commands in parallel

### Legacy bibliography workflow

- when useful references already exist in `/Users/abin/Paper/analysis.bib`, extract only the keys you plan to cite into `paper/references_analysis_subset.bib`
- do not point the manuscript directly at the full legacy bibliography unless you have checked it for malformed entries
- keep `paper2-body.tex` in a supporting role only; it is useful for recovering acquisition details, but it is explicitly outdated and should be cross-checked against current configs

## Figure And Table Builders

### `scripts/build_paper_summary_assets.py`

- purpose: manuscript-specific summary layer
- outputs:
  - `paper/figures/figure1_sweep_overview.png`
  - `paper/tables/best_run_comparison.tex`
  - `paper/tables/hybrid_recurrent_biomarkers.tex`
  - `paper/data/manuscript_facts.json`
- usage:

```bash
uv run python3 scripts/build_paper_summary_assets.py \
  --artifacts-dir sweeps/artifacts \
  --mlruns-root mlruns \
  --output-root paper
```

### `scripts/plot_paper_figures.py`

- purpose: single-run standalone performance and biomarker figures
- current primary run:
  - hybrid: `561eb29a046946818320bea18f21bed6`
  - svm: `e0f3834332d545da893277f039d28e4b`
- hybrid example:

```bash
uv run python3 scripts/plot_paper_figures.py \
  --artifacts-dir sweeps/artifacts \
  --output-dir paper/assets/paper_figures \
  --mlruns-root mlruns \
  --mlflow-run-id 561eb29a046946818320bea18f21bed6 \
  --model advanced_hybrid_1dcnn_lstm \
  --window-size 6 \
  --inner-k 1 \
  --outer-k 10 \
  --fs-method select_k_best_f_classif \
  --n-features 10 \
  --ordering sequential \
  --equalize-lopo-groups true \
  --use-smote true
```

### `scripts/compose_paper_panels.py`

- purpose: turn the single-run outputs into manuscript-ready composites
- hybrid example:

```bash
uv run python3 scripts/compose_paper_panels.py \
  --artifacts-dir sweeps/artifacts \
  --mlruns-root mlruns \
  --paper-figures-dir paper/assets/paper_figures \
  --output-dir paper/assets/paper_panels \
  --select mlflow-run-id \
  --mlflow-run-id 561eb29a046946818320bea18f21bed6 \
  --model advanced_hybrid_1dcnn_lstm \
  --window-size 6 \
  --inner-k 1 \
  --outer-k 10 \
  --fs-method select_k_best_f_classif \
  --n-features 10 \
  --ordering sequential \
  --equalize-lopo-groups true \
  --use-smote true
```

### `scripts/plot_sweep_roc_auc.py`

- purpose: generic sweep trend plots across metrics
- note: useful for exploration, but not sufficient by itself for the paper because the default aggregate lines collapse across models
- paper-oriented preset:

```bash
uv run python3 scripts/plot_sweep_roc_auc.py \
  --artifacts-dir sweeps/artifacts \
  --output-dir paper/assets/sweep_preset \
  --preset paper_main
```

## LaTeX Build

- working bibliography files used by the draft:
  - `paper/references.bib`
  - `paper/references_analysis_subset.bib`
- default rule:
  - do not compile after routine manuscript edits
  - compile only when the user explicitly asks for it or when a milestone build is needed to verify figure paths, citations, or layout
- build command:

```bash
cd paper
latexmk -pdf -output-directory=build_curated main.tex
```

- cleanup command:

```bash
cd paper
latexmk -c -output-directory=build_curated main.tex
```

## Drafting Workflow

1. Re-run literature searches for the section you are actively writing.
2. Append relevant JSON outputs into `paper/references_raw.bib` serially.
3. Pull any high-value older references from `/Users/abin/Paper/analysis.bib` into `paper/references_analysis_subset.bib`.
4. Refresh `paper/references.bib` from the curated working file.
5. Rebuild `scripts/build_paper_summary_assets.py` outputs if manuscript claims change.
6. Re-open the figures before changing captions or result statements.
7. Compile with `latexmk` only on explicit request or at a milestone check.
