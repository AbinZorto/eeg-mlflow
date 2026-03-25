# Paper Workspace

This directory holds the working LaTeX manuscript and its paper-specific assets.

## Build

```bash
cd paper
latexmk -pdf -output-directory=build main.tex
```

## Refresh Core Assets

```bash
uv run python3 scripts/build_paper_summary_assets.py \
  --artifacts-dir sweeps/artifacts \
  --mlruns-root mlruns \
  --output-root paper
```

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

## Literature

- raw search outputs live under `paper/literature/raw/`
- the working bibliography pool is `paper/references.bib`
- this is intentionally broader than the current prose because the literature search is ongoing
- the draft currently uses `\nocite{*}` to keep the working bibliography visible; remove that during the final venue pass
