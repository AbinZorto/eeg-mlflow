# Plot Comparisons Registry

Persistent tracker for requested experiment plot/comparison outputs.

## Status Legend

- `requested`: captured request, not started
- `in_progress`: implementation in progress
- `implemented`: script/notebook implemented
- `validated`: outputs reviewed/accepted
- `blocked`: waiting on data or decisions

## Entry Template

Copy this block for each new request:

```md
### plot-####
- requested_on: YYYY-MM-DD
- requested_by: user
- status: requested
- comparison_goal:
- plot_type: TBD
- grouping_dimensions:
- required_inputs:
- acceptance_criteria:
- implementation_refs:
  - script:
  - notebook:
- artifact_outputs:
  - 
- notes:
```

## Requests

### plot-0001
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show window size vs patient ROC-AUC with one dim trajectory per full setting combination and one stronger average line.
- plot_type: multi-line trend plot
- grouping_dimensions: x=`window_seconds`; dim lines grouped by `model`, `inner_k`, `outer_k`, `equalize_lopo_groups`, `use_smote`, `fs_enabled`, `fs_method`, `n_features`, `ordering`, and `dataset_run_id`
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with `mlflow_metrics.patient_roc_auc`
- acceptance_criteria: Generate a plot with dim individual trajectories, one stronger average line, and a CSV export of the deduplicated plotting table.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/window_size_vs_patient_roc_auc.png`
  - `sweeps/plots/plot_data_window_size_vs_patient_roc_auc.csv`
- notes: Uses deduplicated latest successful attempts only; the plotting engine is now metric-generic and defaults to `patient_roc_auc`.

### plot-0002
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show inner-k vs patient ROC-AUC with one dim trajectory per full setting combination and one stronger average line.
- plot_type: multi-line trend plot
- grouping_dimensions: x=`inner_k`; dim lines grouped by `model`, `window_seconds`, `outer_k`, `equalize_lopo_groups`, `use_smote`, `fs_enabled`, `fs_method`, `n_features`, `ordering`, and `dataset_run_id`
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with `mlflow_metrics.patient_roc_auc`
- acceptance_criteria: Generate a plot with dim individual trajectories, one stronger average line, and a CSV export of the deduplicated plotting table.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/inner_k_vs_patient_roc_auc.png`
  - `sweeps/plots/plot_data_inner_k_vs_patient_roc_auc.csv`
- notes: Uses deduplicated latest successful attempts only; the plotting engine is now metric-generic and defaults to `patient_roc_auc`.

### plot-0023
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show window size vs patient PR-AUC and inner-k vs patient PR-AUC using the generalized sweep plotting engine.
- plot_type: multi-line trend plot
- grouping_dimensions: x=`window_seconds` or `inner_k`; dim lines grouped by the remaining sweep-defining settings
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with `mlflow_metrics.patient_pr_auc`
- acceptance_criteria: Generate metric-specific plots and CSVs for `patient_pr_auc` with the same dim-trajectory plus strong-average styling as ROC-AUC.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/window_size_vs_patient_pr_auc.png`
  - `sweeps/plots/inner_k_vs_patient_pr_auc.png`
- notes: Invoked through `--metric patient_pr_auc`.

### plot-0024
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show window size vs patient balanced accuracy and inner-k vs patient balanced accuracy using the generalized sweep plotting engine.
- plot_type: multi-line trend plot
- grouping_dimensions: x=`window_seconds` or `inner_k`; dim lines grouped by the remaining sweep-defining settings
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with `mlflow_metrics.patient_balanced_accuracy`
- acceptance_criteria: Generate metric-specific plots and CSVs for `patient_balanced_accuracy`.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/window_size_vs_patient_balanced_accuracy.png`
  - `sweeps/plots/inner_k_vs_patient_balanced_accuracy.png`
- notes: Invoked through `--metric patient_balanced_accuracy`.

### plot-0025
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show window size vs mean pairwise Jaccard and inner-k vs mean pairwise Jaccard using the generalized sweep plotting engine.
- plot_type: multi-line trend plot
- grouping_dimensions: x=`window_seconds` or `inner_k`; dim lines grouped by the remaining sweep-defining settings
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with `mlflow_metrics.feature_selection_mean_pairwise_jaccard`
- acceptance_criteria: Generate metric-specific plots and CSVs for `feature_selection_mean_pairwise_jaccard`.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/window_size_vs_feature_selection_mean_pairwise_jaccard.png`
  - `sweeps/plots/inner_k_vs_feature_selection_mean_pairwise_jaccard.png`
- notes: Invoked through `--metric feature_selection_mean_pairwise_jaccard`.

### plot-0026
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Support generalized sweep plots for flattened biomarker summary metrics such as Kuncheva, top-k overlap, effect sign consistency, and importance variance.
- plot_type: multi-line trend plot
- grouping_dimensions: x=`window_seconds` or `inner_k`; dim lines grouped by the remaining sweep-defining settings
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with the requested flattened biomarker metric in `mlflow_metrics`
- acceptance_criteria: Allow `scripts/plot_sweep_roc_auc.py --metric <metric_key>` to plot biomarker summary metrics beyond Jaccard without code duplication.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/window_size_vs_feature_selection_kuncheva_index_mean.png`
  - `sweeps/plots/inner_k_vs_feature_selection_kuncheva_index_mean.png`
  - `sweeps/plots/window_size_vs_feature_selection_mean_top_k_overlap.png`
  - `sweeps/plots/window_size_vs_feature_selection_effect_mean_sign_consistency.png`
- notes: The plotting engine auto-scales potentially signed metrics such as Kuncheva instead of forcing a `[0, 1]` y-range.

### plot-0027
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Support preset-driven sweep plot batches for common performance and biomarker metric families.
- plot_type: multi-metric sweep plot batch
- grouping_dimensions: x=`window_seconds` or `inner_k`; one metric-specific plot pair per preset member
- required_inputs: `sweeps/artifacts/*.results.jsonl` success rows with flattened `mlflow_metrics` for the requested preset metrics
- acceptance_criteria: Allow `scripts/plot_sweep_roc_auc.py --preset <name>` to emit multiple metric-specific sweep plots in one invocation while tolerating missing metrics within the batch.
- implementation_refs:
  - script: `scripts/plot_sweep_roc_auc.py`
  - notebook:
- artifact_outputs:
  - `sweeps/plots/window_size_vs_<metric>.png`
  - `sweeps/plots/inner_k_vs_<metric>.png`
  - `sweeps/plots/plot_data_window_size_vs_<metric>.csv`
  - `sweeps/plots/plot_data_inner_k_vs_<metric>.csv`
- notes: Implemented presets are `performance_core`, `performance_extended`, `biomarker_core`, `biomarker_extended`, `biomarker_availability`, and `paper_main`. `--list-presets` prints the current registry from the script.

### plot-0003
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show patient-level ROC discrimination for one resolved MLflow training run.
- plot_type: ROC curve
- grouping_dimensions: x=`false_positive_rate`; y=`true_positive_rate`
- required_inputs: resolved MLflow run with patient prediction probabilities from `*_patient_predictions.csv` or nested `fold_patient_predictions.json`
- acceptance_criteria: Generate a standalone patient ROC figure with ROC-AUC annotation and save it under the paper-figure output tree.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/performance/patient_roc_curve.png`
- notes: Skipped automatically if patient probabilities are unavailable or only one class is present.

### plot-0004
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show patient-level precision-recall behavior for one resolved MLflow training run.
- plot_type: precision-recall curve
- grouping_dimensions: x=`recall`; y=`precision`
- required_inputs: resolved MLflow run with patient prediction probabilities from `*_patient_predictions.csv` or nested `fold_patient_predictions.json`
- acceptance_criteria: Generate a standalone patient PR figure with PR-AUC annotation and prevalence baseline.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/performance/patient_pr_curve.png`
- notes: Skipped automatically if patient probabilities are unavailable or only one class is present.

### plot-0005
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show patient-level confusion structure for one resolved MLflow training run.
- plot_type: annotated confusion matrix
- grouping_dimensions: x=`predicted_label`; y=`true_label`
- required_inputs: resolved MLflow run with patient true/predicted labels from `*_patient_predictions.csv` or nested `fold_patient_predictions.json`
- acceptance_criteria: Generate a confusion matrix showing raw counts and row-normalized percentages.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/performance/patient_confusion_matrix.png`
- notes:

### plot-0006
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Summarize patient performance metrics with confidence intervals for one resolved run.
- plot_type: metric summary bar plot
- grouping_dimensions: x=`metric`; y=`score`
- required_inputs: `clinical_metrics_summary.json` patient metrics and confidence intervals, with bootstrap fallback from patient predictions when a CI is missing
- acceptance_criteria: Generate a bar plot covering balanced accuracy, ROC-AUC, PR-AUC, F1, and MCC with CI error bars.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/performance/patient_metric_summary.png`
- notes: Handles metrics such as MCC that can extend below zero.

### plot-0007
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show how patient fold accuracy varies across outer folds.
- plot_type: violin plus jitter distribution plot
- grouping_dimensions: x=`patient_fold_accuracy`; y=`accuracy`
- required_inputs: `clinical_metrics_summary.json` fold metrics or reconstructable patient fold rows
- acceptance_criteria: Generate a fold-level performance distribution figure for the resolved run.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/fold_behavior/patient_fold_accuracy_distribution.png`
- notes:

### plot-0008
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show the most stable features ranked by overall selection frequency.
- plot_type: horizontal bar plot
- grouping_dimensions: y=`feature`; x=`selection_frequency`
- required_inputs: `feature_selection_stability.json` or reconstructed fold feature-selection rows
- acceptance_criteria: Generate a top-feature selection-frequency figure for the resolved run.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_stability/selection_frequency_top_features.png`
- notes:

### plot-0009
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show the global distribution of feature selection frequencies.
- plot_type: histogram
- grouping_dimensions: x=`selection_frequency`; y=`feature_count`
- required_inputs: `feature_selection_stability.json` or reconstructed fold feature-selection rows
- acceptance_criteria: Generate a histogram summarizing overall selection sparsity versus stability.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_stability/selection_frequency_distribution.png`
- notes:

### plot-0010
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Visualize pairwise Jaccard overlap across fold feature sets.
- plot_type: heatmap
- grouping_dimensions: x=`fold`; y=`fold`
- required_inputs: nested fold `selected_features_list` artifacts for the resolved run
- acceptance_criteria: Generate a fold-by-fold Jaccard heatmap when at least two fold feature sets are available.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_stability/pairwise_jaccard_heatmap.png`
- notes: Skipped automatically if there are fewer than two usable fold feature sets.

### plot-0011
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Summarize the distribution of pairwise Jaccard similarities across folds.
- plot_type: histogram
- grouping_dimensions: x=`pairwise_jaccard`; y=`pair_count`
- required_inputs: nested fold `selected_features_list` artifacts for the resolved run
- acceptance_criteria: Generate a histogram of pairwise Jaccard values when at least one fold pair is available.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_stability/pairwise_jaccard_distribution.png`
- notes: Skipped automatically if there are not enough fold feature sets.

### plot-0012
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Report Kuncheva stability for the resolved run.
- plot_type: summary bar plot
- grouping_dimensions: x=`summary_statistic`; y=`kuncheva_index`
- required_inputs: `feature_selection_stability.json` Kuncheva fields
- acceptance_criteria: Generate a figure showing mean and median Kuncheva values when available.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_stability/kuncheva_summary.png`
- notes: Skipped automatically when Kuncheva cannot be computed.

### plot-0013
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Map class-conditional biomarker bias against overall stability.
- plot_type: scatter plot
- grouping_dimensions: x=`delta_remission_minus_non_remission`; y=`selection_frequency`; color=`cohens_d_mean`
- required_inputs: feature-selection summary with class-conditional selection fields and effect-stability rows
- acceptance_criteria: Generate a delta-vs-frequency scatter for the resolved run and label the highest-frequency features.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_interpretation/delta_vs_frequency_scatter.png`
- notes:

### plot-0014
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Compare remission and non-remission conditional selection rates feature by feature.
- plot_type: paired horizontal bar plot
- grouping_dimensions: y=`feature`; x=`conditional_selection_frequency`
- required_inputs: feature-selection summary with `share_remission` and `share_non_remission`
- acceptance_criteria: Generate a side-by-side conditional selection bar chart for the top features.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_interpretation/class_conditional_selection_bars.png`
- notes:

### plot-0015
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show the global distribution of class-conditional selection deltas.
- plot_type: histogram
- grouping_dimensions: x=`delta_remission_minus_non_remission`; y=`feature_count`
- required_inputs: feature-selection summary with class-conditional selection fields
- acceptance_criteria: Generate a delta histogram for the resolved run.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_interpretation/delta_distribution.png`
- notes:

### plot-0016
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show mean effect size and fold-wise variability for top biomarker candidates.
- plot_type: horizontal bar plot with error bars
- grouping_dimensions: y=`feature`; x=`cohens_d_mean`
- required_inputs: `feature_selection_stability.json` effect-stability rows
- acceptance_criteria: Generate an effect-size plot with fold-wise standard-deviation bars.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_interpretation/effect_size_top_features.png`
- notes:

### plot-0017
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show how consistently each top biomarker keeps the same effect direction across folds.
- plot_type: horizontal bar plot
- grouping_dimensions: y=`feature`; x=`sign_consistency`
- required_inputs: `feature_selection_stability.json` effect-stability rows
- acceptance_criteria: Generate a sign-consistency figure for top biomarker candidates.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_interpretation/sign_consistency_top_features.png`
- notes:

### plot-0018
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Compare biomarker stability with mean effect size in one view.
- plot_type: scatter plot
- grouping_dimensions: x=`selection_frequency`; y=`cohens_d_mean`; color=`effect_sign`
- required_inputs: merged feature-selection summary and effect-stability rows
- acceptance_criteria: Generate a selection-frequency-vs-effect-size scatter for the resolved run.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/biomarker_interpretation/effect_size_vs_frequency_scatter.png`
- notes:

### plot-0019
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Show probability calibration of patient-level predictions for one resolved run.
- plot_type: calibration curve
- grouping_dimensions: x=`mean_predicted_probability`; y=`observed_remission_frequency`
- required_inputs: patient prediction probabilities with enough distinct values
- acceptance_criteria: Generate a calibration plot when there is enough probability variation to bin the held-out predictions.
- implementation_refs:
  - script: `scripts/plot_paper_figures.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_figures/<selection_slug>/fold_behavior/patient_calibration_plot.png`
- notes: Skipped automatically if there are too few held-out participants or too little probability variation.

### plot-0020
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Package the primary single-run performance figures into one manuscript-ready composite panel.
- plot_type: 2x2 multi-panel composite
- grouping_dimensions: panels=`roc`, `pr`, `confusion`, `metric_summary`
- required_inputs: resolved run selection from `sweeps/artifacts/*.results.jsonl` plus single-panel paper figures under `sweeps/paper_figures/<selection_slug>/`
- acceptance_criteria: Generate a main-results composite figure with panel labels and write a panel manifest recording the resolved run and any missing source panels.
- implementation_refs:
  - script: `scripts/compose_paper_panels.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_panels/<selection_slug>/main_results_composite.png`
- notes: Supports best-overall, explicit MLflow run id, and explicit run-signature selection modes.

### plot-0021
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Package the stability figures into one manuscript-ready biomarker-stability composite.
- plot_type: 2x2 multi-panel composite
- grouping_dimensions: panels=`selection_frequency`, `jaccard_heatmap`, `jaccard_distribution`, `kuncheva_summary`
- required_inputs: resolved run selection from `sweeps/artifacts/*.results.jsonl` plus single-panel paper figures under `sweeps/paper_figures/<selection_slug>/`
- acceptance_criteria: Generate a biomarker-stability composite figure with placeholders when a source panel is unavailable and record missing inputs in the manifest.
- implementation_refs:
  - script: `scripts/compose_paper_panels.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_panels/<selection_slug>/biomarker_stability_composite.png`
- notes: Supports best-overall, explicit MLflow run id, and explicit run-signature selection modes.

### plot-0022
- requested_on: 2026-03-24
- requested_by: user
- status: implemented
- comparison_goal: Package the class-conditional biomarker interpretation figures into one manuscript-ready composite.
- plot_type: 2x2 multi-panel composite
- grouping_dimensions: panels=`delta_scatter`, `effect_size`, `sign_consistency`, `class_conditional_bars`
- required_inputs: resolved run selection from `sweeps/artifacts/*.results.jsonl` plus single-panel paper figures under `sweeps/paper_figures/<selection_slug>/`
- acceptance_criteria: Generate a biomarker-interpretation composite figure with panel labels and write a panel manifest recording the resolved run and any missing source panels.
- implementation_refs:
  - script: `scripts/compose_paper_panels.py`
  - notebook:
- artifact_outputs:
  - `sweeps/paper_panels/<selection_slug>/biomarker_interpretation_composite.png`
- notes: Supports best-overall, explicit MLflow run id, and explicit run-signature selection modes.
