# Metrics Reference

This document describes the metrics currently produced by the traditional and deep-learning training routes in the unified EEG pipeline.

## Where Metrics Are Logged

- `clinical_metrics_summary.json`: full structured report for patient-level, window-level, fold-level, statistical, and feature-selection metrics.
- `feature_selection_stability.json`: feature-selection and biomarker-stability subset of the report.
- MLflow scalar metrics: overall metrics and summary-safe aggregates flattened from the structured report.
- Nested fold artifacts:
  - `fold_patient_predictions.json`
  - `fold_window_predictions.json`

Class convention used throughout this document:

- `1` = remission / positive class
- `0` = non-remission / negative class

## Patient-Level Metrics

These are the primary outcome metrics reported over the concatenated held-out participant predictions across outer folds.

| Metric | Meaning |
| --- | --- |
| `accuracy` | Fraction of held-out participants classified correctly. |
| `balanced_accuracy` | Mean of sensitivity and specificity. Useful when class counts are imbalanced. Returns `null` if only one class is present. |
| `precision` | Of the participants predicted as remission, the fraction truly in remission. |
| `recall` | Fraction of true remission participants recovered by the model. |
| `sensitivity` | Same value as `recall`; retained as an explicitly clinical name. |
| `specificity` | Fraction of true non-remission participants correctly identified as non-remission. |
| `f1` | Harmonic mean of precision and recall. |
| `roc_auc` | Area under the ROC curve using participant-level probabilities. Participant probability is the mean window probability for that participant. This is the default primary metric. |
| `pr_auc` | Area under the precision-recall curve using participant-level probabilities. |
| `npv` | Negative predictive value: of participants predicted as non-remission, the fraction truly non-remission. |
| `mcc` | Matthews correlation coefficient. Correlation-style binary metric that remains informative under imbalance. |
| `true_positives`, `true_negatives`, `false_positives`, `false_negatives` | Participant-level confusion counts. |
| `n_patients` | Number of held-out participant predictions contributing to the report. |

Important detail:

- `roc_auc` and `pr_auc` use aggregated participant probabilities.
- Hard-label metrics use the participant-level predicted label.

## Window-Level Metrics

These are supporting metrics computed over the concatenated held-out window predictions across outer folds.

| Metric | Meaning |
| --- | --- |
| `accuracy` | Fraction of held-out windows classified correctly. |
| `f1` | Harmonic mean of window-level precision and recall. |
| `roc_auc` | Area under the ROC curve using held-out window probabilities. Enabled by default. |
| `true_positives`, `true_negatives`, `false_positives`, `false_negatives` | Window-level confusion counts. |
| `n_windows` | Number of held-out windows contributing to the report. |

## Fold-Level Metrics

Both patient and window reporting now include explicit per-fold metric tables built from raw held-out predictions.

Artifacts and report fields:

- `patient.fold_metrics`
- `patient.fold_metric_summary`
- `window.fold_metrics`
- `window.fold_metric_summary`
- nested fold artifacts:
  - `fold_patient_predictions.json`
  - `fold_window_predictions.json`

Per-fold patient metrics:

- `accuracy`
- `balanced_accuracy`
- `precision`
- `recall`
- `sensitivity`
- `specificity`
- `f1`
- `roc_auc`
- `pr_auc`
- `npv`
- `mcc`
- confusion counts
- `n_patients`

Per-fold window metrics:

- `accuracy`
- `f1`
- `roc_auc`
- confusion counts
- `n_windows`

Per-fold summary fields:

| Field | Meaning |
| --- | --- |
| `mean` | Mean metric value across folds where that metric was defined. |
| `std` | Population standard deviation across folds where that metric was defined. |
| `n_folds` | Number of folds that contributed a valid value for that metric. |

Notes:

- Some fold metrics can be `null` when the held-out fold does not contain both classes, especially `balanced_accuracy`, `roc_auc`, and `pr_auc`.
- In the current LOPO setup, each fold usually contains one held-out participant, so patient-level fold metrics should be interpreted mainly as fold diagnostics rather than stable standalone performance estimates.

## Statistical Diagnostics

The report also includes statistical confidence and permutation diagnostics for patient-level metrics.

### Bootstrap confidence intervals

Enabled by `metrics_reporting.bootstrap_ci_enabled`.

Fields:

- `patient.confidence_intervals.<metric>.mean`
- `patient.confidence_intervals.<metric>.std`
- `patient.confidence_intervals.<metric>.ci_low`
- `patient.confidence_intervals.<metric>.ci_high`
- `patient.confidence_intervals.<metric>.valid_samples`

Metrics currently bootstrapped:

- `accuracy`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `f1`
- `roc_auc`

Meaning:

- bootstrap resamples the held-out participant predictions with replacement and recomputes the metric.
- `ci_low` and `ci_high` are the empirical interval bounds from the configured confidence level.

### Permutation test

Enabled by `metrics_reporting.permutation_test_enabled`.

Fields:

- `stats.permutation_test.metric_name`
- `stats.permutation_test.observed`
- `stats.permutation_test.p_value`
- `stats.permutation_test.null_mean`
- `stats.permutation_test.null_std`
- `stats.permutation_test.iterations`

Meaning:

- the held-out participant labels are permuted repeatedly
- the configured metric is recomputed under the null
- the reported `p_value` is the one-sided proportion of null scores greater than or equal to the observed score, with the standard `+1 / +1` correction

## Feature-Selection Stability And Biomarker Metrics

These metrics summarize outer-fold feature-selection behavior. They are descriptive and post hoc. They do not change model fitting.

Top-level report fields:

- `feature_set_count`
- `average_features_per_fold`
- `mean_pairwise_jaccard`
- `median_pairwise_jaccard`
- `unique_feature_count`
- `top_feature`
- `top_feature_frequency`
- `top_feature_share`
- `selected_features_per_fold`
- `fixed_k_detected`
- `kuncheva_index_mean`
- `kuncheva_index_median`
- `kuncheva_reason`
- `fold_class_counts`
- `class_conditional_selection_available`
- `class_conditional_selection_reason`

Meaning of the main feature-set metrics:

| Metric | Meaning |
| --- | --- |
| `feature_set_count` | Number of outer folds that contributed a selected feature set. |
| `average_features_per_fold` | Mean number of selected features per outer fold. |
| `mean_pairwise_jaccard` | Mean pairwise Jaccard overlap of selected feature sets across folds. Higher means more stable hard selection. |
| `median_pairwise_jaccard` | Median pairwise Jaccard overlap. More robust to outlier folds. |
| `unique_feature_count` | Number of distinct features selected at least once. |
| `top_feature` | Most frequently selected feature across folds. |
| `top_feature_frequency` | Number of folds in which the top feature was selected. |
| `top_feature_share` | Share of folds in which the top feature was selected. |
| `selected_features_per_fold` | List of selected feature counts, one entry per contributing fold. |
| `fixed_k_detected` | `true` if each contributing fold selected the same number of features. |

## Consensus Features

The pipeline tracks two different feature-selection outputs:

- per-fold selected features
- final consensus selected features

These are related, but they are not the same object.

### Per-fold selected features

When feature selection is enabled, each outer training fold selects its own feature subset from the non-held-out training data.

Relevant controls:

- `inner_k`: target number of features selected inside each outer fold
- `feature_selection_method`: selector used inside each fold

Per-fold logging:

- nested fold run param: `selected_features_list`
- nested fold run param: `num_selected_features`
- nested fold run param: `feature_selection_method`

These per-fold feature sets are the inputs to the stability metrics documented below.

### Final consensus selected features

After the outer folds finish, the trainer builds one final feature list for the final full-data model fit.

Current rule:

1. start from the folds marked `correct`
2. count how often each feature appeared across those folds
3. rank features by frequency
4. keep the top `outer_k` features if `outer_k` is set, otherwise keep the configured feature-count target

Fallback behavior:

1. if there are no correctly predicted folds, use all fold feature sets instead
2. if there are no fold feature sets at all, fall back to all available features

Relevant controls:

- `inner_k`: per-fold feature count before consensus
- `outer_k`: final consensus feature count

Top-level MLflow logging for the final consensus list:

- param: `selected_features_list`
- param: `num_selected_features`
- param: `feature_selection_final_strategy = consensus_frequency`
- param: `feature_selection_consensus_source`
- param: `feature_selection_consensus_folds`
- artifact: `feature_selection_consensus_counts.json`

Meaning of the main consensus fields:

| Field | Meaning |
| --- | --- |
| `feature_selection_final_strategy` | Current final-selection rule. At present this is frequency-based consensus across fold selections. |
| `feature_selection_consensus_source` | Which fold pool supplied the consensus counts: `correct_folds`, `all_folds_fallback`, or `all_features_fallback`. |
| `feature_selection_consensus_folds` | Number of fold feature sets used to build the final consensus counts. |
| `selected_features_list` | Final consensus feature list on the top-level training run. |
| `feature_selection_consensus_counts.json` | Ranked feature-frequency counts used to form the final consensus set. |

Interpretation:

- per-fold selected features tell you what the selector chose in each outer fold
- stability metrics tell you how reproducible those selections were across folds
- the final consensus feature list is the feature set actually used to train the final model after cross-validation

This distinction matters:

- the stability report describes fold behavior
- the final consensus list describes deployment-time model training input

They are expected to be related, but they should not be treated as interchangeable.

### Per-feature selection frequency

Each entry in `selection_frequency_by_feature` contains:

| Field | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `count` | Number of folds in which the feature was selected. |
| `share` | `count / feature_set_count`. Overall selection frequency across folds. |
| `count_remission` | Number of remission folds in which the feature was selected. |
| `share_remission` | `count_remission / remission_folds`. Conditional selection frequency within remission folds. |
| `count_non_remission` | Number of non-remission folds in which the feature was selected. |
| `share_non_remission` | `count_non_remission / non_remission_folds`. Conditional selection frequency within non-remission folds. |
| `delta_remission_minus_non_remission` | `share_remission - share_non_remission`. Positive values mean the feature is selected more often in remission folds; negative values mean the opposite. |
| `ratio_remission_to_non_remission` | `share_remission / (share_non_remission + 1e-9)`. Ratio-style view of remission bias. |

Class-conditional availability fields:

| Field | Meaning |
| --- | --- |
| `fold_class_counts.remission_folds` | Number of folds whose held-out participant class was remission. |
| `fold_class_counts.non_remission_folds` | Number of folds whose held-out participant class was non-remission. |
| `fold_class_counts.mixed_folds` | Number of folds with mixed held-out classes. Normally zero in LOPO. |
| `fold_class_counts.unknown_folds` | Number of folds where held-out class could not be inferred. |
| `class_conditional_selection_available` | `true` when at least one remission fold and one non-remission fold are available. |
| `class_conditional_selection_reason` | Reason class-conditional statistics are unavailable, usually `insufficient_class_fold_coverage`. |

Interpretation:

- high `share` and large positive `delta_remission_minus_non_remission`: stable feature with remission association
- high `share` and large negative `delta_remission_minus_non_remission`: stable feature with non-remission association
- high `share` and near-zero `delta`: stable but not class-specific
- low `share` and large absolute `delta`: class-biased but unstable

### Kuncheva stability

Fields:

- `kuncheva_index_mean`
- `kuncheva_index_median`
- `kuncheva_reason`

Meaning:

- pairwise overlap metric adjusted for chance under fixed-size selection
- only available when:
  - at least two folds contributed feature sets
  - all folds selected the same number of features
  - candidate feature count was available and valid

If unavailable, `kuncheva_reason` explains why, for example:

- `insufficient_folds`
- `candidate_feature_count_unavailable`
- `variable_selected_feature_count`

### Ranking stability

Fields under `ranking_stability`:

- `available`
- `reason`
- `top_k`
- `mean_top_k_overlap`
- `median_top_k_overlap`
- `n_ranked_folds`

Meaning:

- compares the overlap of the top-ranked features across folds when ranking scores are available
- `top_k` comes from `metrics_reporting.ranking_stability_top_k`
- overlap is normalized by the smallest available top-k set size in each pair

### Effect stability

Fields under `effect_stability`:

- `available`
- `reason`
- `feature_count`
- `mean_sign_consistency`
- `features`

Each per-feature effect record contains:

| Field | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `fold_count` | Number of folds with valid effect statistics for that feature. |
| `selection_share` | Overall selection frequency of that feature. |
| `sign_consistency` | Dominant sign fraction across folds. `1.0` means the direction never changed. |
| `positive_effect_fraction` | Fraction of valid folds with positive Cohen's d. |
| `negative_effect_fraction` | Fraction of valid folds with negative Cohen's d. |
| `zero_effect_fraction` | Fraction of valid folds with effectively zero Cohen's d. |
| `cohens_d_mean` | Mean Cohen's d across valid folds. |
| `cohens_d_variance` | Variance of Cohen's d across valid folds. |
| `cohens_d_std` | Standard deviation of Cohen's d across valid folds. |

Meaning:

- effect stability captures whether a feature changes in a consistent direction across folds
- it complements selection frequency: stable selection does not guarantee stable effect direction

### Importance stability

Fields under `importance_stability`:

- `available`
- `reason`
- `feature_count`
- `mean_importance_variance`
- `features`

Each per-feature importance record contains:

| Field | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `fold_count` | Number of ranked folds contributing an importance value. |
| `importance_mean` | Mean ranking score / importance across folds. |
| `importance_variance` | Variance of the ranking score / importance across folds. |
| `importance_std` | Standard deviation of the ranking score / importance across folds. |

Meaning:

- available only when the selector or model emits fold-level ranking scores
- low variance means the ranking magnitude is stable across folds

### Resampling stability

Fields under `resampling_stability`:

- `enabled`
- `method`
- `iterations`
- `reason`
- `results`

Current behavior:

- schema exists for future repeated nested CV or bootstrap stability analysis
- default config keeps it disabled
- when enabled today, it reports `not_implemented` rather than executing extra resampling

## MLflow Scalar Logging

MLflow scalar metrics contain a summary-safe subset of the report. Important examples:

- `patient_<metric>`
- `window_<metric>`
- `patient_fold_<metric>_mean`
- `patient_fold_<metric>_std`
- `window_fold_<metric>_mean`
- `window_fold_<metric>_std`
- `feature_selection_mean_pairwise_jaccard`
- `feature_selection_kuncheva_index_mean`
- `feature_selection_mean_top_k_overlap`
- `feature_selection_effect_mean_sign_consistency`
- `feature_selection_importance_mean_variance`
- `feature_selection_remission_fold_count`
- `feature_selection_non_remission_fold_count`

Detailed per-feature and per-fold rows stay in JSON artifacts rather than being expanded into one MLflow scalar per row.
