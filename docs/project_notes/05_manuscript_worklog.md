# Manuscript Worklog

Persistent log for the LaTeX manuscript and supporting paper assets.

## Manuscript State

- working_title: `Low Montage Resting-State EEG Biomarker Discovery of tDCS Treatment Response in Adults with Treatment-Resistant Depression`
- status: `drafting`
- started_on_utc: `2026-03-25`
- primary_story: `Sparse, interpretable short-timescale EEG biomarker discovery under strict patient-level validation`
- primary_best_run:
  - model: `advanced_hybrid_1dcnn_lstm`
  - mlflow_run_id: `561eb29a046946818320bea18f21bed6`
  - window_seconds: `6`
  - inner_k: `1`
  - outer_k: `10`
- baseline_best_run:
  - model: `svm_linear`
  - mlflow_run_id: `e0f3834332d545da893277f039d28e4b`
  - window_seconds: `8`
  - inner_k: `30`
  - outer_k: `10`
- literature_pool:
  - bibtex_path: `paper/references_raw.bib`
  - curated_bibtex_paths: `paper/references.bib`, `paper/references_analysis_subset.bib`
  - raw_search_dir: `paper/literature/raw`
  - current_entry_count: `55`
  - total_entries_across_curated_sources: `90`
  - cited_entry_count_in_manuscript: `61`
  - note: `Continue searching during drafting; do not treat the current bibliography pool as final or complete. Curated manuscript builds should cite explicit sources rather than relying on nocite-all behavior.`

## Evidence Lock

- sweep_scope:
  - successful_runs: `150`
  - models: `advanced_hybrid_1dcnn_lstm`, `svm_linear`
  - window_sizes_seconds: `2, 4, 6, 8, 10`
  - inner_k_values: `1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70`
  - outer_k: `10`
  - fs_method: `select_k_best_f_classif`
  - equalize_lopo_groups: `true`
  - use_smote: `true`
- locked_hybrid_metrics:
  - patient_roc_auc: `0.8163265306122449`
  - patient_pr_auc: `0.5786641929499072`
  - patient_balanced_accuracy: `0.7857142857142857`
  - patient_accuracy: `0.8095238095238095`
  - patient_f1: `0.7142857142857143`
  - patient_mcc: `0.5714285714285714`
  - permutation_p_value: `0.007984031936127744`
- locked_hybrid_feature_selection:
  - average_features_per_fold: `1.0`
  - unique_feature_count: `7`
  - mean_pairwise_jaccard: `0.19047619047619047`
  - kuncheva_index_mean: `0.18721198156682026`
  - mean_sign_consistency: `1.0`
- locked_svm_metrics:
  - patient_roc_auc: `0.5102040816326531`
  - patient_pr_auc: `0.4110644257703081`
  - patient_balanced_accuracy: `0.6428571428571428`
  - patient_accuracy: `0.6666666666666666`
  - patient_f1: `0.5333333333333333`
  - patient_mcc: `0.2773500981126146`
  - permutation_p_value: `0.469061876247505`
- evidence_files:
  - `paper/data/manuscript_facts.json`
  - `paper/data/best_run_comparison.csv`
  - `paper/data/hybrid_recurrent_biomarkers.csv`
  - `paper/assets/paper_figures/run-561eb29a__model-advanced-hybrid-1dcnn-lstm__ws-6__ik-1__ok-10__eq-true__sm-true/figure_manifest.json`
  - `paper/assets/paper_figures/run-e0f38343__model-svm-linear__ws-8__ik-30__ok-10__eq-true__sm-true/figure_manifest.json`

## Figure And Table Map

- figure_1: `paper/figures/figure1_sweep_overview.png`
- figure_2: `paper/figures/figure2_main_results.png`
- figure_3: `paper/figures/figure3_biomarker_stability.png`
- figure_4: `paper/figures/figure4_biomarker_interpretation.png`
- table_1: `paper/tables/cohort_summary.tex`
- table_2: `paper/tables/best_run_comparison.tex`
- table_3: `paper/tables/hybrid_recurrent_biomarkers.tex`

## Claim Guardrails

- Do not claim definitive clinical prediction or clinical readiness.
- Do not describe the biomarkers as highly stable across folds or as definitive clinical markers; `recurrent`, `plausible`, `credible in this cohort`, and `consistent effect direction` are acceptable framings.
- Use class-conditional deltas more heavily than ratios when zero counts inflate ratios.
- Re-open sweep plots, MLflow metrics, and composite figures whenever the narrative changes.
- Treat the bottom-row feature-selection trends in Figure 1 as shared pipeline behavior, not classifier-specific behavior.
- Treat `/Users/abin/Paper/paper2-body.tex` as an outdated cross-check on the same data, not as an authoritative methodology source.
- Use acquisition details from legacy materials only when corroborated by the current repo configs or artifacts, or when the user explicitly requests that they be carried into the manuscript as acquisition context.
- Keep legacy acquisition descriptors separate from the current sweep preprocessing description.
- Use colleague papers from the same data program as corroborative context and regional plausibility support, not as direct performance benchmarks.

## Ongoing Search Backlog

- Find more low-density or wearable resting-state EEG references that are directly tied to affective disorders.
- Find more tDCS-specific EEG response-prediction papers rather than broader neuromodulation biomarker papers.
- Add methodological references on feature-selection stability and biomarker reproducibility for the discussion section.
- Prune clearly irrelevant high-citation search hits from the working bibliography during later passes.

## Session Log

### 2026-03-25

- generated best-run paper figures for the hybrid and SVM sweep winners
- composed manuscript-ready hybrid composites for performance, biomarker stability, and biomarker interpretation
- added `scripts/build_paper_summary_assets.py` to export the manuscript summary figure, tables, and facts JSON
- verified that the hybrid best run uses `6 s` windows and `inner_k=1`, while the SVM best run uses `8 s` windows and `inner_k=30`
- verified that the hybrid run has modest hard-set overlap (`mean_pairwise_jaccard=0.190`) but perfect mean effect-direction consistency (`1.0`)
- patched `paper-scripts/search_citations.py` so `.env` is auto-loaded, CORE works without shell exporting, and Semantic Scholar failures are surfaced more clearly
- expanded the working bibliography pool to `56` entries across depression, tDCS, EEG biomarker, and treatment-response queries
- re-opened and visually checked Figures 1--4; current narrative remains consistent with the visuals:
  - Figure 1 supports a hybrid advantage concentrated in sparse settings rather than uniformly across the sweep
  - Figure 3 shows modest hard-set overlap, not strong foldwise stability
  - Figure 4 supports stronger effect-direction consistency than selection-frequency universality
- used `/Users/abin/Paper/analysis.bib` as a legacy bibliography source and extracted a local curated subset into `paper/references_analysis_subset.bib`
- treated `/Users/abin/Paper/paper2-body.tex` as an outdated manuscript draft and only reused acquisition details that were also consistent with the current processing config
- expanded the manuscript to `51` explicit citations across `paper/references.bib` and `paper/references_analysis_subset.bib`
- switched the draft away from `\nocite{*}` and verified that the curated LaTeX build succeeds at `paper/build_curated/main.pdf`
- imported the user-requested acquisition details from `/Users/abin/Paper/paper2-body.tex` into the methods section and regenerated `paper/tables/cohort_summary.tex` so the cohort table now includes demographics, recording structure, export length, and window counts with an explicit `woodham2025home` source row
- validated that the best-hybrid Jaccard heatmap and histogram in the stability figure are reconstructed from nested MLflow fold selections rather than stale image assets; for `inner_k=1`, the fold-pair Jaccard values are expected to be binary because each fold selects one feature
- decluttered `paper/figures/figure1_sweep_overview.png` and updated the shared feature-selection aggregation so matched model duplicates no longer double-count the bottom-row confidence bands
- reduced the biomarker-stability composite to panels A and D for the manuscript because the verified Jaccard heatmap and histogram are visually unhelpful in the sparse best-run setting
- re-read the same-data colleague papers from the local Zotero store and revised the introduction/discussion so the manuscript now connects its TP10 and frontal-temporal recurrent candidates to prior PSD-based and PLV-based findings from the same home-based 4-channel program without direct metric benchmarking
- added `Moncy2025BD` to `paper/references_analysis_subset.bib` as supportive adjacent-context evidence for low-montage AF7 and TP10 signal relevance in bipolar depression

### 2026-03-27

- added a new citation cluster to support the manuscript's methods-rigor framing:
  - `Widge2019` for treatment-response biomarker reproducibility limits and publication bias
  - `Shim2021` for leakage from feature selection
  - `Chawla2002` for SMOTE
  - `Shen2025` for data-centric and interpretable EEG modeling
- revised the introduction and methods so the methodological novelty is now framed as the integrated pipeline:
  - participant-level outer validation
  - fold-internal selection and rebalancing
  - sparse `inner-k` sweep
  - recurrence and effect-direction summaries
- expanded the biomarker discussion with explicit source-backed interpretation of the main recurrent features:
  - `tp10_spectral_entropy` now linked to broader entropy and complexity work, with mixed directionality noted via `Jaworska2018` and `Lord2023`
  - frontal-temporal beta and gamma difference features now linked to temporal-region and higher-frequency asymmetry literature via `Mahato2020` and `Mumtaz2017`
  - `af7_zero_crossings` now treated explicitly as an exploratory frontal marker with only indirect support from adjacent AF7 and frontal-asymmetry literature
- added verified BibTeX entries for:
  - `Widge2019`, `Shim2021`, `Chawla2002`, `Mumtaz2017`, `Shen2025`, `Jaworska2018`, `Lord2023`, `Monni2022`, `Acharya2015`, `Faust2014`
- rebuilt the curated manuscript successfully at `paper/build_curated/main.pdf`
- added implementation-grounded feature-extraction detail to the methods section:
  - stated the full `247` EEG-derived measures per window
  - broke the extractor down into `156` channel-local measures and `91` regional or synchrony-derived measures
  - described the per-channel composition as `21` spectral, `12` time-domain or Hjorth, and `6` entropy or nonlinear complexity measures
- added a manuscript table summarizing the extracted feature families, counts, and computational rationale so the feature space is easier to parse without relying on prose alone
- removed `outer-k` and `consensus feature budget` wording from the manuscript narrative and cohort summary, since those settings are not part of the interpreted results
- added a generated end-to-end processing flowchart to the methods section and saved the source generator at `scripts/generate_processing_flowchart.py`
- expanded the model description with implementation-grounded detail:
  - clarified that the hybrid path begins with a 128-unit feature embedding, then reshapes into a learned one-dimensional sequence
  - described the three CNN blocks, two bidirectional LSTM layers, multi-head attention, feature-pyramid fusion, and dense head from the actual trainer code
  - added a compact model-configuration table so the operative SVM and hybrid settings are visible without relying on config-file inspection
- reduced repeated same-program citation pairings so the manuscript now uses the PSD-based and PLV-based same-program studies more selectively
- shifted the biomarker wording slightly upward from `candidate` language toward `plausible` and `credible in this cohort`, while still preserving the no-definitive-biomarker guardrail
- corrected the lingering `Ulrich2025` citation-key mismatch and revalidated the curated LaTeX build; only the pre-existing underfull-box warnings and the empty-journal warning for `babiloni2012_guidelines` remain
