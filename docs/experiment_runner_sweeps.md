# Experiment Runner Sweep Guide

This guide documents how to run sweeps with:
- `scripts/run_experiments.py` CLI flags
- shell loops for `inner-k`, `outer-k`, and `window-size`

All commands assume repo root as working directory.

Methodology note:
- Outer evaluation remains Leave-One-Participant/Group-Out throughout.
- `--inner-k` controls how many features are selected on each outer training fold.
- `--outer-k` controls how many consensus features are kept at the end from correctly predicted outer folds.
- These flags do not change the number of CV splits.

## 1) Quick Start

```bash
./scripts/run_experiments.py \
  --config eeg_analysis/configs/model_config.yaml \
  --processing-config eeg_analysis/configs/processing_config.yaml \
  --model-sets traditional \
  --mode baseline \
  --window-sizes 10
```

Use `--dry-run` first to validate command expansion:

```bash
./scripts/run_experiments.py \
  --config eeg_analysis/configs/model_config.yaml \
  --model-sets traditional \
  --mode baseline \
  --window-sizes 10 \
  --dry-run
```

## 2) Sweep with CLI Flags (Single Command)

### Sweep windows + models + FS methods + feature counts

```bash
./scripts/run_experiments.py \
  --config eeg_analysis/configs/model_config.yaml \
  --processing-config eeg_analysis/configs/processing_config.yaml \
  --model-sets traditional,boosting,deep_learning \
  --mode both \
  --fs-methods select_k_best_f_classif,select_k_best_mutual_info \
  --feature-counts 5,10,20 \
  --window-sizes 2,4,6,8,10 \
  --ordering sequential
```

### Sweep per-fold and consensus feature counts for one fixed run

```bash
./scripts/run_experiments.py \
  --config eeg_analysis/configs/model_config.yaml \
  --model-sets traditional \
  --mode fs \
  --fs-methods select_k_best_f_classif \
  --feature-counts 10 \
  --window-sizes 10 \
  --inner-k 5 \
  --outer-k 10
```

## 3) Sweep with Shell Loops

### Triple loop: window-size × inner-k × outer-k

```bash
for ws in 2 4 6 8 10; do
  for ik in 2 4 6 8 10; do
    for ok in 2 4 6 8 10; do
      ./scripts/run_experiments.py \
        --config eeg_analysis/configs/model_config.yaml \
        --processing-config eeg_analysis/configs/processing_config.yaml \
        --model-sets traditional \
        --mode fs \
        --fs-methods select_k_best_f_classif,select_k_best_mutual_info \
        --feature-counts 10 \
        --window-sizes "$ws" \
        --ordering sequential \
        --inner-k "$ik" \
        --outer-k "$ok"
    done
  done
done
```

### Use a fixed dataset run ID in loop sweeps

```bash
for ws in 2 4 6 8 10; do
  for ik in 5 10 15; do
    for ok in 5 10 15; do
      ./scripts/run_experiments.py \
        --config eeg_analysis/configs/model_config.yaml \
        --model-sets traditional \
        --mode fs \
        --fs-methods select_k_best_f_classif \
        --feature-counts 10 \
        --window-sizes "$ws" \
        --dataset-run-id 962a96b061dd45c1b005fc48c7e6e895 \
        --inner-k "$ik" \
        --outer-k "$ok"
    done
  done
done
```

## 4) Key Flags

- `--model-sets`: choose model families (`traditional`, `boosting`, `deep_learning`, `all`)
- `--models`: explicit model names (overrides `--model-sets`)
- `--mode`: `baseline`, `fs`, or `both`
- `--fs-methods`: comma-separated feature selection methods
- `--feature-counts`: comma-separated feature counts
- `--window-sizes`: comma-separated window sizes
- `--inner-k`: features to select within each outer LOPO training fold
- `--outer-k`: final consensus features kept from correctly predicted outer folds
- `--ordering`: dataset ordering requirement (`sequential` or `completion`)
- `--dataset-run-id`: pin dataset lineage for all launched jobs
- `--dry-run`: print planned jobs without executing
- `--stop-on-error`: stop at first failed job

## 5) Sweep Count Formula

Per invocation of `run_experiments.py`:
- `jobs = (#models) * (#windows) * (baseline_jobs + fs_jobs)`
- `baseline_jobs = 1` if mode is `baseline` or `both`, else `0`
- `fs_jobs = (#fs_methods * #feature_counts)` if mode is `fs` or `both`, else `0`

With outer shell loops:
- `total_jobs = jobs * (#window loop) * (#inner-k loop) * (#outer-k loop)`

## 6) Practical Workflow

1. Start with `--dry-run` and verify model/window expansion.
2. Run small subsets first (`--models` or one `--window-sizes` value).
3. Launch full sweep once command expansion and job count look correct.

## 7) Checkpoint / Resume Across Outer Loops

`run_experiments.py` derives artifact names from a normalized command payload and
stores per-signature files under `--artifacts-dir` by default:
- `<stem>__<sig12>.checkpoint.json`
- `<stem>__<sig12>.results.jsonl`

The signature includes sweep-defining args such as `window-sizes`, `inner-k`, and `outer-k`,
so rerunning outer shell loops resumes each combination independently even when all invocations
share the same artifacts directory.
Whitespace/casing-only argument differences normalize to the same signature.

Example:

```bash
for ws in 2 4 6 8 10; do
  for ik in 2 4 6 8 10; do
    for ok in 2 4 6 8 10; do
      ./scripts/run_experiments.py \
        --config eeg_analysis/configs/model_config.yaml \
        --processing-config eeg_analysis/configs/processing_config.yaml \
        --model-sets traditional \
        --mode fs \
        --fs-methods select_k_best_f_classif,select_k_best_mutual_info \
        --feature-counts 10 \
        --window-sizes "$ws" \
        --ordering sequential \
        --inner-k "$ik" \
        --outer-k "$ok" \
        --artifacts-dir sweeps/artifacts
    done
  done
done
```

Checkpoint/results control flags:
- `--artifacts-dir <path>`: base directory for auto-generated checkpoint/results files
- `--checkpoint-file <path>`: explicit checkpoint path override
- `--results-file <path>`: explicit results path override
- `--no-resume`: do not skip completed jobs for this invocation
- `--reset-checkpoint`: clear saved state for this command signature, then run
- `--disable-checkpoint`: turn checkpointing off
- `--disable-results-ledger`: disable results ledger writes

Results ledger (`*.results.jsonl`) schema notes:
- Append-only attempt records per job key (`attempt_index` increments across retries)
- Includes status, command context, dataset/model/FS metadata, and return code
- Includes MLflow linkage/snapshot when a train run id is available:
  `mlflow_run_id`, `mlflow_experiment_id`, `mlflow_params`, `mlflow_metrics`, `mlflow_tags`
