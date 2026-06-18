#!/usr/bin/env bash
# ==========================================================================
# Reviewer Response Experiments: CPU Baselines
# ==========================================================================
# Runs 8 CPU-safe classifiers at the best sweep configuration.
# GPU experiments documented at the bottom.
#
# Usage:
#   bash scripts/reviewer_experiments.sh              (dry-run: show commands)
#   bash scripts/reviewer_experiments.sh --run-cpu    (execute CPU experiments)
# ==========================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP_DIR="$ROOT/sweeps/reviewer_experiments"
CFG="$ROOT/eeg_analysis/configs/model_config.yaml"
PROC_CFG="$ROOT/eeg_analysis/configs/processing_config.yaml"

mkdir -p "$SWEEP_DIR"

MODELS="random_forest,gradient_boosting,extra_trees,svm_rbf,svm_linear,knn,decision_tree,ada_boost"

CMD=(
    uv run python3 "$ROOT/scripts/run_experiments.py"
    --config "$CFG"
    --processing-config "$PROC_CFG"
    --models "$MODELS"
    --mode fs
    --fs-methods select_k_best_f_classif
    --feature-counts 10
    --window-sizes 6
    --ordering sequential
    --inner-k 1
    --outer-k 10
    --equalize-lopo-groups true
    --use-smote true
    --artifacts-dir "$SWEEP_DIR"
    --stop-on-error
    --dataset-run-id f7cc3f2991cf46b58bbb3528f4944b25
    --reset-checkpoint
)

echo "=============================================="
echo "REVIEWER RESPONSE: CPU BASELINES (R5#7)"
echo "=============================================="
echo "Models: $MODELS"
echo "Configuration: ws=6s, inner-k=1, SMOTE=true, LOPO-equalized"
echo "Artifacts: $SWEEP_DIR"
echo ""

if [[ "${1:-}" == "--run-cpu" ]]; then
    echo ">>> EXECUTING..."
    "${CMD[@]}"
    echo ""
    echo "=============================================="
    echo "DONE. Check results:"
    echo "  ls $SWEEP_DIR/*.results.jsonl"
    echo ""
    echo "To analyze, use:"
    echo "  uv run python3 scripts/plot_sweep_roc_auc.py \\"
    echo "    --artifacts-dir $SWEEP_DIR \\"
    echo "    --output-dir $SWEEP_DIR/plots \\"
    echo "    --preset performance_core"
else
    echo ">>> DRY RUN. Use --run-cpu to execute."
    echo ""
    echo "${CMD[@]}"
    echo ""
    echo ""
    echo "=============================================="
    echo "GPU EXPERIMENTS (run when GPU becomes available)"
    echo "=============================================="
    echo ""
    echo "# R5#2: Noise sanity check — hybrid with --gaussian-noise 0.5"
    echo "  uv run python3 $ROOT/eeg_analysis/run_pipeline.py --config $CFG train \\"
    echo "    --window-size 6 --model-type advanced_hybrid_1dcnn_lstm \\"
    echo "    --use-dataset-from-run f7cc3f2991cf46b58bbb3528f4944b25 \\"
    echo "    --enable-feature-selection --n-features-select 10 \\"
    echo "    --fs-method select_k_best_f_classif --inner-k 1 --outer-k 10 \\"
    echo "    --equalize-lopo-groups true --use-smote true"
    echo "    # + add --gaussian-noise 0.5 (if flag exists)"
    echo ""
    echo "# R5#4: No-SMOTE ablation"
    echo "  (same command but --use-smote false)"
    echo ""
    echo "# R2#7: GAP architecture variant"
    echo "  (same command but --model-type hybrid_1dcnn_lstm_gap)"
    echo ""
    echo "# R5#7: MLP baseline"
    echo "  (same command but --model-type pytorch_mlp)"
fi
