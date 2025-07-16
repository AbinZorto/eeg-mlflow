#!/bin/bash

# EEG Analysis - Complete Model Training Script
# This script runs all available models with different feature selection configurations

set -e  # Exit on any error

# Parse command line arguments
DATASET_RUN_ID=""
SHOW_HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-run-id)
            DATASET_RUN_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Activate conda environment with GPU support
echo "Activating conda eeg-env environment (with GPU support)..."
source /home/abin/anaconda3/etc/profile.d/conda.sh
conda activate eeg-env

# Configuration
CONFIG_FILE="eeg_analysis/configs/window_model_config_ultra_extreme.yaml"  # ULTRA-EXTREME configuration for all models
LEVEL="window"
PYTHON_CMD="python"

# Available models - Mix of GPU-accelerated and traditional models
GPU_ML_MODELS=("xgboost_gpu" "catboost_gpu" "lightgbm_gpu")
GPU_DL_MODELS=("pytorch_mlp" "keras_mlp")
TRADITIONAL_MODELS=("random_forest" "svm_rbf")
MODELS=("${GPU_ML_MODELS[@]}" "${GPU_DL_MODELS[@]}" "${TRADITIONAL_MODELS[@]}")

# Feature selection configurations
FEATURE_SELECTION_METHODS=("select_k_best_f_classif" "select_k_best_mutual_info")
FEATURE_COUNTS=(10 15 20)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="experiment_log_$(date +%Y%m%d_%H%M%S).txt"

log_message() {
    local message="$1"
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $message" | tee -a "$LOG_FILE"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}[SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="$1"
    echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}[WARNING]${NC} $message" | tee -a "$LOG_FILE"
}

# Function to run a single experiment
run_experiment() {
    local model_type="$1"
    local use_fs="$2"
    local fs_method="$3"
    local n_features="$4"
    
    # Create intelligent experiment name based on model type and configuration
    local exp_name=""
    local experiment_suffix=""
    
    if [ "$use_fs" = "true" ]; then
        exp_name="${model_type}_fs_${fs_method}_${n_features}"
        experiment_suffix="_feature_selection"
    else
        exp_name="${model_type}_no_fs"
        experiment_suffix="_baseline"
    fi
    
        # Set experiment name based on model type category
    local mlflow_experiment=""
    if [[ "$model_type" == "pytorch_mlp" || "$model_type" == "keras_mlp" ]]; then
        mlflow_experiment="eeg_deep_learning_gpu${experiment_suffix}"
    elif [[ "$model_type" == "xgboost_gpu" || "$model_type" == "catboost_gpu" || "$model_type" == "lightgbm_gpu" ]]; then
        mlflow_experiment="eeg_boosting_gpu${experiment_suffix}"
    elif [[ "$model_type" == "random_forest" || "$model_type" == "svm_rbf" ]]; then
        mlflow_experiment="eeg_traditional_models${experiment_suffix}"
    else
        mlflow_experiment="eeg_other_models${experiment_suffix}"
    fi
    
    # Export experiment name as environment variable for run_pipeline.py to use
    export MLFLOW_EXPERIMENT_NAME="$mlflow_experiment"
    
    # Use the unified ultra-extreme configuration for all GPU models
    local config_file="$CONFIG_FILE"
    log_message "Using ULTRA-EXTREME GPU configuration for $model_type 🚀"
    
    local cmd="$PYTHON_CMD eeg_analysis/run_pipeline.py --config $config_file train --level $LEVEL --model-type $model_type"
    
    # Add dataset run ID if specified
    if [ -n "$DATASET_RUN_ID" ]; then
        cmd="$cmd --use-dataset-from-run $DATASET_RUN_ID"
        log_message "Using specified dataset from run: $DATASET_RUN_ID"
    else
        log_message "Using automatic dataset selection based on configuration"
    fi
    
    if [ "$use_fs" = "true" ]; then
        cmd="$cmd --enable-feature-selection --n-features-select $n_features --fs-method $fs_method"
    fi
    
    log_message "Starting experiment: $exp_name"
    log_message "MLflow experiment: $mlflow_experiment"
    log_message "Command: $cmd"
    
    if $cmd; then
        log_success "Completed: $exp_name"
        unset MLFLOW_EXPERIMENT_NAME  # Clean up environment variable
        return 0
    else
        log_error "Failed: $exp_name"
        unset MLFLOW_EXPERIMENT_NAME  # Clean up environment variable
        return 1
    fi
}

# Function to display progress
show_progress() {
    local current="$1"
    local total="$2"
    local percentage=$((current * 100 / total))
    printf "\r${BLUE}Progress: [%-50s] %d%% (%d/%d)${NC}" \
           "$(printf '#%.0s' $(seq 1 $((percentage / 2))))" \
           "$percentage" "$current" "$total"
}

# Check for help
if [[ "$SHOW_HELP" == "true" ]]; then
    echo "EEG Analysis Complete Model Training Script"
    echo ""
    echo "This script runs all available models with different configurations:"
    echo "1. All models without feature selection"
    echo "2. All models with select_k_best_f_classif (10, 15, 20 features)"
    echo "3. All models with select_k_best_mutual_info (10, 15, 20 features)"
    echo ""
    echo "Models: xgboost_gpu, catboost_gpu, lightgbm_gpu, pytorch_mlp, keras_mlp, random_forest, svm_rbf"
    echo "Configuration: Mix of GPU-ACCELERATED and traditional models with ULTRA-EXTREME optimization"
    echo "Total experiments: 49"
    echo ""
    echo "MLflow Experiment Organization:"
    echo "- eeg_deep_learning_gpu_baseline: Deep learning GPU models without feature selection"
    echo "- eeg_deep_learning_gpu_feature_selection: Deep learning GPU models with feature selection"
    echo "- eeg_boosting_gpu_baseline: Gradient boosting GPU models without feature selection"
    echo "- eeg_boosting_gpu_feature_selection: Gradient boosting GPU models with feature selection"
    echo "- eeg_traditional_models_baseline: Traditional models (RF, SVM) without feature selection"
    echo "- eeg_traditional_models_feature_selection: Traditional models with feature selection"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset-run-id <run_id>   Use specific MLflow dataset from this run ID"
    echo "  --dry-run                   Show what would be executed without running"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Dataset Selection:"
    echo "  • By default, the script automatically finds datasets matching your current"
    echo "    processing configuration (window size + channels from processing_config.yaml)"
    echo "  • Use --dataset-run-id to force a specific dataset"
    echo "  • Use 'python eeg_analysis/run_pipeline.py --config $CONFIG_FILE list-datasets'"
    echo "    to see available datasets that match your configuration"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Auto-select matching dataset"
    echo "  $0 --dataset-run-id abc123def456      # Use specific dataset"
    echo "  $0 --dry-run                          # Show commands without running"
    exit 0
fi

# Dry run option
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN MODE - Commands that would be executed:"
    echo ""
    
    if [ -n "$DATASET_RUN_ID" ]; then
        echo "Using specified dataset from run: $DATASET_RUN_ID"
    else
        echo "Using automatic dataset selection based on configuration"
    fi
    echo ""
    
    echo "=== Without feature selection ==="
    for model in "${MODELS[@]}"; do
        # Determine experiment name - all models use the same unified config
        if [[ "$model" == "pytorch_mlp" || "$model" == "keras_mlp" ]]; then
            exp_name="eeg_deep_learning_gpu_baseline"
        elif [[ "$model" == "xgboost_gpu" || "$model" == "catboost_gpu" || "$model" == "lightgbm_gpu" ]]; then
            exp_name="eeg_boosting_gpu_baseline"
        elif [[ "$model" == "random_forest" || "$model" == "svm_rbf" ]]; then
            exp_name="eeg_traditional_models_baseline"
        else
            exp_name="eeg_other_models_baseline"
        fi
        
        cmd="MLFLOW_EXPERIMENT_NAME=$exp_name $PYTHON_CMD eeg_analysis/run_pipeline.py --config $CONFIG_FILE train --level $LEVEL --model-type $model"
        if [ -n "$DATASET_RUN_ID" ]; then
            cmd="$cmd --use-dataset-from-run $DATASET_RUN_ID"
        fi
        echo "$cmd"
    done
    
    echo ""
    echo "=== With feature selection ==="
    for model in "${MODELS[@]}"; do
        for fs_method in "${FEATURE_SELECTION_METHODS[@]}"; do
            for n_features in "${FEATURE_COUNTS[@]}"; do
                # Determine experiment name - all models use the same unified config
                if [[ "$model" == "pytorch_mlp" || "$model" == "keras_mlp" ]]; then
                    exp_name="eeg_deep_learning_gpu_feature_selection"
                elif [[ "$model" == "xgboost_gpu" || "$model" == "catboost_gpu" || "$model" == "lightgbm_gpu" ]]; then
                    exp_name="eeg_boosting_gpu_feature_selection"
                elif [[ "$model" == "random_forest" || "$model" == "svm_rbf" ]]; then
                    exp_name="eeg_traditional_models_feature_selection"
                else
                    exp_name="eeg_other_models_feature_selection"
                fi
                
                cmd="MLFLOW_EXPERIMENT_NAME=$exp_name $PYTHON_CMD eeg_analysis/run_pipeline.py --config $CONFIG_FILE train --level $LEVEL --model-type $model --enable-feature-selection --n-features-select $n_features --fs-method $fs_method"
                if [ -n "$DATASET_RUN_ID" ]; then
                    cmd="$cmd --use-dataset-from-run $DATASET_RUN_ID"
                fi
                echo "$cmd"
            done
        done
    done
    
    echo ""
    echo "Total: $(( ${#MODELS[@]} + ${#MODELS[@]} * ${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]} )) experiments"
    echo ""
    echo "Experiments will be organized into 6 MLflow experiments:"
    echo "- eeg_deep_learning_gpu_baseline (2 runs: pytorch_mlp, keras_mlp)"
    echo "- eeg_deep_learning_gpu_feature_selection (12 runs: 2 models × 2 methods × 3 features)"
    echo "- eeg_boosting_gpu_baseline (3 runs: xgboost_gpu, catboost_gpu, lightgbm_gpu)"
    echo "- eeg_boosting_gpu_feature_selection (18 runs: 3 models × 2 methods × 3 features)"
    echo "- eeg_traditional_models_baseline (2 runs: random_forest, svm_rbf)"
    echo "- eeg_traditional_models_feature_selection (12 runs: 2 models × 2 methods × 3 features)"
    exit 0
fi

# Main execution
main() {
    log_message "Starting EEG Analysis Complete Model Training"
    log_message "Using ULTRA-EXTREME unified configuration: $CONFIG_FILE"
    log_message "Log file: $LOG_FILE"
    
    # Log dataset selection method
    if [ -n "$DATASET_RUN_ID" ]; then
        log_message "Dataset selection: Using specified run ID: $DATASET_RUN_ID"
    else
        log_message "Dataset selection: Automatic based on processing configuration"
        log_message "  (To use a specific dataset, run with --dataset-run-id <run_id>)"
        log_message "  (To list available datasets, run: python eeg_analysis/run_pipeline.py --config $CONFIG_FILE list-datasets)"
    fi
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Calculate total experiments
    local total_experiments=$(( ${#MODELS[@]} + ${#MODELS[@]} * ${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]} ))
    log_message "Total experiments to run: $total_experiments"
    
    local current_experiment=0
    local successful_experiments=0
    local failed_experiments=0
    
    echo ""
    log_message "=== PHASE 1: Models without feature selection ==="
    
    # Run models without feature selection
    for model in "${MODELS[@]}"; do
        current_experiment=$((current_experiment + 1))
        show_progress $current_experiment $total_experiments
        echo ""
        
        if run_experiment "$model" "false" "" ""; then
            successful_experiments=$((successful_experiments + 1))
        else
            failed_experiments=$((failed_experiments + 1))
        fi
        
        sleep 2  # Brief pause between experiments
    done
    
    echo ""
    log_message "=== PHASE 2: Models with feature selection ==="
    
    # Run models with feature selection
    for model in "${MODELS[@]}"; do
        for fs_method in "${FEATURE_SELECTION_METHODS[@]}"; do
            for n_features in "${FEATURE_COUNTS[@]}"; do
                current_experiment=$((current_experiment + 1))
                show_progress $current_experiment $total_experiments
                echo ""
                
                if run_experiment "$model" "true" "$fs_method" "$n_features"; then
                    successful_experiments=$((successful_experiments + 1))
                else
                    failed_experiments=$((failed_experiments + 1))
                fi
                
                sleep 2  # Brief pause between experiments
            done
        done
    done
    
    echo ""
    echo ""
    log_message "=== EXPERIMENT SUMMARY ==="
    log_message "Total experiments: $total_experiments"
    log_success "Successful: $successful_experiments"
    if [ $failed_experiments -gt 0 ]; then
        log_error "Failed: $failed_experiments"
    else
        log_success "Failed: $failed_experiments"
    fi
    
    # Generate summary report
    echo ""
    log_message "=== DETAILED EXPERIMENT LIST ==="
    
    # Models without feature selection
    log_message "Without feature selection:"
    for model in "${MODELS[@]}"; do
        echo "  - $model" | tee -a "$LOG_FILE"
    done
    
    # Models with feature selection
    log_message "With feature selection:"
    for model in "${MODELS[@]}"; do
        for fs_method in "${FEATURE_SELECTION_METHODS[@]}"; do
            for n_features in "${FEATURE_COUNTS[@]}"; do
                echo "  - $model + $fs_method ($n_features features)" | tee -a "$LOG_FILE"
            done
        done
    done
    
    echo ""
    log_message "All experiments completed!"
    log_message "View results with: mlflow ui"
    log_message "Full log saved to: $LOG_FILE"
    
    if [ $failed_experiments -gt 0 ]; then
        log_warning "Some experiments failed. Check the log file for details."
        exit 1
    else
        log_success "All experiments completed successfully!"
    fi
}

# Trap to handle interruption
trap 'echo -e "\n${RED}Experiment interrupted by user${NC}"; exit 130' INT

# Run main function
main 