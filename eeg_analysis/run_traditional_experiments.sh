#!/bin/bash

# EEG Analysis - Traditional ML Models Training Script
# This script runs traditional ML models (excluding deep learning) with different feature selection configurations

set -e  # Exit on any error

# Activate conda environment
echo "Activating conda base environment..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate base

# Configuration
CONFIG_FILE="eeg_analysis/configs/window_model_config.yaml"
LEVEL="window"
PYTHON_CMD="python"

# Available traditional ML models (excluding deep learning)
MODELS=("random_forest" "gradient_boosting" "logistic_regression" "svm")

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
LOG_FILE="traditional_experiment_log_$(date +%Y%m%d_%H%M%S).txt"

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
    if [[ "$model_type" == "random_forest" || "$model_type" == "gradient_boosting" || "$model_type" == "extra_trees" ]]; then
        mlflow_experiment="eeg_tree_models${experiment_suffix}"
    elif [[ "$model_type" == "logistic_regression" || "$model_type" == "svm" || "$model_type" == "sgd" ]]; then
        mlflow_experiment="eeg_linear_models${experiment_suffix}"
    else
        mlflow_experiment="eeg_other_models${experiment_suffix}"
    fi
    
    # Export experiment name as environment variable for run_pipeline.py to use
    export MLFLOW_EXPERIMENT_NAME="$mlflow_experiment"
    
    local cmd="$PYTHON_CMD eeg_analysis/run_pipeline.py --config $CONFIG_FILE train --level $LEVEL --model-type $model_type"
    
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

# Main execution
main() {
    log_message "Starting EEG Analysis Traditional ML Model Training"
    log_message "Configuration file: $CONFIG_FILE"
    log_message "Log file: $LOG_FILE"
    log_message "Models: ${MODELS[*]}"
    
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

# Check for help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "EEG Analysis Traditional ML Model Training Script"
    echo ""
    echo "This script runs traditional ML models (excluding deep learning) with different configurations:"
    echo "1. All models without feature selection"
    echo "2. All models with select_k_best_f_classif (10, 15, 20 features)"
    echo "3. All models with select_k_best_mutual_info (10, 15, 20 features)"
    echo ""
    echo "Models: random_forest, gradient_boosting, logistic_regression, svm"
    echo "Total experiments: 28"
    echo ""
    echo "MLflow Experiment Organization:"
    echo "- eeg_tree_models_baseline: Tree-based models without feature selection"
    echo "- eeg_tree_models_feature_selection: Tree-based models with feature selection"
    echo "- eeg_linear_models_baseline: Linear models without feature selection"
    echo "- eeg_linear_models_feature_selection: Linear models with feature selection"
    echo ""
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show what would be executed without running"
    echo "  -h, --help   Show this help message"
    exit 0
fi

# Dry run option
if [[ "$1" == "--dry-run" ]]; then
    echo "DRY RUN MODE - Commands that would be executed:"
    echo ""
    
    echo "=== Without feature selection ==="
    for model in "${MODELS[@]}"; do
        # Determine experiment name for this model
        if [[ "$model" == "random_forest" || "$model" == "gradient_boosting" || "$model" == "extra_trees" ]]; then
            exp_name="eeg_tree_models_baseline"
        elif [[ "$model" == "logistic_regression" || "$model" == "svm" || "$model" == "sgd" ]]; then
            exp_name="eeg_linear_models_baseline"
        else
            exp_name="eeg_other_models_baseline"
        fi
        echo "MLFLOW_EXPERIMENT_NAME=$exp_name $PYTHON_CMD eeg_analysis/run_pipeline.py --config $CONFIG_FILE train --level $LEVEL --model-type $model"
    done
    
    echo ""
    echo "=== With feature selection ==="
    for model in "${MODELS[@]}"; do
        for fs_method in "${FEATURE_SELECTION_METHODS[@]}"; do
            for n_features in "${FEATURE_COUNTS[@]}"; do
                # Determine experiment name for this model
                if [[ "$model" == "random_forest" || "$model" == "gradient_boosting" || "$model" == "extra_trees" ]]; then
                    exp_name="eeg_tree_models_feature_selection"
                elif [[ "$model" == "logistic_regression" || "$model" == "svm" || "$model" == "sgd" ]]; then
                    exp_name="eeg_linear_models_feature_selection"
                else
                    exp_name="eeg_other_models_feature_selection"
                fi
                echo "MLFLOW_EXPERIMENT_NAME=$exp_name $PYTHON_CMD eeg_analysis/run_pipeline.py --config $CONFIG_FILE train --level $LEVEL --model-type $model --enable-feature-selection --n-features-select $n_features --fs-method $fs_method"
            done
        done
    done
    
    echo ""
    echo "Total: $(( ${#MODELS[@]} + ${#MODELS[@]} * ${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]} )) experiments"
    echo ""
    echo "Experiments will be organized into 4 MLflow experiments:"
    echo "- eeg_tree_models_baseline (2 runs: random_forest, gradient_boosting)"
    echo "- eeg_tree_models_feature_selection (12 runs: 2 models × 2 methods × 3 features)"
    echo "- eeg_linear_models_baseline (2 runs: logistic_regression, svm)"
    echo "- eeg_linear_models_feature_selection (12 runs: 2 models × 2 methods × 3 features)"
    exit 0
fi

# Run main function
main 