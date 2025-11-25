#!/bin/bash

# EEG Analysis - Complete Model Training Script
# This script runs all available models with different feature selection configurations

set -e  # Exit on any error

# Parse command line arguments
DATASET_RUN_ID=""
ORDERING_METHOD=""
SELECTED_MODEL=""
SHOW_HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-run-id)
            DATASET_RUN_ID="$2"
            shift 2
            ;;
        --ordering)
            ORDERING_METHOD="$2"
            # Validate ordering method
            if [[ "$ORDERING_METHOD" != "sequential" && "$ORDERING_METHOD" != "completion" ]]; then
                echo "Error: --ordering must be 'sequential' or 'completion'"
                exit 1
            fi
            shift 2
            ;;
        --model)
            SELECTED_MODEL="$2"
            # Validate model type
            VALID_MODELS=("xgboost_gpu" "catboost_gpu" "lightgbm_gpu" "pytorch_mlp" "keras_mlp" "random_forest" "svm_rbf")
            if [[ ! " ${VALID_MODELS[@]} " =~ " ${SELECTED_MODEL} " ]]; then
                echo "Error: --model must be one of: ${VALID_MODELS[*]}"
                exit 1
            fi
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
ALL_MODELS=("${GPU_ML_MODELS[@]}" "${GPU_DL_MODELS[@]}" "${TRADITIONAL_MODELS[@]}")

# Set models to run based on selection
if [[ -n "$SELECTED_MODEL" ]]; then
    MODELS=("$SELECTED_MODEL")
else
    MODELS=("${ALL_MODELS[@]}")
fi

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

# Function to find 4-channel dataset by window size only
find_4channel_dataset_by_window_size() {
    local window_seconds="$1"
    local ordering_method="$2"  # Add ordering method parameter
    
    log_message "🔍 Searching for 4-channel dataset with window size: ${window_seconds}s" >&2
    if [[ -n "$ordering_method" ]]; then
        log_message "🎯 Filtering for ordering method: $ordering_method" >&2
    fi
    
    # Use the standalone Python script to search for datasets
    local dataset_info
    if [[ -n "$ordering_method" ]]; then
        dataset_info=$(python3 find_dataset.py "$window_seconds" "$ordering_method" 2>/dev/null)
    else
        dataset_info=$(python3 find_dataset.py "$window_seconds" 2>/dev/null)
    fi
    
    local status=$(echo "$dataset_info" | cut -d: -f1)
    
    if [[ "$status" == "SUCCESS" ]]; then
        local run_id=$(echo "$dataset_info" | cut -d: -f2)
        local dataset_name=$(echo "$dataset_info" | cut -d: -f3)
        
        # Validate that we actually got a run ID
        if [[ -z "$run_id" || "$run_id" == "" ]]; then
            log_error "Failed to find 4-channel dataset for ${window_seconds}s: Empty run ID returned" >&2
            return 1
        fi
        
        log_success "Found 4-channel dataset: $dataset_name" >&2
        log_message "Dataset run ID: $run_id" >&2
        echo "$run_id"
        return 0
    else
        local error_msg=$(echo "$dataset_info" | cut -d: -f2-)
        if [[ -n "$ordering_method" ]]; then
            log_error "Failed to find 4-channel $ordering_method dataset for ${window_seconds}s: $error_msg" >&2
        else
            log_error "Failed to find 4-channel dataset for ${window_seconds}s: $error_msg" >&2
        fi
        return 1
    fi
}

# Function to get selected channels from processing config
get_selected_channels() {
    local config_file="$1"
    
    # Extract channels from processing config
    local channels=$(python3 -c "
import yaml
import sys

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    channels = config['data_loader']['channels']
    print(' '.join(channels))
except Exception as e:
    print('ERROR', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

    if [[ $? -eq 0 ]]; then
        echo "$channels"
        return 0
    else
        log_error "Failed to extract channels from config: $config_file"
        return 1
    fi
}

# Function to filter dataset columns based on selected channels
filter_dataset_columns() {
    local run_id="$1"
    local selected_channels="$2"
    local window_seconds="$3"
    
    log_message "🔧 Filtering dataset columns for channels: $selected_channels" >&2
    
    # Convert space-separated channels to array
    local channels_array=($selected_channels)
    
    # Check if we need all 4 channels (no filtering needed)
    if [[ ${#channels_array[@]} -eq 4 ]]; then
        local has_all_channels=true
        for required_ch in af7 af8 tp9 tp10; do
            local found=false
            for ch in "${channels_array[@]}"; do
                if [[ "$ch" == "$required_ch" ]]; then
                    found=true
                    break
                fi
            done
            if [[ "$found" == "false" ]]; then
                has_all_channels=false
                break
            fi
        done
        
        if [[ "$has_all_channels" == "true" ]]; then
            log_message "All 4 channels selected, no filtering needed" >&2
            echo "$run_id"  # Return original dataset run ID
            return 0
        fi
    fi
    
    # Need to filter columns - create new filtered dataset
    
    local channels_str=$(echo "$selected_channels" | tr ' ' '-')
    local new_dataset_name="EEG_${window_seconds}s_${channels_str}_filtered"
    local new_dataset_path="eeg_analysis/data/processed/features/${new_dataset_name}.parquet"
    
    log_message "Creating filtered dataset: $new_dataset_name" >&2
    log_message "Output path: $new_dataset_path" >&2
    
    # Create filtered dataset using the standalone Python script
    local filter_result=$(python3 filter_dataset.py "$run_id" "$selected_channels" "$window_seconds" 2>/tmp/filter_debug.log)
    
    local status=$(echo "$filter_result" | cut -d: -f1)
    
    if [[ "$status" == "SUCCESS" ]]; then
        local new_run_id=$(echo "$filter_result" | cut -d: -f2)
        local new_dataset_name=$(echo "$filter_result" | cut -d: -f3-)
        log_success "Created filtered dataset: $new_dataset_name" >&2
        log_message "New dataset run ID: $new_run_id" >&2
        echo "$new_run_id"
        return 0
    else
        local error_msg=$(echo "$filter_result" | cut -d: -f2-)
        log_error "Failed to create filtered dataset: $error_msg" >&2
        if [[ -f /tmp/filter_debug.log ]]; then
            log_error "Debug output:" >&2
            cat /tmp/filter_debug.log >&2
        fi
        return 1
    fi
}

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
    local dataset_run_id="$5"
    
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
    
    # Add dataset run ID (prioritize function parameter, then global variable)
    local final_dataset_id="$dataset_run_id"
    if [ -z "$final_dataset_id" ]; then
        final_dataset_id="$DATASET_RUN_ID"
    fi
    
    if [ -n "$final_dataset_id" ]; then
        cmd="$cmd --use-dataset-from-run $final_dataset_id"
        log_message "Using dataset from run: $final_dataset_id"
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
    echo "Total experiments: 49 per window size × number of window sizes (all models)"
    echo "                   7 per window size × number of window sizes (single model)"
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
    echo "  --ordering <sequential|completion>  Select dataset ordering (sequential or completion)"
    echo "  --model <model_name>        Run experiments for specific model only"
    echo "                              Available: xgboost_gpu, catboost_gpu, lightgbm_gpu,"
    echo "                                        pytorch_mlp, keras_mlp, random_forest, svm_rbf"
    echo "  --dry-run                   Show what would be executed without running"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Dataset Selection:"
    echo "  • NEW: The script now runs experiments for ALL window sizes from processing_config.yaml"
    echo "  • It searches for 4-channel datasets by WINDOW SIZE and ORDERING METHOD"
    echo "  • Use --ordering to specify 'sequential' or 'completion' dataset types"
    echo "  • Each window size gets its own set of experiments with appropriate datasets"
    echo "  • Use --dataset-run-id to force a specific dataset for ALL window sizes (overrides --ordering)"
    echo "  • Requirements: You must have datasets for each window size and ordering type"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all experiments (uses any available datasets)"
    echo "  $0 --ordering sequential              # Use only sequential datasets"
    echo "  $0 --ordering completion              # Use only completion datasets"
    echo "  $0 --model keras_mlp                  # Run only keras_mlp experiments"
    echo "  $0 --model xgboost_gpu --ordering sequential  # Run only xgboost_gpu with sequential datasets"
    echo "  $0 --dataset-run-id abc123def456      # Use specific dataset for ALL window sizes"
    echo "  $0 --dry-run --ordering sequential    # Show what would be executed with sequential datasets"
    echo ""
    echo "Prerequisites:"
    echo "  • Create datasets for all window sizes and ordering types by running:"
    echo "    ./run_all_processing.sh with ordering_method: 'sequential' in processing_config.yaml"
    echo "    ./run_all_processing.sh with ordering_method: 'completion' in processing_config.yaml"
    exit 0
fi

# Dry run option
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN MODE - Commands that would be executed:"
    echo ""
    
    # Extract window sizes and channels for dry run
    processing_config="eeg_analysis/configs/processing_config.yaml"
    window_sizes_str=$(python3 -c "
import yaml
try:
    with open('$processing_config', 'r') as f:
        config = yaml.safe_load(f)
    window_config = config['window_slicer']['window_seconds']
    if isinstance(window_config, list):
        print(' '.join(map(str, window_config)))
    else:
        print(str(window_config))
except Exception:
    print('UNKNOWN')
" 2>/dev/null)
    
    selected_channels=$(python3 -c "
import yaml
try:
    with open('$processing_config', 'r') as f:
        config = yaml.safe_load(f)
    print(' '.join(config['data_loader']['channels']))
except Exception:
    print('UNKNOWN')
" 2>/dev/null)
    
    # Convert to array for counting
    read -a window_sizes_dry <<< "$window_sizes_str"
    
    echo "Configuration:"
    echo "  Window sizes: ${window_sizes_str}s"
    echo "  Selected channels: $selected_channels"
    echo "  Total window sizes: ${#window_sizes_dry[@]}"
    if [ -n "$ORDERING_METHOD" ]; then
        echo "  Ordering method: $ORDERING_METHOD"
    fi
    if [ -n "$SELECTED_MODEL" ]; then
        echo "  Selected model: $SELECTED_MODEL"
        echo "  Running single model experiments only"
    else
        echo "  Models: All available (${ALL_MODELS[*]})"
    fi
    echo ""
    
    if [ -n "$DATASET_RUN_ID" ]; then
        echo "Dataset: Using specified run ID: $DATASET_RUN_ID for ALL window sizes"
    else
        echo "Dataset Strategy:"
        for ws in $window_sizes_str; do
            if [ -n "$ORDERING_METHOD" ]; then
                echo "  - ${ws}s: Search for 4-channel $ORDERING_METHOD dataset, filter for channels: $selected_channels"
            else
                echo "  - ${ws}s: Search for 4-channel dataset (any ordering), filter for channels: $selected_channels"
            fi
        done
    fi
    echo ""
    
    experiments_per_window=$(( ${#MODELS[@]} + ${#MODELS[@]} * ${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]} ))
    total_experiments_dry=$(( experiments_per_window * ${#window_sizes_dry[@]} ))
    
    echo "=== Experiments for each window size ==="
    for ws in $window_sizes_str; do
        echo ""
        echo "--- Window Size: ${ws}s ---"
        echo "Without feature selection:"
        for model in "${MODELS[@]}"; do
            # Determine experiment name
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
            else
                cmd="$cmd --use-dataset-from-run <filtered_dataset_for_${ws}s>"
            fi
            echo "  $cmd"
        done
        
        echo "With feature selection (showing first method only for brevity):"
        for model in "${MODELS[@]}"; do
            # Show only first feature selection method and count for brevity
            fs_method="${FEATURE_SELECTION_METHODS[0]}"
            n_features="${FEATURE_COUNTS[0]}"
            
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
            else
                cmd="$cmd --use-dataset-from-run <filtered_dataset_for_${ws}s>"
            fi
            echo "  $cmd"
            echo "    ... (plus ${#FEATURE_SELECTION_METHODS[@]} methods × ${#FEATURE_COUNTS[@]} features = $((${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]} - 1)) more variations)"
            break  # Only show first model for brevity
        done
        echo "  ... (plus $((${#MODELS[@]} - 1)) more models)"
    done
    
    echo ""
    echo "Total: $experiments_per_window experiments per window size × ${#window_sizes_dry[@]} window sizes = $total_experiments_dry experiments"
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
    if [ -n "$ORDERING_METHOD" ]; then
        log_message "Dataset ordering method: $ORDERING_METHOD"
    fi
    if [ -n "$SELECTED_MODEL" ]; then
        log_message "🎯 Running experiments for selected model only: $SELECTED_MODEL"
    else
        log_message "🚀 Running experiments for all models: ${ALL_MODELS[*]}"
    fi
    log_message "Log file: $LOG_FILE"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Get window sizes from processing config (support both single value and list)
    log_message "🔍 Extracting window sizes from processing configuration..."
    
    processing_config="eeg_analysis/configs/processing_config.yaml"
    if [ ! -f "$processing_config" ]; then
        log_error "Processing configuration file not found: $processing_config"
        exit 1
    fi
    
    window_sizes_str=$(python3 -c "
import yaml
try:
    with open('$processing_config', 'r') as f:
        config = yaml.safe_load(f)
    window_config = config['window_slicer']['window_seconds']
    
    # Handle both single value and list
    if isinstance(window_config, list):
        print(' '.join(map(str, window_config)))
    else:
        print(str(window_config))
except Exception as e:
    print('ERROR')
" 2>/dev/null)
    
    if [[ "$window_sizes_str" == "ERROR" ]]; then
        log_error "Failed to extract window sizes from processing config"
        exit 1
    fi
    
    # Convert to array
    read -a window_sizes_array <<< "$window_sizes_str"
    log_message "Window sizes from config: ${window_sizes_array[*]}s"
    
    # Get selected channels from processing config
    log_message "📡 Extracting selected channels from processing configuration..."
    selected_channels=$(get_selected_channels "$processing_config")
    if [[ $? -ne 0 ]]; then
        log_error "Failed to extract channels from processing config"
        exit 1
    fi
    
    log_message "Selected channels: $selected_channels"
    
    # Calculate total experiments across all window sizes
    experiments_per_window=$(( ${#MODELS[@]} + ${#MODELS[@]} * ${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]} ))
    total_experiments=$(( experiments_per_window * ${#window_sizes_array[@]} ))
    log_message "Experiments per window size: $experiments_per_window"
    log_message "Total experiments across all window sizes: $total_experiments"
    
    current_experiment=0
    successful_experiments=0
    failed_experiments=0
    
    # Run experiments for each window size
    for window_seconds in "${window_sizes_array[@]}"; do
        log_message ""
        log_message "🔥 ==================== WINDOW SIZE: ${window_seconds}s ===================="
        
        # Determine dataset to use for this window size
        final_dataset_run_id=""
        
        if [ -n "$DATASET_RUN_ID" ]; then
            log_message "📊 Using user-specified dataset run ID: $DATASET_RUN_ID"
            final_dataset_run_id="$DATASET_RUN_ID"
        else
            log_message "🔎 Searching for 4-channel dataset with window size ${window_seconds}s..."
            if [ -n "$ORDERING_METHOD" ]; then
                log_message "🎯 Looking specifically for $ORDERING_METHOD ordering method"
            fi
            
            # Find 4-channel dataset by window size
            base_dataset_run_id=$(find_4channel_dataset_by_window_size "$window_seconds" "$ORDERING_METHOD")
            find_result=$?
            
            if [[ $find_result -ne 0 || -z "$base_dataset_run_id" || "$base_dataset_run_id" == "" ]]; then
                log_error "❌ No 4-channel dataset found for window size ${window_seconds}s"
                log_error "  Skipping experiments for this window size..."
                log_error "  To create dataset: ./run_all_processing.sh (or process with ${window_seconds}s config)"
                continue  # Skip to next window size
            fi
            
            log_success "✅ Found 4-channel dataset for ${window_seconds}s"
            
            # Filter dataset columns based on selected channels
            log_message "🔧 Filtering dataset for selected channels..."
            final_dataset_run_id=$(filter_dataset_columns "$base_dataset_run_id" "$selected_channels" "$window_seconds")
            if [[ $? -ne 0 ]]; then
                log_error "❌ Failed to create filtered dataset for selected channels"
                log_error "  Skipping experiments for this window size..."
                continue  # Skip to next window size
            fi
            
            log_success "✅ Dataset prepared for training with selected channels"
        fi
        
        log_message "📋 Using dataset run ID: $final_dataset_run_id"
        log_message "🚀 Starting experiments for ${window_seconds}s windows with:"
        log_message "  - Channels: $selected_channels"
        log_message "  - Dataset run ID: $final_dataset_run_id"
        
        echo ""
        log_message "=== PHASE 1 (${window_seconds}s): Models without feature selection ==="
        
        # Run models without feature selection for this window size
        for model in "${MODELS[@]}"; do
            current_experiment=$((current_experiment + 1))
            show_progress $current_experiment $total_experiments
            echo ""
            
            if run_experiment "$model" "false" "" "" "$final_dataset_run_id"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
            fi
            
            sleep 2  # Brief pause between experiments
        done
        
        echo ""
        log_message "=== PHASE 2 (${window_seconds}s): Models with feature selection ==="
        
        # Run models with feature selection for this window size
        for model in "${MODELS[@]}"; do
            for fs_method in "${FEATURE_SELECTION_METHODS[@]}"; do
                for n_features in "${FEATURE_COUNTS[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    show_progress $current_experiment $total_experiments
                    echo ""
                    
                    if run_experiment "$model" "true" "$fs_method" "$n_features" "$final_dataset_run_id"; then
                        successful_experiments=$((successful_experiments + 1))
                    else
                        failed_experiments=$((failed_experiments + 1))
                    fi
                    
                    sleep 2  # Brief pause between experiments
                done
            done
        done
        
        log_message "✅ Completed all experiments for window size ${window_seconds}s"
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
    
    # Summary by window size
    log_message "Experiments by window size:"
    for window_seconds in "${window_sizes_array[@]}"; do
        echo "  Window size ${window_seconds}s:" | tee -a "$LOG_FILE"
        echo "    - ${#MODELS[@]} models without feature selection" | tee -a "$LOG_FILE"
        echo "    - ${#MODELS[@]} models × ${#FEATURE_SELECTION_METHODS[@]} methods × ${#FEATURE_COUNTS[@]} features = $((${#MODELS[@]} * ${#FEATURE_SELECTION_METHODS[@]} * ${#FEATURE_COUNTS[@]})) with feature selection" | tee -a "$LOG_FILE"
        echo "    - Total: $experiments_per_window experiments" | tee -a "$LOG_FILE"
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