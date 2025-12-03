#!/bin/bash

# Script to rerun specific experiments with feature selection
# All experiments use: 10s windows, configurable features and channels

set -e

# Parse command line arguments
CHANNELS=""
WINDOW_SIZE=10
N_FEATURES=5
FS_METHOD="select_k_best_f_classif"
SHOW_HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --channels)
      CHANNELS="$2"
      shift 2
      ;;
    --window-size)
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --n-features)
      N_FEATURES="$2"
      shift 2
      ;;
    --fs-method)
      FS_METHOD="$2"
      shift 2
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

# Show help if requested
if [[ "$SHOW_HELP" == "true" ]]; then
  echo "Rerun Experiments Script"
  echo ""
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --channels CHANNELS     Space-separated list of channels (e.g., 'af7 af8 tp9 tp10')"
  echo "                          Default: Uses all 4 channels from config"
  echo "  --window-size SIZE      Window size in seconds (default: 10)"
  echo "  --n-features N          Number of features to select (default: 3)"
  echo "  --fs-method METHOD      Feature selection method (default: select_k_best_f_classif)"
  echo "  -h, --help              Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 --channels 'af7 af8'              # Run with only frontal channels"
  echo "  $0 --channels 'tp9 tp10'              # Run with only temporal channels"
  echo "  $0 --channels 'af7 af8 tp9 tp10'      # Run with all 4 channels"
  echo "  $0 --n-features 5                     # Select 5 features instead of 3"
  echo ""
  exit 0
fi

CONFIG="eeg_analysis/configs/window_model_config_ultra_extreme.yaml"
LEVEL="window"

# Array of model types to run
MODELS=(
  "advanced_hybrid_1dcnn_lstm"
  "svm_linear"
  "efficient_tabular_mlp"
  "gradient_boosting"
  "svm_rbf"
  "advanced_1dcnn"
)

# Function to find 4-channel dataset by window size
find_4channel_dataset() {
  local window_seconds="$1"
  
  echo "🔍 Searching for 4-channel dataset with window size: ${window_seconds}s" >&2
  
  # Use the standalone Python script to search for datasets
  local dataset_info=$(python3 find_dataset.py "$window_seconds" 2>/dev/null)
  
  local status=$(echo "$dataset_info" | cut -d: -f1)
  
  if [[ "$status" == "SUCCESS" ]]; then
    local run_id=$(echo "$dataset_info" | cut -d: -f2)
    local dataset_name=$(echo "$dataset_info" | cut -d: -f3)
    
    if [[ -z "$run_id" || "$run_id" == "" ]]; then
      echo "ERROR:Empty run ID returned" >&2
      return 1
    fi
    
    echo "✅ Found 4-channel dataset: $dataset_name" >&2
    echo "$run_id"
    return 0
  else
    local error_msg=$(echo "$dataset_info" | cut -d: -f2-)
    echo "ERROR:Failed to find 4-channel dataset for ${window_seconds}s: $error_msg" >&2
    return 1
  fi
}

# Function to filter dataset columns based on selected channels
filter_dataset_columns() {
  local run_id="$1"
  local selected_channels="$2"
  local window_seconds="$3"
  
  echo "🔧 Filtering dataset columns for channels: $selected_channels" >&2
  
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
      echo "All 4 channels selected, no filtering needed" >&2
      echo "$run_id"  # Return original dataset run ID
      return 0
    fi
  fi
  
  # Need to filter columns - create new filtered dataset
  echo "Creating filtered dataset for channels: $selected_channels" >&2
  
  # Create filtered dataset using the standalone Python script
  local filter_result=$(python3 filter_dataset.py "$run_id" "$selected_channels" "$window_seconds" 2>/tmp/filter_debug.log)
  
  local status=$(echo "$filter_result" | cut -d: -f1)
  
  if [[ "$status" == "SUCCESS" ]]; then
    local new_run_id=$(echo "$filter_result" | cut -d: -f2)
    local new_dataset_name=$(echo "$filter_result" | cut -d: -f3-)
    echo "✅ Created filtered dataset: $new_dataset_name" >&2
    echo "$new_run_id"
    return 0
  else
    local error_msg=$(echo "$filter_result" | cut -d: -f2-)
    echo "ERROR:Failed to create filtered dataset: $error_msg" >&2
    if [[ -f /tmp/filter_debug.log ]]; then
      echo "Debug output:" >&2
      cat /tmp/filter_debug.log >&2
    fi
    return 1
  fi
}

# Determine dataset to use
DATASET_RUN_ID=""

if [[ -n "$CHANNELS" ]]; then
  echo "📡 Using specified channels: $CHANNELS"
  
  # Find base 4-channel dataset
  base_dataset_run_id=$(find_4channel_dataset "$WINDOW_SIZE")
  if [[ $? -ne 0 || -z "$base_dataset_run_id" ]]; then
    echo "❌ Failed to find base 4-channel dataset for ${WINDOW_SIZE}s windows"
    echo "   Please run preprocessing first: ./run_all_processing.sh"
    exit 1
  fi
  
  # Filter dataset for selected channels
  DATASET_RUN_ID=$(filter_dataset_columns "$base_dataset_run_id" "$CHANNELS" "$WINDOW_SIZE")
  if [[ $? -ne 0 ]]; then
    echo "❌ Failed to filter dataset for channels: $CHANNELS"
    exit 1
  fi
  
  echo "✅ Using filtered dataset run ID: $DATASET_RUN_ID"
else
  echo "📡 Using channels from processing config (auto-discovery)"
fi

echo ""
echo "🚀 Rerunning experiments with:"
echo "  - Window size: ${WINDOW_SIZE}s"
echo "  - Channels: ${CHANNELS:-"auto (from config)"}"
echo "  - Feature selection: $N_FEATURES features using $FS_METHOD"
echo "  - Models: ${#MODELS[@]} models"
echo ""

# Build base command
BASE_CMD="uv run python3 eeg_analysis/run_pipeline.py \
  --config \"$CONFIG\" \
  train \
  --level \"$LEVEL\" \
  --enable-feature-selection \
  --n-features-select \"$N_FEATURES\" \
  --fs-method \"$FS_METHOD\""

# Add dataset run ID if we have one
if [[ -n "$DATASET_RUN_ID" ]]; then
  BASE_CMD="$BASE_CMD --use-dataset-from-run \"$DATASET_RUN_ID\""
fi

for model in "${MODELS[@]}"; do
  echo "=========================================="
  echo "Running: $model"
  echo "=========================================="
  
  eval "$BASE_CMD --model-type \"$model\""
  
  echo ""
  echo "✅ Completed: $model"
  echo ""
  sleep 2  # Brief pause between experiments
done

echo "🎉 All experiments completed!"


