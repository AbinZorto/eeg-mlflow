#!/bin/bash

# EEG Analysis - Complete Data Processing Script
# This script runs data processing for all window sizes to create datasets

set -e  # Exit on any error

# Parse command line arguments
SHOW_HELP=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Configuration
PROCESSING_CONFIG="eeg_analysis/configs/processing_config.yaml"
PYTHON_CMD="python"

# Window sizes to process (matching your config)
WINDOW_SIZES=(2 4 6 8 10)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="processing_log_$(date +%Y%m%d_%H%M%S).txt"

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

# Function to create temporary config for specific window size
create_temp_config() {
    local window_size="$1"
    local temp_config="eeg_analysis/configs/temp_processing_config_${window_size}s.yaml"
    
    # Copy original config and modify window_seconds to single value
    python3 -c "
import yaml
import sys

try:
    with open('$PROCESSING_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set window_seconds to single value instead of list
    config['window_slicer']['window_seconds'] = $window_size
    
    # Update MLflow experiment and run names to include window size
    config['mlflow']['experiment_name'] = 'eeg_processing'
    config['mlflow']['run_name'] = f'processing_${window_size}s_window'
    
    with open('$temp_config', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        echo "$temp_config"
        return 0
    else
        log_error "Failed to create temporary config for ${window_size}s"
        return 1
    fi
}

# Function to run processing for a single window size
run_processing() {
    local window_size="$1"
    
    log_message "Creating temporary config for ${window_size}s windows..."
    local temp_config=$(create_temp_config "$window_size")
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    log_message "Starting processing for ${window_size}s windows..."
    log_message "Using config: $temp_config"
    
    # Set MLflow experiment name for this processing run
    export MLFLOW_EXPERIMENT_NAME="eeg_processing"
    
    local cmd="$PYTHON_CMD eeg_analysis/run_pipeline.py --config $temp_config process"
    
    log_message "Command: $cmd"
    
    if $cmd; then
        log_success "Completed processing for ${window_size}s windows"
        
        # Clean up temporary config
        rm -f "$temp_config"
        log_message "Cleaned up temporary config: $temp_config"
        
        unset MLFLOW_EXPERIMENT_NAME
        return 0
    else
        log_error "Failed processing for ${window_size}s windows"
        
        # Clean up temporary config even on failure
        rm -f "$temp_config"
        log_message "Cleaned up temporary config: $temp_config"
        
        unset MLFLOW_EXPERIMENT_NAME
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
    echo "EEG Analysis Complete Data Processing Script"
    echo ""
    echo "This script runs data processing for all window sizes to create datasets:"
    echo "- Window sizes: 2s, 4s, 6s, 8s, 10s"
    echo "- Uses 4-channel configuration (af7, af8, tp9, tp10)"
    echo "- Creates separate MLflow runs for each window size"
    echo "- Generates datasets that can be filtered for different channel subsets"
    echo ""
    echo "Output:"
    echo "- 5 datasets with different window sizes"
    echo "- All stored in MLflow experiment: eeg_processing"
    echo "- Each dataset can be filtered for any channel subset later"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show what would be executed without running"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0           # Run processing for all window sizes"
    echo "  $0 --dry-run # Show commands without running"
    echo ""
    echo "Prerequisites:"
    echo "  • Raw EEG data file must be available"
    echo "  • Processing config should have all 4 channels: ['af7', 'af8', 'tp9', 'tp10']"
    echo "  • Sufficient disk space for interim and final datasets"
    echo ""
    echo "After completion:"
    echo "  • Use 'python eeg_analysis/run_pipeline.py --config <config> list-datasets' to view"
    echo "  • Use './run_all_experiments.sh' for training with automatic dataset selection"
    exit 0
fi

# Dry run option
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN MODE - Commands that would be executed:"
    echo ""
    
    # Check current channel configuration
    selected_channels=$(python3 -c "
import yaml
try:
    with open('$PROCESSING_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    print(' '.join(config['data_loader']['channels']))
except Exception:
    print('UNKNOWN')
" 2>/dev/null)
    
    echo "Configuration:"
    echo "  Processing config: $PROCESSING_CONFIG"
    echo "  Channels: $selected_channels"
    echo "  Window sizes: ${WINDOW_SIZES[*]}"
    echo "  MLflow experiment: eeg_processing"
    echo ""
    
    for window_size in "${WINDOW_SIZES[@]}"; do
        temp_config="eeg_analysis/configs/temp_processing_config_${window_size}s.yaml"
        echo "Processing ${window_size}s windows:"
        echo "  1. Create temporary config: $temp_config"
        echo "  2. MLFLOW_EXPERIMENT_NAME=eeg_processing $PYTHON_CMD eeg_analysis/run_pipeline.py --config $temp_config process"
        echo "  3. Clean up: rm $temp_config"
        echo ""
    done
    
    echo "Total: ${#WINDOW_SIZES[@]} processing runs"
    echo ""
    echo "Each run will create a dataset named like: EEG_<window_size>s_af7-af8-tp9-tp10_<windows_count>"
    echo "These 4-channel datasets can later be filtered for any channel subset."
    exit 0
fi

# Main execution
main() {
    log_message "Starting EEG Analysis Complete Data Processing"
    log_message "Log file: $LOG_FILE"
    
    # Check if config file exists
    if [ ! -f "$PROCESSING_CONFIG" ]; then
        log_error "Processing configuration file not found: $PROCESSING_CONFIG"
        exit 1
    fi
    
    # Check current channel configuration
    log_message "🔍 Checking current channel configuration..."
    selected_channels=$(python3 -c "
import yaml
try:
    with open('$PROCESSING_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    channels = config['data_loader']['channels']
    print(' '.join(channels))
    
    # Check if we have all 4 channels
    required_channels = {'af7', 'af8', 'tp9', 'tp10'}
    current_channels = set(channels)
    
    if current_channels == required_channels:
        print('ALL_CHANNELS_OK')
    else:
        print('MISSING_CHANNELS')
        
except Exception as e:
    print('ERROR')
" 2>/dev/null)
    
    # Extract just the channels list
    channels_list=$(echo "$selected_channels" | head -1)
    channel_status=$(echo "$selected_channels" | tail -1)
    
    log_message "Current channels: $channels_list"
    
    if [[ "$channel_status" != "ALL_CHANNELS_OK" ]]; then
        log_warning "⚠️  Not all 4 channels are configured!"
        log_warning "Current channels: $channels_list"
        log_warning "For maximum flexibility, consider using all 4 channels: ['af7', 'af8', 'tp9', 'tp10']"
        log_warning "This allows filtering for any channel subset later."
        log_message "Continuing with current configuration..."
    else
        log_success "✅ All 4 channels configured - datasets will support all channel combinations"
    fi
    
    log_message ""
    log_message "🚀 Starting processing for window sizes: ${WINDOW_SIZES[*]}"
    
    # Calculate total processing runs
    total_runs=${#WINDOW_SIZES[@]}
    log_message "Total processing runs: $total_runs"
    
    current_run=0
    successful_runs=0
    failed_runs=0
    
    echo ""
    
    # Process each window size
    for window_size in "${WINDOW_SIZES[@]}"; do
        current_run=$((current_run + 1))
        show_progress $current_run $total_runs
        echo ""
        
        if run_processing "$window_size"; then
            successful_runs=$((successful_runs + 1))
        else
            failed_runs=$((failed_runs + 1))
        fi
        
        echo ""
        sleep 2  # Brief pause between processing runs
    done
    
    echo ""
    echo ""
    log_message "=== PROCESSING SUMMARY ==="
    log_message "Total runs: $total_runs"
    log_success "Successful: $successful_runs"
    if [ $failed_runs -gt 0 ]; then
        log_error "Failed: $failed_runs"
    else
        log_success "Failed: $failed_runs"
    fi
    
    # Generate summary report
    echo ""
    log_message "=== DATASETS CREATED ==="
    
    for window_size in "${WINDOW_SIZES[@]}"; do
        echo "  - EEG_${window_size}s_${channels_list// /-}_<window_count>" | tee -a "$LOG_FILE"
    done
    
    echo ""
    log_message "All processing completed!"
    log_message "View datasets with: python eeg_analysis/run_pipeline.py --config $PROCESSING_CONFIG list-datasets"
    log_message "Start training with: ./run_all_experiments.sh"
    log_message "Full log saved to: $LOG_FILE"
    
    if [ $failed_runs -gt 0 ]; then
        log_warning "Some processing runs failed. Check the log file for details."
        exit 1
    else
        log_success "All processing runs completed successfully!"
    fi
}

# Trap to handle interruption
trap 'echo -e "\n${RED}Processing interrupted by user${NC}"; exit 130' INT

# Run main function
main 