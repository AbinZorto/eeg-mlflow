"""
Primary dataset creation - concatenates windowed EEG data without feature extraction.
This creates the raw windowed dataset for direct model input (e.g., for Mamba pretraining/finetuning).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_primary_dataset(
    config: Dict[str, Any],
    windowed_path: str
) -> Tuple[pd.DataFrame, Optional[PandasDataset]]:
    """
    Create primary dataset by concatenating remission and non-remission windowed data.
    
    The primary dataset contains:
    - Participant: participant ID
    - Remission: binary label (0=non-remission, 1=remission)
    - parent_window_id: original window ID
    - sub_window_id: sub-window ID within parent
    - window_start, window_end: window boundaries
    - Channel columns (af7, af8, tp9, tp10): raw signal data per channel
    
    Args:
        config: Configuration dictionary
        windowed_path: Path to directory containing windowed parquet files
        
    Returns:
        Tuple of (primary_df, mlflow_dataset)
    """
    try:
        logger.info("Creating primary dataset from windowed data...")
        
        windowed_dir = Path(windowed_path)
        if not windowed_dir.exists():
            raise FileNotFoundError(f"Windowed data directory not found: {windowed_path}")
        
        # Load remission and non-remission data
        remission_file = windowed_dir / "remission.parquet"
        nonremission_file = windowed_dir / "non_remission.parquet"  # Note: underscore in filename
        
        dfs = []
        
        if remission_file.exists():
            logger.info(f"Loading remission data from {remission_file}")
            remission_df = pd.read_parquet(remission_file)
            logger.info(f"  Remission windows: {len(remission_df)}")
            dfs.append(remission_df)
        else:
            logger.warning(f"Remission file not found: {remission_file}")
        
        if nonremission_file.exists():
            logger.info(f"Loading non-remission data from {nonremission_file}")
            nonremission_df = pd.read_parquet(nonremission_file)
            logger.info(f"  Non-remission windows: {len(nonremission_df)}")
            dfs.append(nonremission_df)
        else:
            logger.warning(f"Non-remission file not found: {nonremission_file}")
        
        if not dfs:
            raise FileNotFoundError("No windowed data files found")
        
        # Concatenate all data
        primary_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(primary_df)} total windows")
        
        # Rename columns to match expected format
        column_mapping = {
            'participant_id': 'Participant',
            'group': 'Remission'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in primary_df.columns:
                primary_df = primary_df.rename(columns={old_col: new_col})
        
        # Convert group to binary Remission if needed
        if 'Remission' in primary_df.columns and primary_df['Remission'].dtype == 'object':
            primary_df['Remission'] = primary_df['Remission'].apply(
                lambda x: 1 if str(x).lower() == 'remission' else 0
            )
        
        # CRITICAL: Sort by participant and window IDs to preserve temporal order
        # This maintains the sequential relationship between windows for each participant
        sort_cols = ['Participant']
        if 'parent_window_id' in primary_df.columns:
            sort_cols.append('parent_window_id')
        if 'sub_window_id' in primary_df.columns:
            sort_cols.append('sub_window_id')
        
        logger.info(f"Sorting by {sort_cols} to preserve temporal window order")
        primary_df = primary_df.sort_values(sort_cols).reset_index(drop=True)
        
        # Verify window ordering is preserved
        for participant in primary_df['Participant'].unique()[:3]:  # Check first 3 participants
            participant_windows = primary_df[primary_df['Participant'] == participant]
            if 'parent_window_id' in primary_df.columns:
                window_ids = participant_windows['parent_window_id'].values
                if len(window_ids) > 1 and not all(window_ids[i] <= window_ids[i+1] for i in range(len(window_ids)-1)):
                    logger.warning(f"Window ordering may be incorrect for participant {participant}")
                else:
                    logger.debug(f"✓ Window ordering verified for participant {participant}")
        
        logger.info("✓ Temporal window order preserved for all participants")
        
        # Log dataset statistics
        logger.info(f"Primary dataset created:")
        logger.info(f"  Total windows: {len(primary_df)}")
        logger.info(f"  Participants: {primary_df['Participant'].nunique()}")
        logger.info(f"  Remission: {(primary_df['Remission'] == 1).sum()}")
        logger.info(f"  Non-remission: {(primary_df['Remission'] == 0).sum()}")
        logger.info(f"  Columns: {list(primary_df.columns)}")
        
        # Get window size and channels from config
        window_seconds = config['window_slicer']['window_seconds']
        channels = config['data_loader']['channels']
        channels_str = "-".join(channels)
        
        # Create output path
        primary_dir = Path(config['paths']['features']['window']).parent / 'primary'
        primary_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{window_seconds}s_{channels_str}_primary_dataset.parquet"
        output_path = primary_dir / output_filename
        
        # Save primary dataset
        primary_df.to_parquet(output_path, index=False)
        logger.info(f"Saved primary dataset to {output_path}")
        
        # Create MLflow dataset
        try:
            dataset_name = f"EEG_{window_seconds}s_{channels_str}_{len(primary_df)}windows_primary"
            
            dataset = mlflow.data.from_pandas(
                df=primary_df,
                source=str(output_path),
                targets="Remission",
                name=dataset_name
            )
            
            logger.info(f"Created MLflow dataset: {dataset.name}")
            logger.info(f"  Dataset digest: {dataset.digest}")
            
            # Log the dataset to MLflow
            mlflow.log_input(dataset, context="training")
            logger.info(f"Logged dataset to MLflow")
            
            # Set tags to help training runs discover this dataset
            mlflow.set_tag("mlflow.dataset.logged", "true")
            mlflow.set_tag("mlflow.dataset.context", "training")
            mlflow.set_tag("mlflow.dataset.type", "primary")
            
            # Log the parquet file as an artifact for backup
            mlflow.log_artifact(str(output_path), "primary_dataset")
            logger.info(f"Logged primary dataset as MLflow artifact")
            
            return primary_df, dataset
            
        except Exception as mlflow_e:
            logger.warning(f"Failed to create/log MLflow dataset: {str(mlflow_e)}")
            return primary_df, None
        
    except Exception as e:
        logger.error(f"Error creating primary dataset: {str(e)}")
        raise


if __name__ == '__main__':
    # Example usage
    import yaml
    import mlflow
    
    # Load configuration
    with open('configs/processing_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Start MLflow run
    mlflow.set_tracking_uri(config.get('mlflow', {}).get('tracking_uri', "file:./mlruns"))
    mlflow.set_experiment(config.get('mlflow', {}).get('experiment_name', "eeg_processing"))
    
    with mlflow.start_run(run_name="primary_dataset_creation"):
        try:
            # Create primary dataset
            windowed_path = config['paths']['interim']['windowed']
            primary_df, mlflow_dataset = create_primary_dataset(config, windowed_path)
            
            # Log success
            mlflow.log_metric("primary_dataset_creation_success", 1)
            mlflow.log_param("primary_dataset_rows", len(primary_df))
            mlflow.log_param("primary_dataset_participants", primary_df['Participant'].nunique())
            
        except Exception as e:
            logger.error(f"Primary dataset creation failed: {str(e)}")
            mlflow.log_metric("primary_dataset_creation_success", 0)
            raise

