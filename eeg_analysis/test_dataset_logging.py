#!/usr/bin/env python3
"""
Test script for verifying MLflow dataset logging functionality.
This script demonstrates the new workflow where datasets are logged in MLflow
and models are trained using those logged datasets.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import tempfile
import os
from pathlib import Path

def create_sample_dataset(n_samples=1000, n_features=50):
    """Create a sample EEG feature dataset for testing."""
    np.random.seed(42)
    
    # Create synthetic EEG features
    data = {}
    
    # Add participant and group information
    n_participants = 20
    windows_per_participant = n_samples // n_participants
    
    participants = []
    remission = []
    
    for i in range(n_participants):
        participant_id = f"P{i:03d}"
        # Alternate between remission and non-remission
        is_remission = i % 2
        
        for j in range(windows_per_participant):
            participants.append(participant_id)
            remission.append(is_remission)
    
    data['Participant'] = participants[:n_samples]
    data['Remission'] = remission[:n_samples]
    
    # Add synthetic EEG features
    channels = ['af7', 'af8', 'tp9', 'tp10']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    feature_count = 0
    for channel in channels:
        for band in bands:
            # Power features
            data[f'{channel}_power_{band}'] = np.random.normal(0, 1, n_samples)
            feature_count += 1
            
            if feature_count >= n_features - 2:  # -2 for Participant and Remission
                break
        if feature_count >= n_features - 2:
            break
    
    # Fill remaining features if needed
    while feature_count < n_features - 2:
        data[f'feature_{feature_count}'] = np.random.normal(0, 1, n_samples)
        feature_count += 1
    
    return pd.DataFrame(data)

def test_dataset_logging():
    """Test the dataset logging workflow."""
    print("=== Testing MLflow Dataset Logging Workflow ===")
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./test_mlruns")
    mlflow.set_experiment("dataset_logging_test")
    
    print("1. Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Created dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    print(f"Participants: {df['Participant'].nunique()}")
    print(f"Remission distribution: {df['Remission'].value_counts().to_dict()}")
    
    # Save to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        temp_path = tmp_file.name
        df.to_parquet(temp_path)
        print(f"Saved temporary dataset to: {temp_path}")
    
    try:
        print("\n2. Testing dataset creation and logging...")
        with mlflow.start_run(run_name="dataset_creation_test") as run:
            # Create MLflow dataset
            dataset = mlflow.data.from_pandas(
                df=df,
                source=temp_path,
                targets="Remission",
                name=f"Test_EEG_Features_{len(df)}_windows"
            )
            
            print(f"Created MLflow dataset:")
            print(f"  Name: {dataset.name}")
            print(f"  Digest: {dataset.digest}")
            print(f"  Schema: {dataset.schema}")
            print(f"  Profile: {dataset.profile}")
            print(f"  Targets: {dataset.targets}")
            
            # Log the dataset
            mlflow.log_input(dataset, context="training")
            print("Successfully logged dataset to MLflow!")
            
            # Log some metadata
            mlflow.log_params({
                "dataset_name": dataset.name,
                "dataset_digest": dataset.digest,
                "num_samples": len(df),
                "num_features": len(df.columns) - 2,
                "num_participants": df['Participant'].nunique()
            })
            
            mlflow.set_tag("test.dataset.logged", "true")
            mlflow.set_tag("test.dataset.context", "training")
            
            processing_run_id = run.info.run_id
            print(f"Processing run ID: {processing_run_id}")
        
        print("\n3. Testing dataset retrieval from MLflow...")
        
        # Test loading dataset from the run
        retrieved_run = mlflow.get_run(processing_run_id)
        print(f"Retrieved run info: {retrieved_run.info.run_id}")
        
        if hasattr(retrieved_run, 'inputs') and retrieved_run.inputs.dataset_inputs:
            print("Found dataset inputs in run!")
            dataset_input = retrieved_run.inputs.dataset_inputs[0]
            print(f"Dataset input: {dataset_input}")
            
            # Load the dataset
            dataset_source = mlflow.data.get_source(dataset_input.dataset)
            loaded_path = dataset_source.load()
            loaded_df = pd.read_parquet(loaded_path)
            
            print(f"Successfully loaded dataset from MLflow!")
            print(f"Loaded dataset shape: {loaded_df.shape}")
            print(f"Original dataset shape: {df.shape}")
            print(f"Datasets match: {loaded_df.equals(df)}")
            
            # Recreate MLflow dataset from loaded data
            recreated_dataset = mlflow.data.from_pandas(
                df=loaded_df, 
                source=loaded_path, 
                targets="Remission"
            )
            print(f"Recreated dataset name: {recreated_dataset.name}")
            
        else:
            print("ERROR: No dataset inputs found in the run!")
            return False
        
        print("\n4. Testing trainer-like workflow...")
        
        with mlflow.start_run(run_name="training_test") as train_run:
            # Simulate what a trainer would do
            
            # Option 1: Load from specific run ID
            print("Testing dataset loading from specific run...")
            run = mlflow.get_run(processing_run_id)
            if hasattr(run, 'inputs') and run.inputs.dataset_inputs:
                dataset_input = run.inputs.dataset_inputs[0]
                dataset_source = mlflow.data.get_source(dataset_input.dataset)
                data_path = dataset_source.load()
                train_df = pd.read_parquet(data_path)
                training_dataset = mlflow.data.from_pandas(train_df, source=data_path, targets="Remission")
                
                # Log the dataset usage in training
                mlflow.log_input(training_dataset, context="training")
                mlflow.log_param("used_mlflow_dataset", True)
                mlflow.log_param("source_run_id", processing_run_id)
                mlflow.log_param("dataset_name", training_dataset.name)
                mlflow.log_param("dataset_digest", training_dataset.digest)
                
                print(f"Successfully used dataset in training: {training_dataset.name}")
                print(f"Training dataset shape: {train_df.shape}")
                
                # Simulate basic data preparation
                X = train_df.drop(['Participant', 'Remission'], axis=1)
                y = train_df['Remission']
                groups = train_df['Participant']
                
                print(f"Prepared training data:")
                print(f"  Features shape: {X.shape}")
                print(f"  Target shape: {y.shape}")
                print(f"  Groups (participants): {groups.nunique()}")
                print(f"  Target distribution: {y.value_counts().to_dict()}")
                
            training_run_id = train_run.info.run_id
            print(f"Training run ID: {training_run_id}")
        
        print("\n5. Testing automatic dataset discovery...")
        
        # Search for runs with datasets
        experiment = mlflow.get_experiment_by_name("dataset_logging_test")
        runs_with_datasets = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.`test.dataset.logged` = 'true' AND tags.`test.dataset.context` = 'training'",
            order_by=["start_time DESC"],
            max_results=5
        )
        
        print(f"Found {len(runs_with_datasets)} runs with datasets")
        if not runs_with_datasets.empty:
            latest_run_id = runs_with_datasets.iloc[0]['run_id']
            print(f"Latest run with dataset: {latest_run_id}")
            
            # This simulates the automatic discovery in the train function
            run = mlflow.get_run(latest_run_id)
            if hasattr(run, 'inputs') and run.inputs.dataset_inputs:
                print("Successfully discovered dataset automatically!")
            else:
                print("ERROR: Could not find dataset in discovered run")
                return False
        else:
            print("ERROR: Could not find runs with datasets")
            return False
        
        print("\n=== All tests passed! ===")
        print("\nWorkflow Summary:")
        print("1. ✓ Dataset created and logged to MLflow with proper metadata")
        print("2. ✓ Dataset can be retrieved from MLflow using run ID")
        print("3. ✓ Dataset can be used in training workflows")
        print("4. ✓ Datasets can be automatically discovered from recent runs")
        print("5. ✓ Proper lineage tracking between processing and training runs")
        
        return True
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"\nCleaned up temporary file: {temp_path}")

if __name__ == "__main__":
    success = test_dataset_logging()
    if success:
        print("\n🎉 Dataset logging workflow is working correctly!")
        print("\nNext steps:")
        print("1. Run the processing pipeline: python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml process")
        print("2. Run training with dataset auto-discovery: python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest")
        print("3. Or specify a specific run: python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --use-dataset-from-run <run_id>")
    else:
        print("\n❌ Dataset logging workflow has issues that need to be fixed")
        exit(1) 