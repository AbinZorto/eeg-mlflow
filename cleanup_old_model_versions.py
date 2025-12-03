#!/usr/bin/env python3
"""
Clean up old model versions from MLflow Model Registry.
Keeps only the latest version of each model.
"""

import mlflow
from mlflow.tracking import MlflowClient

# Initialize client
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

def cleanup_model_versions(model_name: str, keep_latest_n: int = 1):
    """
    Delete old versions of a model, keeping only the latest N versions.
    
    Args:
        model_name: Name of the registered model
        keep_latest_n: Number of latest versions to keep (default: 1)
    """
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        # Sort by version number (descending)
        versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
        
        print(f"Found {len(versions_sorted)} versions of model '{model_name}'")
        
        # Keep the latest N, delete the rest
        versions_to_delete = versions_sorted[keep_latest_n:]
        
        if not versions_to_delete:
            print(f"No versions to delete (keeping latest {keep_latest_n})")
            return
        
        print(f"Deleting {len(versions_to_delete)} old versions...")
        for version in versions_to_delete:
            print(f"  Deleting version {version.version}...")
            client.delete_model_version(model_name, version.version)
        
        print(f"✓ Cleanup complete! Kept {keep_latest_n} latest version(s)")
        
    except Exception as e:
        print(f"Error cleaning up model '{model_name}': {e}")


def list_all_models():
    """List all registered models and their version counts."""
    models = client.search_registered_models()
    
    if not models:
        print("No registered models found.")
        return []
    
    print("\nRegistered Models:")
    print("-" * 60)
    
    model_names = []
    for model in models:
        versions = client.search_model_versions(f"name='{model.name}'")
        print(f"  {model.name}: {len(versions)} versions")
        model_names.append(model.name)
    
    print("-" * 60)
    return model_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old MLflow model versions")
    parser.add_argument("--model", type=str, help="Model name to clean up (if not provided, lists all models)")
    parser.add_argument("--keep", type=int, default=1, help="Number of latest versions to keep (default: 1)")
    parser.add_argument("--all", action="store_true", help="Clean up all registered models")
    
    args = parser.parse_args()
    
    if args.all:
        model_names = list_all_models()
        if model_names:
            print(f"\nCleaning up all {len(model_names)} models...")
            for model_name in model_names:
                cleanup_model_versions(model_name, keep_latest_n=args.keep)
    elif args.model:
        cleanup_model_versions(args.model, keep_latest_n=args.keep)
    else:
        list_all_models()
        print("\nUsage:")
        print("  # List all models")
        print("  python cleanup_old_model_versions.py")
        print("\n  # Clean up specific model, keep latest 1 version")
        print("  python cleanup_old_model_versions.py --model mamba2_eeg_d256_l2_m20 --keep 1")
        print("\n  # Clean up all models, keep latest 2 versions of each")
        print("  python cleanup_old_model_versions.py --all --keep 2")

