import hashlib
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
import shutil
from datetime import datetime
import numpy as np

class DataVersioner:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / 'versions'
        self.metadata_path = self.base_path / 'metadata.json'
        self.versions_path.mkdir(parents=True, exist_ok=True)
        
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'versions': {}}

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_hash(self, df):
        # Convert NumPy arrays to strings before hashing
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if isinstance(x, np.ndarray) else x)
        
        return hashlib.md5(pd.util.hash_pandas_object(df_copy).values).hexdigest()

    def save_version(self, df: pd.DataFrame, version_info: Dict[str, Any]) -> str:
        data_hash = self._compute_hash(df)
        timestamp = datetime.now().isoformat()
        
        version_id = f"v_{timestamp.replace(':', '-')}"
        version_path = self.versions_path / f"{version_id}.parquet"
        
        # Save data
        df.to_parquet(version_path)
        
        # Update metadata
        self.metadata['versions'][version_id] = {
            'hash': data_hash,
            'timestamp': timestamp,
            'path': str(version_path),
            'info': version_info
        }
        self._save_metadata()
        
        return version_id

    def load_version(self, version_id: str) -> pd.DataFrame:
        if version_id not in self.metadata['versions']:
            raise ValueError(f"Version {version_id} not found")
        
        version_path = Path(self.metadata['versions'][version_id]['path'])
        return pd.read_parquet(version_path)

    def get_version_info(self, version_id: str) -> Dict[str, Any]:
        return self.metadata['versions'].get(version_id, {})

    def list_versions(self) -> Dict[str, Dict[str, Any]]:
        return self.metadata['versions']

class ProcessingTracker:
    def __init__(self, versioner: DataVersioner):
        self.versioner = versioner
        
    def track_processing(self, input_df: pd.DataFrame, 
                        process_name: str,
                        parameters: Dict[str, Any]) -> str:
        version_info = {
            'process_name': process_name,
            'parameters': parameters,
            'input_hash': self.versioner._compute_hash(input_df)
        }
        return self.versioner.save_version(input_df, version_info)

def create_data_versioner(base_path: str) -> DataVersioner:
    return DataVersioner(base_path)