# EEG Analysis Pipeline

A machine learning pipeline for analyzing EEG data to predict depression remission.

## Project Structure

```
eeg_analysis/
├── data/                # Data directory
├── models/             # Saved models
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
├── mlruns/           # MLflow tracking
├── tests/            # Test files
└── configs/          # Configuration files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Data Processing:
```bash
python run_pipeline.py process --config configs/processing_config.yaml
```

2. Model Training:
```bash
python run_pipeline.py train --config configs/model_config.yaml
```

3. View Results:
```bash
mlflow ui
```

## Features

- EEG signal processing pipeline
- Feature extraction (spectral, statistical, complexity measures)
- Two-level analysis:
  - Window-level classification
  - Patient-level classification
- MLflow experiment tracking
- Comprehensive evaluation metrics

## Configuration

Edit the YAML files in `configs/` to modify:
- Processing parameters
- Model hyperparameters
- Training settings

## Development

1. Run tests:
```bash
pytest tests/
```

2. Format code:
```bash
black src/ tests/
```

## License

[Your License]

## Contributors

[Your Name]