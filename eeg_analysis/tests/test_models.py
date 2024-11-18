import pytest
import numpy as np
import pandas as pd
from src.models.model_utils import create_classifier
from src.models.patient_trainer import PatientLevelTrainer
from src.models.window_trainer import WindowLevelTrainer

@pytest.fixture
def sample_training_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Assign to participants
    n_participants = 50
    participants = [f'P{i:03d}' for i in range(n_participants)]
    participant_ids = np.random.choice(participants, n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Participant'] = participant_ids
    df['Remission'] = y
    
    return df

@pytest.fixture
def training_config():
    """Generate training configuration."""
    return {
        'model_name': 'test_model',
        'classifier': 'random_forest',
        'classifier_params': {
            'n_estimators': 100,
            'min_samples_leaf': 20
        },
        'output_path': 'models/test'
    }

class TestModelUtils:
    def test_classifier_creation(self):
        """Test classifier creation with different configurations."""
        classifier_types = ['random_forest', 'gradient_boosting', 
                          'logistic_regression', 'svm']
        
        for clf_type in classifier_types:
            clf = create_classifier(clf_type)
            assert clf is not None
            assert len(clf.named_steps) == 2  # Scaler and classifier
            assert 'scaler' in clf.named_steps
            assert 'classifier' in clf.named_steps
    
    def test_invalid_classifier(self):
        """Test error handling for invalid classifier type."""
        with pytest.raises(ValueError):
            create_classifier('invalid_classifier')

class TestPatientTrainer:
    def test_patient_aggregation(self, sample_training_data):
        """Test window-to-patient feature aggregation."""
        trainer = PatientLevelTrainer({})
        patient_df = trainer.aggregate_windows(sample_training_data)
        
        # Check result properties
        assert len(patient_df) == len(sample_training_data['Participant'].unique())
        assert all(col in patient_df.columns for col in ['Participant', 'Remission'])
        
        # Check aggregation functions
        feature_cols = sample_training_data.columns.difference(['Participant', 'Remission'])
        for col in feature_cols:
            assert f"{col}_mean" in patient_df.columns
            assert f"{col}_std" in patient_df.columns
    
    def test_patient_training(self, sample_training_data, training_config):
        """Test patient-level model training."""
        trainer = PatientLevelTrainer(training_config)
        
        # Split features and target
        X = sample_training_data.drop(['Participant', 'Remission'], axis=1)
        y = sample_training_data['Remission']
        groups = sample_training_data['Participant']
        
        # Train and evaluate
        metrics, predictions = trainer.train_evaluate(X, y, groups)
        
        # Check results
        assert all(metric in metrics for metric in 
                  ['accuracy', 'precision', 'recall', 'f1'])
        assert len(predictions) == len(groups.unique())

class TestWindowTrainer:
    def test_window_aggregation(self, sample_training_data):
        """Test window prediction aggregation to patient level."""
        trainer = WindowLevelTrainer({})
        window_preds = np.random.randint(0, 2, len(sample_training_data))
        window_probs = np.random.random(len(sample_training_data))
        
        patient_pred, patient_prob = trainer.calculate_patient_prediction(
            window_preds, window_probs
        )
        
        assert isinstance(patient_pred, int)
        assert isinstance(patient_prob, float)
        assert 0 <= patient_prob <= 1
    
    def test_window_training(self, sample_training_data, training_config):
        """Test window-level model training."""
        trainer = WindowLevelTrainer(training_config)
        
        # Split features and target
        X = sample_training_data.drop(['Participant', 'Remission'], axis=1)
        y = sample_training_data['Remission']
        groups = sample_training_data['Participant']
        
        # Train and evaluate
        metrics, window_preds, patient_preds = trainer.train_evaluate(X, y, groups)
        
        # Check results
        assert 'window_level' in metrics
        assert 'patient_level' in metrics
        assert len(window_preds) == len(sample_training_data)
        assert len(patient_preds) == len(groups.unique())