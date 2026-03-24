import pytest
import numpy as np
import pandas as pd
from src.models.model_utils import create_classifier
from src.models.trainer import Trainer
from src.models.evaluation import ModelEvaluator
from src.models.deep_learning_trainer import PyTorchMLPClassifier, _resolve_training_batch_config

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
    """Generate minimal valid window trainer configuration."""
    return {
        'model_type': 'random_forest',
        'model': {
            'params': {
                'random_forest': {
                    'n_estimators': 10,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            }
        },
        'paths': {
            'models': 'models/test'
        },
        'metrics': {
            'window': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        },
        'use_smote': False
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

class TestTrainer:
    def test_window_to_patient_prediction_aggregation(self, training_config):
        """Trainer should retain patient-level aggregation outputs."""
        trainer = Trainer(training_config)
        groups = pd.Series(['P001', 'P001', 'P001', 'P001'])
        y_true = pd.Series([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.2, 0.7])

        patient_pred = trainer._create_patient_prediction(
            fold_idx=0,
            groups=groups,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        assert patient_pred['participant'] == 'P001'
        assert patient_pred['true_label'] == 1
        assert patient_pred['predicted_label'] == 1
        assert patient_pred['probability'] == 0.75
        assert patient_pred['n_windows'] == 4
        assert patient_pred['n_positive_windows'] == 3


class TestEvaluator:
    def test_patient_level_metrics_available(self):
        """Patient-level metrics remain available for window-based aggregations."""
        evaluator = ModelEvaluator()
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        y_prob = np.array([0.8, 0.1, 0.4, 0.2])

        metrics = evaluator.evaluate_patient_predictions(y_true, y_pred, y_prob)

        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'specificity' in metrics
        assert 'sensitivity' in metrics
        assert 'roc_auc' in metrics
        assert 'n_patients' in metrics
        assert metrics['n_patients'] == 4


class TestDeepLearningBatching:
    def test_pytorch_mlp_defaults_to_single_gpu_mode(self):
        classifier = PyTorchMLPClassifier(use_multi_gpu=False)

        assert classifier.use_multi_gpu is False

    def test_drop_last_when_tail_batch_is_unsafe_for_dataparallel_batchnorm(self):
        config = _resolve_training_batch_config(dataset_size=3970, batch_size=128, num_devices=2)

        assert config["batch_size"] == 128
        assert config["drop_last"] is True

    def test_keep_last_batch_when_tail_batch_is_large_enough(self):
        config = _resolve_training_batch_config(dataset_size=4050, batch_size=128, num_devices=2)

        assert config["drop_last"] is False
