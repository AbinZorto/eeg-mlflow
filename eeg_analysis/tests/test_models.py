import pytest
import numpy as np
import pandas as pd
from src.models.model_utils import create_classifier
import src.models.trainer as trainer_module
import src.models.deep_learning_trainer as deep_learning_trainer_module
from src.models.trainer import Trainer
from src.models.evaluation import ModelEvaluator
from src.models.deep_learning_trainer import DeepLearningTrainer, PyTorchMLPClassifier, _resolve_training_batch_config

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


@pytest.fixture
def deep_learning_config():
    """Generate minimal valid deep learning trainer configuration."""
    return {
        'model_type': 'pytorch_mlp',
        'deep_learning': {
            'pytorch_mlp': {
                'hidden_layers': [32, 16],
                'dropout_rate': 0.1,
                'weight_decay': 0.001,
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 3,
                'early_stopping_patience': 2,
                'batch_norm': True,
                'activation': 'relu',
                'optimizer': 'adam',
            }
        },
        'model': {},
        'paths': {
            'models': 'models/test'
        },
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

    def test_balance_groups_for_lopo_undersamples_by_group(self, training_config):
        trainer = Trainer({**training_config, 'random_seed': 7})
        X = pd.DataFrame({'feature_0': np.arange(10, dtype=float)})
        y = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        groups = pd.Series(['P0', 'P0', 'P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4'])

        X_bal, y_bal, groups_bal, info = trainer._balance_groups_for_lopo(X, y, groups, enabled=True)

        assert info['applied'] is True
        assert info['group_label_counts_before'] == {'0': 3, '1': 2}
        assert info['group_label_counts_after'] == {'0': 2, '1': 2}
        assert groups_bal.nunique() == 4
        assert y_bal.groupby(groups_bal).first().value_counts().to_dict() == {0: 2, 1: 2}

        original_group_sizes = groups.value_counts().to_dict()
        balanced_group_sizes = groups_bal.value_counts().to_dict()
        assert all(
            balanced_group_sizes[group_name] == original_group_sizes[group_name]
            for group_name in balanced_group_sizes
        )
        assert set(groups_bal.unique()).issubset(set(groups.unique()))

    def test_balance_groups_for_lopo_can_be_disabled(self, training_config):
        trainer = Trainer(training_config)
        X = pd.DataFrame({'feature_0': np.arange(6, dtype=float)})
        y = pd.Series([0, 0, 0, 0, 1, 1])
        groups = pd.Series(['P0', 'P0', 'P1', 'P1', 'P2', 'P2'])

        X_bal, y_bal, groups_bal, info = trainer._balance_groups_for_lopo(X, y, groups, enabled=False)

        pd.testing.assert_frame_equal(X_bal, X)
        pd.testing.assert_series_equal(y_bal, y)
        pd.testing.assert_series_equal(groups_bal, groups)
        assert info['applied'] is False
        assert info['reason'] == 'disabled'

    def test_shuffle_training_rows_is_deterministic_and_preserves_alignment(self, training_config):
        trainer = Trainer({**training_config, 'random_seed': 11})
        X = pd.DataFrame({'feature_0': [10.0, 20.0, 30.0, 40.0]})
        y = pd.Series([0, 1, 0, 1])

        X_shuffled_a, y_shuffled_a, info_a = trainer._shuffle_training_rows(X, y, seed_offset=3)
        X_shuffled_b, y_shuffled_b, info_b = trainer._shuffle_training_rows(X, y, seed_offset=3)

        pd.testing.assert_frame_equal(X_shuffled_a, X_shuffled_b)
        pd.testing.assert_series_equal(y_shuffled_a, y_shuffled_b)
        assert info_a["random_seed"] == info_b["random_seed"] == 14
        assert info_a["applied"] is True

        original_pairs = list(zip(X["feature_0"].tolist(), y.tolist()))
        shuffled_pairs = list(zip(X_shuffled_a["feature_0"].tolist(), y_shuffled_a.tolist()))
        assert shuffled_pairs != original_pairs
        assert sorted(shuffled_pairs) == sorted(original_pairs)

    def test_infer_feature_selection_fold_target_class(self, training_config):
        trainer = Trainer(training_config)

        remission = trainer._infer_feature_selection_fold_target_class(
            [{"true_label": 1}, {"true_label": 1}]
        )
        non_remission = trainer._infer_feature_selection_fold_target_class(
            [{"true_label": 0}, {"true_label": 0}]
        )
        mixed = trainer._infer_feature_selection_fold_target_class(
            [{"true_label": 1}, {"true_label": 0}]
        )
        unknown = trainer._infer_feature_selection_fold_target_class(
            [{"true_label": None}, {"predicted_label": 1}]
        )

        assert remission == "remission"
        assert non_remission == "non_remission"
        assert mixed == "mixed"
        assert unknown == "unknown"

    def test_create_and_train_model_disables_class_weight_when_smote_enabled(self, training_config, monkeypatch):
        captured = {}

        class DummyEstimator:
            def fit(self, X_train, y_train):
                captured['fit_shape'] = X_train.shape
                captured['fit_target_size'] = len(y_train)
                return self

        def fake_create_classifier(model_type, model_params):
            captured['model_type'] = model_type
            captured['model_params'] = dict(model_params)
            return DummyEstimator()

        monkeypatch.setattr(trainer_module, 'create_classifier', fake_create_classifier)

        trainer = Trainer(
            {
                **training_config,
                'use_smote': True,
                'model': {
                    'params': {
                        'random_forest': {
                            'n_estimators': 10,
                            'min_samples_leaf': 2,
                            'random_state': 42,
                            'class_weight': 'balanced',
                        }
                    }
                },
            }
        )

        trainer._create_and_train_model(
            pd.DataFrame({'feature_0': [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 0, 1, 1]),
        )

        assert captured['model_type'] == 'random_forest'
        assert captured['model_params']['class_weight'] is None
        assert captured['fit_shape'] == (4, 1)

    def test_trainer_uses_model_level_smote_when_root_flag_missing(self, training_config):
        config_without_root_flag = {key: value for key, value in training_config.items() if key != 'use_smote'}
        trainer = Trainer(
            {
                **config_without_root_flag,
                'model': {
                    **training_config['model'],
                    'use_smote': True,
                },
            }
        )

        assert trainer.use_smote is True

    def test_trainer_root_level_smote_overrides_model_level_flag(self, training_config):
        trainer = Trainer(
            {
                **training_config,
                'use_smote': False,
                'model': {
                    **training_config['model'],
                    'use_smote': True,
                },
            }
        )

        assert trainer.use_smote is False


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

    def test_pytorch_mlp_fit_handles_smote_requested_when_imbalanced_learn_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(deep_learning_trainer_module, "IMBALANCE_AVAILABLE", False)

        classifier = PyTorchMLPClassifier(
            hidden_layers=[8],
            dropout_rate=0.0,
            weight_decay=0.0,
            learning_rate=0.001,
            batch_size=8,
            epochs=1,
            early_stopping_patience=1,
            batch_norm=False,
            use_smote=True,
            use_multi_gpu=False,
        )

        X = pd.DataFrame(np.random.randn(8, 4), columns=[f"feature_{i}" for i in range(4)])
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        classifier.fit(X, y)

        assert classifier.use_smote is False


class TestDeepLearningTrainer:
    def test_deep_learning_trainer_reads_root_level_smote_override(self, deep_learning_config):
        trainer = DeepLearningTrainer({**deep_learning_config, 'use_smote': True})

        assert trainer.use_smote is True
        assert trainer.model_params['use_smote'] is True

    def test_deep_learning_trainer_reads_model_level_smote_fallback(self, deep_learning_config):
        trainer = DeepLearningTrainer(
            {
                **deep_learning_config,
                'model': {'use_smote': True},
            }
        )

        assert trainer.use_smote is True
        assert trainer.model_params['use_smote'] is True
