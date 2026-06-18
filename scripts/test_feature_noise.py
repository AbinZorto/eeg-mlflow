"""Standalone tests for feature noise injection.

Runs on Mac without requiring the full pipeline dependencies (imblearn, torch, etc.).
Tests only import BaseTrainer which has minimal dependencies.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'eeg_analysis')

# Direct import to avoid imblearn dependency chain
import importlib
spec = importlib.util.spec_from_file_location(
    "base_trainer",
    "eeg_analysis/src/models/base_trainer.py",
)
base_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_trainer)
BaseTrainer = base_trainer.BaseTrainer

# ── Test fixtures ──

def make_X(n=1000):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'feat_a': rng.normal(5.0, 2.0, n),
        'feat_b': rng.normal(-3.0, 0.5, n),
        'feat_c': rng.uniform(0, 10, n),
    })

def base_config():
    return {
        'model_name': 'test_model',
        'feature_noise': {'enabled': True, 'std': 1.0},
    }

# ── Tests ──

def test_disabled_returns_unchanged():
    X = make_X()
    trainer = BaseTrainer({'feature_noise': {'enabled': False, 'std': 0.0}})
    result = trainer._inject_feature_noise(X)
    pd.testing.assert_frame_equal(result, X)
    print("  ✓ test_disabled_returns_unchanged")

def test_preserves_shape_and_columns():
    X = make_X()
    trainer = BaseTrainer(base_config())
    result = trainer._inject_feature_noise(X)
    assert result.shape == X.shape
    assert list(result.columns) == list(X.columns)
    print("  ✓ test_preserves_shape_and_columns")

def test_changes_all_values():
    X = make_X()
    trainer = BaseTrainer(base_config())
    result = trainer._inject_feature_noise(X)
    assert not (result.values == X.values).any()
    print("  ✓ test_changes_all_values")

def test_is_zero_mean():
    X = make_X()
    trainer = BaseTrainer(base_config())
    result = trainer._inject_feature_noise(X)
    for col in result.columns:
        assert abs(result[col].mean()) < 0.1, f"{col} mean={result[col].mean():.4f}"
    print("  ✓ test_is_zero_mean")

def test_respects_std():
    X = make_X()
    trainer = BaseTrainer({'feature_noise': {'enabled': True, 'std': 3.0}})
    result = trainer._inject_feature_noise(X)
    for col in result.columns:
        assert 2.5 < result[col].std() < 3.5, f"{col} std={result[col].std():.4f}"
    print("  ✓ test_respects_std")

def test_independent_across_columns():
    X = make_X()
    trainer = BaseTrainer(base_config())
    result = trainer._inject_feature_noise(X)
    corr = result.corr()
    for i in range(len(result.columns)):
        for j in range(i + 1, len(result.columns)):
            assert abs(corr.iloc[i, j]) < 0.1
    print("  ✓ test_independent_across_columns")

def test_original_not_mutated():
    X = make_X()
    original = X.copy()
    trainer = BaseTrainer(base_config())
    _ = trainer._inject_feature_noise(X)
    pd.testing.assert_frame_equal(X, original)
    print("  ✓ test_original_not_mutated")

def test_different_calls_different_noise():
    X = make_X()
    trainer = BaseTrainer(base_config())
    r1 = trainer._inject_feature_noise(X)
    r2 = trainer._inject_feature_noise(X)
    assert not (r1.values == r2.values).any()
    print("  ✓ test_different_calls_different_noise")

def test_config_defaults_disabled():
    X = make_X()
    trainer = BaseTrainer({'model_name': 'test'})
    result = trainer._inject_feature_noise(X)
    pd.testing.assert_frame_equal(result, X)
    print("  ✓ test_config_defaults_disabled")

def test_single_column():
    """Noise works on single-column DataFrames (the inner-k=1 case)."""
    X = pd.DataFrame({'feat_a': np.arange(100, dtype=float)})
    trainer = BaseTrainer(base_config())
    result = trainer._inject_feature_noise(X)
    assert result.shape == (100, 1)
    assert list(result.columns) == ['feat_a']
    assert not (result.values == X.values).any()
    print("  ✓ test_single_column")

# ── Run ──

if __name__ == '__main__':
    print("Feature noise injection tests:")
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as e:
                print(f"  ✗ {name}: {e}")
    print("Done.")
