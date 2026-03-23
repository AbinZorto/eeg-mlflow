from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_MODEL_FAMILY: Dict[str, str] = {
    "random_forest": "traditional",
    "gradient_boosting": "traditional",
    "logistic_regression": "traditional",
    "logistic_regression_l1": "traditional",
    "svm_rbf": "traditional",
    "svm_linear": "traditional",
    "extra_trees": "traditional",
    "ada_boost": "traditional",
    "knn": "traditional",
    "decision_tree": "traditional",
    "sgd": "traditional",
    "xgboost_gpu": "boosting",
    "catboost_gpu": "boosting",
    "lightgbm_gpu": "boosting",
    "pytorch_mlp": "deep_learning",
    "keras_mlp": "deep_learning",
    "hybrid_1dcnn_lstm": "deep_learning",
    "advanced_hybrid_1dcnn_lstm": "deep_learning",
    "efficient_tabular_mlp": "deep_learning",
    "advanced_lstm": "deep_learning",
    "advanced_1dcnn": "deep_learning",
}

DEFAULT_EXPERIMENT_GROUP_BY_FAMILY: Dict[str, str] = {
    "traditional": "eeg_traditional_models",
    "boosting": "eeg_boosting_gpu",
    "deep_learning": "eeg_deep_learning_gpu",
    "other": "eeg_other_models",
}

def load_config_yaml(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_models_from_sections(config: Dict[str, Any]) -> List[str]:
    models: List[str] = []

    model_params = config.get("model", {}).get("params", {})
    if isinstance(model_params, dict):
        models.extend(str(name) for name in model_params.keys())

    deep_learning = config.get("deep_learning", {})
    if isinstance(deep_learning, dict):
        models.extend(str(name) for name in deep_learning.keys())

    model_registry = config.get("model_registry", {})
    if isinstance(model_registry, dict):
        models.extend(str(name) for name in model_registry.keys())

    return _dedupe_preserve_order(models)


def _extract_family_membership(config: Dict[str, Any]) -> Dict[str, str]:
    family_membership: Dict[str, str] = {}
    model_families = config.get("model_families", {})
    if not isinstance(model_families, dict):
        return family_membership

    for family, names in model_families.items():
        if not isinstance(names, (list, tuple, set)):
            continue
        family_name = str(family).strip().lower()
        for model_name in names:
            family_membership[str(model_name)] = family_name

    return family_membership


def get_model_metadata(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build normalized model metadata from config.

    Priority order for metadata:
    1) model_registry.<model>.* (if present)
    2) model_families.<family>: [models] for family
    3) DEFAULT_MODEL_FAMILY or inferred section fallback
    """
    models = _extract_models_from_sections(config)
    family_membership = _extract_family_membership(config)
    model_registry = config.get("model_registry", {})
    if not isinstance(model_registry, dict):
        model_registry = {}

    experiment_groups = config.get("experiment_groups", {})
    if not isinstance(experiment_groups, dict):
        experiment_groups = {}

    deep_learning_models = config.get("deep_learning", {})
    deep_learning_names = set(deep_learning_models.keys()) if isinstance(deep_learning_models, dict) else set()

    metadata: Dict[str, Dict[str, Any]] = {}
    for model_name in models:
        entry = model_registry.get(model_name, {})
        if not isinstance(entry, dict):
            entry = {}

        family = entry.get("family")
        if family is None:
            family = family_membership.get(model_name)
        if family is None:
            family = DEFAULT_MODEL_FAMILY.get(model_name)
        if family is None:
            family = "deep_learning" if model_name in deep_learning_names else "traditional"
        family = str(family).strip().lower()

        trainer = entry.get("trainer")
        if trainer is None:
            trainer = "deep_learning" if family == "deep_learning" else "sklearn"
        trainer = str(trainer).strip().lower()

        if "include_in_auto" in entry:
            include_in_auto = bool(entry.get("include_in_auto"))
        else:
            include_in_auto = trainer != "deep_learning"

        enabled = bool(entry.get("enabled", True))

        experiment_group = entry.get("experiment_group")
        if experiment_group is None:
            experiment_group = experiment_groups.get(family)
        if experiment_group is None:
            experiment_group = DEFAULT_EXPERIMENT_GROUP_BY_FAMILY.get(family, DEFAULT_EXPERIMENT_GROUP_BY_FAMILY["other"])

        metadata[model_name] = {
            "family": family,
            "trainer": trainer,
            "include_in_auto": include_in_auto,
            "enabled": enabled,
            "experiment_group": str(experiment_group),
        }

    return metadata


def get_model_metadata_from_path(config_path: str | Path) -> Dict[str, Dict[str, Any]]:
    return get_model_metadata(load_config_yaml(config_path))


def get_available_models(
    config: Dict[str, Any],
    *,
    include_disabled: bool = False,
) -> List[str]:
    metadata = get_model_metadata(config)

    models: List[str] = []
    for model_name, info in metadata.items():
        if not include_disabled and not info.get("enabled", True):
            continue
        models.append(model_name)
    return models


def get_available_models_from_path(
    config_path: str | Path,
    *,
    include_disabled: bool = False,
) -> List[str]:
    config = load_config_yaml(config_path)
    return get_available_models(config, include_disabled=include_disabled)


def get_auto_models(config: Dict[str, Any]) -> List[str]:
    metadata = get_model_metadata(config)
    models: List[str] = []
    for model_name, info in metadata.items():
        if not info.get("enabled", True):
            continue
        if not info.get("include_in_auto", True):
            continue
        models.append(model_name)
    return models


def is_deep_learning_model(config: Dict[str, Any], model_name: str) -> bool:
    metadata = get_model_metadata(config)
    info = metadata.get(model_name)
    if info is None:
        return False
    return info.get("trainer") == "deep_learning" or info.get("family") == "deep_learning"
