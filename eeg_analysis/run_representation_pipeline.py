import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import click
import mlflow
import yaml

from src.processing.data_loader import load_eeg_data
from src.processing.dc_offset import remove_dc_offset_eeg_data
from src.processing.downsampler import downsample_eeg_data
from src.processing.filter import filter_eeg_data
from src.processing.closed_finetune_dataset import create_closed_finetune_dataset
from src.processing.open_pretrain import pipeline as open_pretrain_pipeline
from src.processing.upsampler import upsample_eeg_data
from src.processing.window_slicer import slice_eeg_windows
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_mlflow_tracking(config):
    """Set up MLflow tracking with environment override and fallback experiment."""
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name is None:
        experiment_name = config.get("mlflow", {}).get("experiment_name", "eeg_representation")
        logger.info(f"Using experiment name from config: {experiment_name}")
    else:
        logger.info(f"Using experiment name from environment: {experiment_name}")

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            mlflow.set_experiment(experiment_name=experiment.name)
            logger.info(
                f"Found existing MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})"
            )
            return experiment.experiment_id

        logger.info(f"MLflow experiment '{experiment_name}' not found. Creating new one.")
        experiment_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        logger.info(f"Created and set experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    except Exception as e:
        fallback_name = f"{experiment_name}_fallback_1"
        logger.warning(
            f"Failed to get/create/set experiment '{experiment_name}': {e}. "
            f"Trying fallback '{fallback_name}'."
        )
        try:
            fallback = mlflow.get_experiment_by_name(fallback_name)
            if fallback is None:
                fallback_id = mlflow.create_experiment(fallback_name)
                mlflow.set_experiment(fallback_name)
                logger.info(f"Created and set fallback experiment: {fallback_name} (ID: {fallback_id})")
                return fallback_id

            mlflow.set_experiment(fallback_name)
            logger.info(f"Using existing fallback experiment: {fallback_name} (ID: {fallback.experiment_id})")
            return fallback.experiment_id
        except Exception as fallback_e:
            raise Exception(
                "MLflow setup failed for both requested and fallback experiments. "
                f"Original error: {e}; fallback error: {fallback_e}"
            )


@click.group()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """EEG representation pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["config_path"] = config


def _run_process_representation(config):
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_id = setup_mlflow_tracking(config)
        logger.info(f"MLflow experiment ID: {experiment_id} selected for representation processing.")
    except Exception as e:
        logger.error(f"Critical error setting up MLflow: {e}")
        raise

    with mlflow.start_run(run_name="closed_finetune_dataset_processing"):
        mlflow.log_params(config.get("processing_params", {}))
        try:
            logger.info("Loading raw EEG data...")
            raw_data = load_eeg_data(config)

            logger.info("Running preprocessing pipeline up to windowing...")
            upsampled = upsample_eeg_data(config, raw_data)
            filtered = filter_eeg_data(config, upsampled)
            downsampled = downsample_eeg_data(config, filtered)
            windowed = slice_eeg_windows(config, downsampled)
            _ = remove_dc_offset_eeg_data(config, windowed)

            logger.info("Building closed_finetune dataset...")
            windowed_path = config["paths"]["interim"]["windowed"]
            dataset_df, mlflow_dataset = create_closed_finetune_dataset(config, windowed_path)

            mlflow.log_metric("closed_finetune_dataset_processing_success", 1)
            mlflow.log_param("closed_finetune_dataset_rows", len(dataset_df))
            mlflow.log_param(
                "closed_finetune_dataset_participants",
                dataset_df["Participant"].nunique(),
            )

            if mlflow_dataset:
                mlflow.log_param("dataset_available_for_training", True)
                mlflow.set_tag("mlflow.dataset.logged", "true")
                mlflow.set_tag("mlflow.dataset.context", "closed_finetune")
                mlflow.set_tag("mlflow.dataset.type", "closed_finetune")
            else:
                mlflow.log_param("dataset_available_for_training", False)

            logger.info("Closed_finetune dataset processing completed successfully.")
        except Exception as e:
            mlflow.log_metric("closed_finetune_dataset_processing_success", 0)
            mlflow.log_param("error_message", str(e))
            logger.error(f"Closed_finetune dataset processing failed: {e}")
            raise


@cli.command(name="process-representation")
@click.pass_context
def process_representation(ctx):
    """Create the closed_finetune dataset from the core EEG pipeline."""
    config = ctx.obj["config"]
    _run_process_representation(config)


@cli.command(name="process-closed-finetune")
@click.pass_context
def process_closed_finetune(ctx):
    """Create the closed_finetune dataset from the core EEG pipeline."""
    _run_process_representation(ctx.obj["config"])


def _run_build_pretraining_dataset(config):
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", f"file:{REPO_ROOT / 'mlruns'}")
    if tracking_uri in {"mlruns", "./mlruns", "file:./mlruns"}:
        tracking_uri = f"file:{REPO_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)

    result = open_pretrain_pipeline.run(config)

    print("\nOpen-pretrain dataset build completed")
    print(f"  Output dir: {result.get('output_dir')}")
    print(
        f"  Subjects: {result.get('num_subjects')}, "
        f"Runs discovered: {result.get('num_runs')}, "
        f"Runs saved: {result.get('runs_saved')}"
    )
    print(f"  Sampling rate: {result.get('sampling_rate')} Hz")
    print()


@cli.command(name="build-pretraining-dataset")
@click.pass_context
def build_pretraining_dataset(ctx):
    """Build and register the open_pretrain dataset (stored under secondarydata)."""
    _run_build_pretraining_dataset(ctx.obj["config"])


@cli.command(name="build-open-pretrain")
@click.pass_context
def build_open_pretrain(ctx):
    """Build and register the open_pretrain dataset (stored under secondarydata)."""
    _run_build_pretraining_dataset(ctx.obj["config"])


@cli.command(name="convert-pretraining-dataset")
@click.option("--input-root", type=click.Path(exists=True), required=True, help="Input dataset root")
@click.option("--output-base", type=click.Path(), required=True, help="Output base directory")
@click.option("--factor", type=int, required=True, help="Window-size multiplier (>=2)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files")
@click.option(
    "--pad-remainder",
    type=str,
    default=None,
    help="Pad remainder group with 'nan' or numeric value instead of dropping",
)
def convert_pretraining_dataset(input_root, output_base, factor, overwrite, pad_remainder):
    """Convert pretraining dataset to a larger window size."""
    _run_convert_pretraining_dataset(input_root, output_base, factor, overwrite, pad_remainder)


def _run_convert_pretraining_dataset(input_root, output_base, factor, overwrite, pad_remainder):
    """Shared implementation for pretraining dataset window-size conversion."""
    converter_script = REPO_ROOT / "scripts" / "convert_open_pretrain_window_size.py"
    if not converter_script.exists():
        raise click.ClickException(f"Converter script not found: {converter_script}")

    cmd = [
        sys.executable,
        str(converter_script),
        "--input-root",
        str(input_root),
        "--output-base",
        str(output_base),
        "--factor",
        str(factor),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if pad_remainder is not None:
        cmd.extend(["--pad-remainder", str(pad_remainder)])

    logger.info("Running converter command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Dataset conversion failed with exit code {e.returncode}")


@cli.command(name="convert-open-pretrain")
@click.option("--input-root", type=click.Path(exists=True), required=True, help="Input dataset root")
@click.option("--output-base", type=click.Path(), required=True, help="Output base directory")
@click.option("--factor", type=int, required=True, help="Window-size multiplier (>=2)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files")
@click.option(
    "--pad-remainder",
    type=str,
    default=None,
    help="Pad remainder group with 'nan' or numeric value instead of dropping",
)
def convert_open_pretrain(input_root, output_base, factor, overwrite, pad_remainder):
    """Convert open_pretrain dataset to a larger window size."""
    _run_convert_pretraining_dataset(input_root, output_base, factor, overwrite, pad_remainder)


def _resolve_pretrain_models(cfg, explicit_models, all_models):
    configured_models = cfg.get("models") or {}
    if configured_models and not isinstance(configured_models, dict):
        raise click.ClickException("'models' in pretrain config must be a mapping")

    def _enabled(model_key):
        model_cfg = configured_models.get(model_key, {})
        if not isinstance(model_cfg, dict):
            return True
        return bool(model_cfg.get("enabled", True))

    enabled_models = [name for name in configured_models.keys() if _enabled(name)]

    if all_models:
        if not configured_models:
            return [cfg.get("model_name", "mamba")]
        if not enabled_models:
            raise click.ClickException("No enabled models found in pretrain config")
        return enabled_models

    if explicit_models:
        if not configured_models:
            return list(explicit_models)
        unknown = [name for name in explicit_models if name not in configured_models]
        if unknown:
            known = ", ".join(configured_models.keys())
            raise click.ClickException(f"Unknown model(s): {', '.join(unknown)}. Available models: {known}")
        disabled = [name for name in explicit_models if configured_models.get(name, {}).get("enabled", True) is False]
        if disabled:
            raise click.ClickException(
                f"Selected model(s) are disabled: {', '.join(disabled)}. "
                "Set models.<name>.enabled=true or pick a different model."
            )
        return list(explicit_models)

    if configured_models:
        active = cfg.get("active_model") or cfg.get("model_name")
        if active and active in configured_models:
            if not _enabled(active):
                raise click.ClickException(
                    f"Configured active model '{active}' is disabled. "
                    "Set models.<active_model>.enabled=true or choose another active model."
                )
            return [active]
        if enabled_models:
            return [enabled_models[0]]
        raise click.ClickException("No enabled models found in pretrain config")

    return [cfg.get("model_name", "mamba")]


def _build_model_specific_config(base_cfg, model_name, models_count):
    run_cfg = deepcopy(base_cfg)
    models_cfg = run_cfg.get("models") or {}

    if models_cfg:
        if model_name not in models_cfg:
            known = ", ".join(models_cfg.keys())
            raise click.ClickException(f"Unknown model '{model_name}'. Available models: {known}")
        model_overrides = deepcopy(models_cfg[model_name])
        if not isinstance(model_overrides, dict):
            raise click.ClickException(f"Config for model '{model_name}' must be a mapping")
        run_cfg.update(model_overrides)

    run_cfg["model_name"] = model_name
    run_cfg["active_model"] = model_name

    # Avoid checkpoint collisions when running more than one model in one invocation.
    if models_count > 1:
        base_save_dir = Path(run_cfg.get("save_dir", "./checkpoints"))
        run_cfg["save_dir"] = str((base_save_dir / model_name).resolve())

    return run_cfg


@cli.command(name="pretrain")
@click.option("--model", "models", multiple=True, help="Model profile(s) defined in pretrain config")
@click.option("--all-models", is_flag=True, help="Run every model profile in pretrain config")
@click.option("--distributed", is_flag=True, help="Pass --distributed to trainer")
@click.option("--backend", type=str, default="nccl", show_default=True, help="DDP backend for trainer")
@click.option("--resume", type=click.Path(exists=True), default="", help="Resume checkpoint path")
@click.pass_context
def pretrain(ctx, models, all_models, distributed, backend, resume):
    """Run representation pretraining for one or more configured models."""
    cfg = ctx.obj["config"]

    selected_models = _resolve_pretrain_models(cfg, models, all_models)
    logger.info(f"Selected models for pretraining: {selected_models}")

    default_trainer = REPO_ROOT / "eeg_analysis" / "src" / "training" / "pretrain_mamba.py"
    if not default_trainer.exists():
        raise click.ClickException(f"Default trainer script not found: {default_trainer}")

    for model_name in selected_models:
        run_cfg = _build_model_specific_config(cfg, model_name, len(selected_models))
        trainer_script = run_cfg.get("trainer_script", str(default_trainer))
        trainer_path = Path(trainer_script)
        if not trainer_path.is_absolute():
            trainer_path = (REPO_ROOT / trainer_path).resolve()
        if not trainer_path.exists():
            raise click.ClickException(f"Trainer script not found for model '{model_name}': {trainer_path}")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".{model_name}.yaml",
            prefix="pretrain_cfg_",
            delete=False,
        ) as tmp:
            yaml.safe_dump(run_cfg, tmp, sort_keys=False)
            temp_cfg_path = tmp.name

        cmd = [sys.executable, str(trainer_path), "--config", temp_cfg_path]
        if distributed:
            cmd.extend(["--distributed", "--backend", backend])
        if resume:
            cmd.extend(["--resume", resume])

        logger.info("Running pretraining command for '%s': %s", model_name, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Pretraining failed for model '{model_name}' with exit code {e.returncode}"
            )
        finally:
            try:
                os.remove(temp_cfg_path)
            except OSError:
                logger.warning(f"Could not remove temporary config: {temp_cfg_path}")


@cli.command(name="finetune")
@click.option("--model", "models", multiple=True, help="Model profile(s) defined in pretrain config")
@click.option("--all-models", is_flag=True, help="Run every model profile in pretrain config")
@click.option(
    "--pretrain-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to pretraining config (defaults to global --config)",
)
@click.option(
    "--finetune-config",
    type=click.Path(exists=True),
    default=str(REPO_ROOT / "eeg_analysis" / "configs" / "finetune.yaml"),
    show_default=True,
    help="Path to fine-tuning YAML config",
)
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to closed_finetune dataset")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory for finetuned models")
@click.pass_context
def finetune(ctx, models, all_models, pretrain_config, finetune_config, data_path, output_dir):
    """Run representation fine-tuning for one or more configured models."""
    pretrain_cfg_path = pretrain_config or ctx.obj["config_path"]
    pretrain_cfg = load_config(pretrain_cfg_path)

    selected_models = _resolve_pretrain_models(pretrain_cfg, models, all_models)
    logger.info(f"Selected models for fine-tuning: {selected_models}")

    default_trainer = REPO_ROOT / "eeg_analysis" / "src" / "training" / "finetune_mamba.py"
    if not default_trainer.exists():
        raise click.ClickException(f"Default fine-tuning trainer script not found: {default_trainer}")

    for model_name in selected_models:
        run_pretrain_cfg = _build_model_specific_config(pretrain_cfg, model_name, len(selected_models))
        trainer_script = run_pretrain_cfg.get("finetune_trainer_script", str(default_trainer))
        trainer_path = Path(trainer_script)
        if not trainer_path.is_absolute():
            trainer_path = (REPO_ROOT / trainer_path).resolve()
        if not trainer_path.exists():
            raise click.ClickException(
                f"Fine-tuning trainer script not found for model '{model_name}': {trainer_path}"
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".{model_name}.yaml",
            prefix="pretrain_cfg_",
            delete=False,
        ) as tmp:
            yaml.safe_dump(run_pretrain_cfg, tmp, sort_keys=False)
            temp_pretrain_cfg_path = tmp.name

        cmd = [
            sys.executable,
            str(trainer_path),
            "--config",
            str(finetune_config),
            "--pretrain-config",
            temp_pretrain_cfg_path,
        ]
        if data_path:
            cmd.extend(["--data-path", str(data_path)])
        if output_dir:
            model_output_dir = Path(output_dir)
            if len(selected_models) > 1:
                model_output_dir = model_output_dir / model_name
            cmd.extend(["--output-dir", str(model_output_dir)])

        logger.info("Running fine-tuning command for '%s': %s", model_name, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Fine-tuning failed for model '{model_name}' with exit code {e.returncode}"
            )
        finally:
            try:
                os.remove(temp_pretrain_cfg_path)
            except OSError:
                logger.warning(f"Could not remove temporary config: {temp_pretrain_cfg_path}")


if __name__ == "__main__":
    cli(obj={})
