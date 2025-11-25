import click
import mlflow
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.processing.secondary import pipeline as secondary_pipeline

logger = setup_logger(__name__)


@click.group()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to secondary processing config file')
@click.pass_context
def cli(ctx, config):
    """Secondary EEG dataset builder CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['config'] = load_config(config)


@cli.command(name="build-secondary")
@click.pass_context
def build_secondary(ctx):
    """Build and register the secondary EEG dataset for Mamba2 pretraining."""
    config = ctx.obj['config']

    # Set up MLflow tracking (pipeline will set experiment and start run)
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    result = secondary_pipeline.run(config)

    # Print summary
    print(f"\n✅ Secondary dataset built successfully")
    print(f"   Output dir: {result.get('output_dir')}")
    print(f"   Subjects: {result.get('num_subjects')}, Runs discovered: {result.get('num_runs')}, Runs saved: {result.get('runs_saved')}")
    print(f"   Sampling rate: {result.get('sampling_rate')} Hz")
    print()


if __name__ == '__main__':
    cli(obj={})


