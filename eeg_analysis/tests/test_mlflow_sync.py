from pathlib import Path

from src.utils.mlflow_sync import (
    build_sync_plan,
    collect_local_inventory,
    rewrite_experiment_meta_text,
    rewrite_run_meta_text,
)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def make_experiment_meta(experiment_id: str, artifact_location: str, name: str = "eeg_processing") -> str:
    return (
        f"artifact_location: {artifact_location}\n"
        "creation_time: 1774268821387\n"
        f"experiment_id: '{experiment_id}'\n"
        "last_update_time: 1774268821387\n"
        "lifecycle_stage: active\n"
        f"name: {name}\n"
    )


def make_run_meta(experiment_id: str, artifact_uri: str, run_id: str) -> str:
    return (
        f"artifact_uri: {artifact_uri}\n"
        "end_time: 1774271258235\n"
        "entry_point_name: ''\n"
        f"experiment_id: '{experiment_id}'\n"
        "lifecycle_stage: active\n"
        f"run_id: {run_id}\n"
        "run_name: processing\n"
        "source_name: ''\n"
        "source_type: 4\n"
        "source_version: ''\n"
        "start_time: 1774269934350\n"
        "status: 3\n"
        "tags: []\n"
        "user_id: abin\n"
    )


def create_run(experiment_root: Path, experiment_id: str, run_id: str, metric_value: str = "1") -> None:
    run_root = experiment_root / run_id
    artifact_uri = f"{experiment_root}/{run_id}/artifacts"
    write_text(run_root / "meta.yaml", make_run_meta(experiment_id, artifact_uri, run_id))
    write_text(run_root / "metrics" / "processing_success", metric_value)
    write_text(run_root / "params" / "window_size", "10")


def create_dataset(experiment_root: Path, dataset_id: str, digest: str = "abc123") -> None:
    dataset_root = experiment_root / "datasets" / dataset_id
    write_text(dataset_root / "meta.yaml", f"digest: {digest}\nname: dataset_{dataset_id}\n")


def create_experiment(root: Path, experiment_id: str, artifact_location: str) -> Path:
    experiment_root = root / experiment_id
    write_text(experiment_root / "meta.yaml", make_experiment_meta(experiment_id, artifact_location))
    return experiment_root


def test_run_digest_ignores_experiment_id_and_artifact_uri(tmp_path):
    local_root = tmp_path / "local" / "mlruns"
    remote_root = tmp_path / "remote" / "mlruns"

    local_experiment = create_experiment(local_root, "134978379211635499", "/Users/abin/eeg-mlflow/mlruns/134978379211635499")
    remote_experiment = create_experiment(remote_root, "714557200892293258", "/home/abin/eeg-mlflow/mlruns/714557200892293258")

    create_run(local_experiment, "134978379211635499", "18efe04bc6a2488fac966a342fe5aa1c")
    create_run(remote_experiment, "714557200892293258", "18efe04bc6a2488fac966a342fe5aa1c")

    local_inventory = collect_local_inventory(local_root, "134978379211635499")
    remote_inventory = collect_local_inventory(remote_root, "714557200892293258")

    local_run = local_inventory.objects[("run", "18efe04bc6a2488fac966a342fe5aa1c")]
    remote_run = remote_inventory.objects[("run", "18efe04bc6a2488fac966a342fe5aa1c")]
    assert local_run.digest == remote_run.digest


def test_experiment_meta_digest_ignores_experiment_id_and_artifact_location(tmp_path):
    local_root = tmp_path / "local" / "mlruns"
    remote_root = tmp_path / "remote" / "mlruns"

    create_experiment(local_root, "134978379211635499", "/Users/abin/eeg-mlflow/mlruns/134978379211635499")
    remote_experiment = remote_root / "714557200892293258"
    write_text(
        remote_experiment / "meta.yaml",
        (
            "artifact_location: /home/abin/eeg-mlflow/mlruns/714557200892293258\n"
            "creation_time: 1888888888888\n"
            "experiment_id: '714557200892293258'\n"
            "last_update_time: 1999999999999\n"
            "lifecycle_stage: active\n"
            "name: eeg_processing\n"
        ),
    )

    local_inventory = collect_local_inventory(local_root, "134978379211635499")
    remote_inventory = collect_local_inventory(remote_root, "714557200892293258")

    local_meta = local_inventory.objects[("experiment_meta", "meta")]
    remote_meta = remote_inventory.objects[("experiment_meta", "meta")]
    assert local_meta.digest == remote_meta.digest


def test_build_sync_plan_detects_conflict_for_changed_run_content(tmp_path):
    local_root = tmp_path / "local" / "mlruns"
    remote_root = tmp_path / "remote" / "mlruns"

    local_experiment = create_experiment(local_root, "134978379211635499", "/Users/abin/eeg-mlflow/mlruns/134978379211635499")
    remote_experiment = create_experiment(remote_root, "714557200892293258", "/home/abin/eeg-mlflow/mlruns/714557200892293258")

    create_run(local_experiment, "134978379211635499", "shared_run", metric_value="1")
    create_run(remote_experiment, "714557200892293258", "shared_run", metric_value="0")

    plan = build_sync_plan(
        collect_local_inventory(local_root, "134978379211635499"),
        collect_local_inventory(remote_root, "714557200892293258"),
        "both",
    )

    assert len(plan.conflicts) == 1
    assert plan.conflicts[0].identifier == "shared_run"


def test_build_sync_plan_splits_push_pull_and_skip(tmp_path):
    local_root = tmp_path / "local" / "mlruns"
    remote_root = tmp_path / "remote" / "mlruns"

    local_experiment = create_experiment(local_root, "134978379211635499", "/Users/abin/eeg-mlflow/mlruns/134978379211635499")
    remote_experiment = create_experiment(remote_root, "714557200892293258", "/home/abin/eeg-mlflow/mlruns/714557200892293258")

    create_run(local_experiment, "134978379211635499", "local_run")
    create_run(remote_experiment, "714557200892293258", "remote_run")
    create_dataset(local_experiment, "local_dataset")
    create_dataset(remote_experiment, "remote_dataset")
    write_text(remote_experiment / "tags" / "owner", "abin")

    plan = build_sync_plan(
        collect_local_inventory(local_root, "134978379211635499"),
        collect_local_inventory(remote_root, "714557200892293258"),
        "both",
    )

    assert [obj.identifier for obj in plan.push] == ["local_run", "local_dataset"]
    assert [obj.identifier for obj in plan.pull] == ["remote_run", "remote_dataset"]
    assert [obj.identifier for obj in plan.skip] == ["meta"]


def test_rewrite_run_meta_text_uses_destination_values():
    original = make_run_meta("old-exp", "/tmp/source/artifacts", "run123")
    rewritten = rewrite_run_meta_text(original, "new-exp", "/Users/abin/eeg-mlflow/mlruns/new-exp/run123/artifacts")

    assert "experiment_id: 'new-exp'" in rewritten
    assert "artifact_uri: /Users/abin/eeg-mlflow/mlruns/new-exp/run123/artifacts" in rewritten
    assert "experiment_id: 'old-exp'" not in rewritten


def test_rewrite_experiment_meta_text_uses_destination_values():
    original = make_experiment_meta("old-exp", "/tmp/source/old-exp")
    rewritten = rewrite_experiment_meta_text(original, "new-exp", "/home/abin/eeg-mlflow/mlruns/new-exp")

    assert "experiment_id: 'new-exp'" in rewritten
    assert "artifact_location: /home/abin/eeg-mlflow/mlruns/new-exp" in rewritten
    assert "experiment_id: 'old-exp'" not in rewritten


def test_build_sync_plan_excludes_run_ids(tmp_path):
    local_root = tmp_path / "local" / "mlruns"
    remote_root = tmp_path / "remote" / "mlruns"

    local_experiment = create_experiment(local_root, "134978379211635499", "/Users/abin/eeg-mlflow/mlruns/134978379211635499")
    remote_experiment = create_experiment(remote_root, "714557200892293258", "/home/abin/eeg-mlflow/mlruns/714557200892293258")

    create_run(local_experiment, "134978379211635499", "keep_local")
    create_run(remote_experiment, "714557200892293258", "exclude_me")

    plan = build_sync_plan(
        collect_local_inventory(local_root, "134978379211635499"),
        collect_local_inventory(remote_root, "714557200892293258"),
        "both",
        exclude_run_ids=["exclude_me"],
    )

    assert [obj.identifier for obj in plan.push] == ["keep_local"]
    assert [obj.identifier for obj in plan.pull] == []
