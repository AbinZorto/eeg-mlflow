from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Sequence, Tuple


ObjectKind = Literal["experiment_meta", "run", "dataset"]
SyncDirection = Literal["both", "push", "pull"]
TEMP_PREFIX = ".sync_tmp_"
EXPERIMENT_ID_PLACEHOLDER = "'__EXPERIMENT_ID__'"
ARTIFACT_URI_PLACEHOLDER = "__ARTIFACT_URI__"
ARTIFACT_LOCATION_PLACEHOLDER = "__ARTIFACT_LOCATION__"
TIMESTAMP_PLACEHOLDER = "__TIMESTAMP__"
HOME_PATH_PATTERN = re.compile(r"/(?:home|Users)/[^/\s\"']+")


REMOTE_INVENTORY_SCRIPT = r"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

TEMP_PREFIX = ".sync_tmp_"
EXPERIMENT_ID_PLACEHOLDER = "'__EXPERIMENT_ID__'"
ARTIFACT_URI_PLACEHOLDER = "__ARTIFACT_URI__"
ARTIFACT_LOCATION_PLACEHOLDER = "__ARTIFACT_LOCATION__"
TIMESTAMP_PLACEHOLDER = "__TIMESTAMP__"
HOME_PATH_PATTERN = re.compile(r"/(?:home|Users)/[^/\s\"']+")


def replace_field(text, field, value):
    lines = text.splitlines()
    replaced = False
    output = []
    for line in lines:
        if line.startswith(f"{field}:"):
            output.append(f"{field}: {value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{field}: {value}")
    return "\n".join(output) + "\n"


def normalize_experiment_meta(text):
    text = replace_field(text, "experiment_id", EXPERIMENT_ID_PLACEHOLDER)
    text = replace_field(text, "artifact_location", ARTIFACT_LOCATION_PLACEHOLDER)
    text = replace_field(text, "creation_time", TIMESTAMP_PLACEHOLDER)
    text = replace_field(text, "last_update_time", TIMESTAMP_PLACEHOLDER)
    return HOME_PATH_PATTERN.sub("__HOME__", text)


def normalize_run_meta(text):
    text = replace_field(text, "experiment_id", EXPERIMENT_ID_PLACEHOLDER)
    text = replace_field(text, "artifact_uri", ARTIFACT_URI_PLACEHOLDER)
    return HOME_PATH_PATTERN.sub("__HOME__", text)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_tree(root, normalize_run):
    hasher = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None

        if normalize_run and rel == "meta.yaml" and text is not None:
            data = normalize_run_meta(text).encode("utf-8")
        elif text is not None:
            data = HOME_PATH_PATTERN.sub("__HOME__", text).encode("utf-8")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hash_bytes(data).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def inventory_experiment(experiment_path):
    experiment_root = Path(experiment_path)
    excluded_runs = set(json.loads(sys.argv[2])) if len(sys.argv) > 2 else set()
    result = {"exists": experiment_root.exists(), "objects": []}
    if not experiment_root.exists():
        return result

    experiment_meta = experiment_root / "meta.yaml"
    if experiment_meta.exists():
        normalized = normalize_experiment_meta(experiment_meta.read_text())
        result["objects"].append(
            {
                "kind": "experiment_meta",
                "identifier": "meta",
                "relative_path": "meta.yaml",
                "digest": hash_bytes(normalized.encode("utf-8")),
            }
        )

    for child in sorted(experiment_root.iterdir()):
        if child.name in {"meta.yaml", "datasets"} or child.name.startswith(TEMP_PREFIX):
            continue
        if child.name in excluded_runs:
            continue
        if child.is_dir() and (child / "meta.yaml").exists():
            result["objects"].append(
                {
                    "kind": "run",
                    "identifier": child.name,
                    "relative_path": child.name,
                    "digest": hash_tree(child, normalize_run=True),
                }
            )

    datasets_root = experiment_root / "datasets"
    if datasets_root.exists():
        for child in sorted(datasets_root.iterdir()):
            if child.name.startswith(TEMP_PREFIX):
                continue
            if child.is_dir():
                result["objects"].append(
                    {
                        "kind": "dataset",
                        "identifier": child.name,
                        "relative_path": f"datasets/{child.name}",
                        "digest": hash_tree(child, normalize_run=False),
                    }
                )

    return result


payload = inventory_experiment(sys.argv[1])
print(json.dumps(payload))
"""


REMOTE_RESOLVE_PATH_SCRIPT = r"""
import os
import sys
print(os.path.realpath(os.path.expanduser(sys.argv[1])))
"""


REMOTE_HOME_SCRIPT = r"""
import os
print(os.path.expanduser("~"))
"""


REMOTE_ENSURE_DIR_SCRIPT = r"""
from pathlib import Path
import sys
Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
"""


REMOTE_REMOVE_PATH_SCRIPT = r"""
from pathlib import Path
import shutil
import sys

target = Path(sys.argv[1])
if target.is_dir():
    shutil.rmtree(target)
elif target.exists():
    target.unlink()
"""


REMOTE_RENAME_PATH_SCRIPT = r"""
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
if destination.exists():
    raise SystemExit(f"Destination already exists: {destination}")
destination.parent.mkdir(parents=True, exist_ok=True)
source.rename(destination)
"""


REMOTE_REWRITE_META_SCRIPT = r"""
from pathlib import Path
import sys


def replace_field(text, field, value):
    lines = text.splitlines()
    replaced = False
    output = []
    for line in lines:
        if line.startswith(f"{field}:"):
            output.append(f"{field}: {value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{field}: {value}")
    return "\n".join(output) + "\n"


kind, meta_path, experiment_id, artifact_path = sys.argv[1:5]
text = Path(meta_path).read_text()
text = replace_field(text, "experiment_id", repr(experiment_id))
field = "artifact_location" if kind == "experiment_meta" else "artifact_uri"
text = replace_field(text, field, artifact_path)
Path(meta_path).write_text(text)
"""


REMOTE_REWRITE_TREE_PATHS_SCRIPT = r"""
from pathlib import Path
import sys


def rewrite_file(path, source_prefix, destination_prefix):
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return

    rewritten = text.replace(source_prefix, destination_prefix)
    if rewritten != text:
        path.write_text(rewritten)


root = Path(sys.argv[1])
source_prefix = sys.argv[2]
destination_prefix = sys.argv[3]

if root.is_file():
    rewrite_file(root, source_prefix, destination_prefix)
else:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rewrite_file(path, source_prefix, destination_prefix)
"""


class SyncError(RuntimeError):
    """Raised when MLflow sync cannot complete safely."""


@dataclass(frozen=True)
class SyncObject:
    kind: ObjectKind
    identifier: str
    relative_path: str
    digest: str


@dataclass(frozen=True)
class SyncConflict:
    kind: ObjectKind
    identifier: str
    local_digest: str
    remote_digest: str


@dataclass
class ExperimentInventory:
    root: str
    experiment_id: str
    exists: bool
    objects: Dict[Tuple[ObjectKind, str], SyncObject] = field(default_factory=dict)


@dataclass
class SyncPlan:
    push: list[SyncObject] = field(default_factory=list)
    pull: list[SyncObject] = field(default_factory=list)
    skip: list[SyncObject] = field(default_factory=list)
    conflicts: list[SyncConflict] = field(default_factory=list)


@dataclass
class SyncResult:
    local_root: Path
    remote_root: str
    local_experiment_id: str
    remote_experiment_id: str
    remote_host: str
    plan: SyncPlan
    executed: bool


def replace_meta_field(text: str, field: str, value: str) -> str:
    lines = text.splitlines()
    replaced = False
    output = []
    for line in lines:
        if line.startswith(f"{field}:"):
            output.append(f"{field}: {value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{field}: {value}")
    return "\n".join(output) + "\n"


def normalize_experiment_meta_text(text: str) -> str:
    text = replace_meta_field(text, "experiment_id", EXPERIMENT_ID_PLACEHOLDER)
    text = replace_meta_field(text, "artifact_location", ARTIFACT_LOCATION_PLACEHOLDER)
    text = replace_meta_field(text, "creation_time", TIMESTAMP_PLACEHOLDER)
    text = replace_meta_field(text, "last_update_time", TIMESTAMP_PLACEHOLDER)
    return HOME_PATH_PATTERN.sub("__HOME__", text)


def normalize_run_meta_text(text: str) -> str:
    text = replace_meta_field(text, "experiment_id", EXPERIMENT_ID_PLACEHOLDER)
    text = replace_meta_field(text, "artifact_uri", ARTIFACT_URI_PLACEHOLDER)
    return HOME_PATH_PATTERN.sub("__HOME__", text)


def rewrite_experiment_meta_text(text: str, experiment_id: str, artifact_location: str) -> str:
    text = replace_meta_field(text, "experiment_id", repr(str(experiment_id)))
    return replace_meta_field(text, "artifact_location", artifact_location)


def rewrite_run_meta_text(text: str, experiment_id: str, artifact_uri: str) -> str:
    text = replace_meta_field(text, "experiment_id", repr(str(experiment_id)))
    return replace_meta_field(text, "artifact_uri", artifact_uri)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_object_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def hash_directory_tree(root: Path, *, normalize_run_meta: bool) -> str:
    hasher = hashlib.sha256()
    for path in _iter_object_files(root):
        rel_path = path.relative_to(root).as_posix()
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None

        if normalize_run_meta and rel_path == "meta.yaml" and text is not None:
            data = normalize_run_meta_text(text).encode("utf-8")
        elif text is not None:
            data = HOME_PATH_PATTERN.sub("__HOME__", text).encode("utf-8")
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hash_bytes(data).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _sync_key(kind: ObjectKind, identifier: str) -> Tuple[ObjectKind, str]:
    return kind, identifier


def object_sort_key(obj: SyncObject) -> tuple[int, str]:
    order = {"experiment_meta": 0, "run": 1, "dataset": 2}
    return order[obj.kind], obj.identifier


def format_sync_object(obj: SyncObject) -> str:
    if obj.kind == "experiment_meta":
        return "experiment meta"
    if obj.kind == "run":
        return f"run {obj.identifier}"
    return f"dataset {obj.identifier}"


def collect_local_inventory(
    root: Path,
    experiment_id: str,
    exclude_run_ids: Sequence[str] | None = None,
) -> ExperimentInventory:
    experiment_root = root / experiment_id
    excluded_runs = set(exclude_run_ids or [])
    objects: Dict[Tuple[ObjectKind, str], SyncObject] = {}
    if not experiment_root.exists():
        return ExperimentInventory(root=str(experiment_root), experiment_id=experiment_id, exists=False, objects=objects)

    experiment_meta = experiment_root / "meta.yaml"
    if experiment_meta.exists():
        digest = hash_bytes(normalize_experiment_meta_text(experiment_meta.read_text()).encode("utf-8"))
        obj = SyncObject(
            kind="experiment_meta",
            identifier="meta",
            relative_path="meta.yaml",
            digest=digest,
        )
        objects[_sync_key(obj.kind, obj.identifier)] = obj

    for child in sorted(experiment_root.iterdir()):
        if child.name in {"meta.yaml", "datasets"} or child.name.startswith(TEMP_PREFIX):
            continue
        if child.name in excluded_runs:
            continue
        if child.is_dir() and (child / "meta.yaml").exists():
            obj = SyncObject(
                kind="run",
                identifier=child.name,
                relative_path=child.name,
                digest=hash_directory_tree(child, normalize_run_meta=True),
            )
            objects[_sync_key(obj.kind, obj.identifier)] = obj

    datasets_root = experiment_root / "datasets"
    if datasets_root.exists():
        for child in sorted(datasets_root.iterdir()):
            if child.name.startswith(TEMP_PREFIX):
                continue
            if child.is_dir():
                obj = SyncObject(
                    kind="dataset",
                    identifier=child.name,
                    relative_path=f"datasets/{child.name}",
                    digest=hash_directory_tree(child, normalize_run_meta=False),
                )
                objects[_sync_key(obj.kind, obj.identifier)] = obj

    return ExperimentInventory(root=str(experiment_root), experiment_id=experiment_id, exists=True, objects=objects)


def build_sync_plan(
    local_inventory: ExperimentInventory,
    remote_inventory: ExperimentInventory,
    direction: SyncDirection,
    exclude_run_ids: Sequence[str] | None = None,
) -> SyncPlan:
    plan = SyncPlan()
    excluded_runs = set(exclude_run_ids or [])
    keys = sorted(set(local_inventory.objects) | set(remote_inventory.objects))
    for key in keys:
        local_obj = local_inventory.objects.get(key)
        remote_obj = remote_inventory.objects.get(key)

        exemplar = local_obj or remote_obj
        if exemplar and exemplar.kind == "run" and exemplar.identifier in excluded_runs:
            continue

        if local_obj and remote_obj:
            if local_obj.digest == remote_obj.digest:
                plan.skip.append(local_obj)
            else:
                plan.conflicts.append(
                    SyncConflict(
                        kind=local_obj.kind,
                        identifier=local_obj.identifier,
                        local_digest=local_obj.digest,
                        remote_digest=remote_obj.digest,
                    )
                )
            continue

        if local_obj:
            if direction in {"both", "push"}:
                plan.push.append(local_obj)
            else:
                plan.skip.append(local_obj)
            continue

        if remote_obj:
            if direction in {"both", "pull"}:
                plan.pull.append(remote_obj)
            else:
                plan.skip.append(remote_obj)

    plan.push.sort(key=object_sort_key)
    plan.pull.sort(key=object_sort_key)
    plan.skip.sort(key=object_sort_key)
    plan.conflicts.sort(key=lambda conflict: object_sort_key(SyncObject(conflict.kind, conflict.identifier, "", "")))
    return plan


def run_command(command: Sequence[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        list(command),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise SyncError(f"{' '.join(command)}: {message}")
    return completed.stdout.strip()


def run_remote_python(remote_host: str, script: str, *args: str) -> str:
    remote_command = "python3 -"
    if args:
        remote_command = f"{remote_command} {' '.join(shlex.quote(arg) for arg in args)}"
    command = ["ssh", "-o", "ClearAllForwardings=yes", remote_host, remote_command]
    return run_command(command, input_text=script)


def resolve_remote_home(remote_host: str) -> str:
    return run_remote_python(remote_host, REMOTE_HOME_SCRIPT)


def resolve_remote_root(remote_host: str, remote_root: str) -> str:
    resolve_remote_home(remote_host)
    return run_remote_python(remote_host, REMOTE_RESOLVE_PATH_SCRIPT, remote_root)


def collect_remote_inventory(
    remote_host: str,
    root: str,
    experiment_id: str,
    exclude_run_ids: Sequence[str] | None = None,
) -> ExperimentInventory:
    experiment_root = posixpath.join(root, experiment_id)
    payload = run_remote_python(remote_host, REMOTE_INVENTORY_SCRIPT, experiment_root, json.dumps(list(exclude_run_ids or [])))
    parsed = json.loads(payload)
    objects = {
        _sync_key(item["kind"], item["identifier"]): SyncObject(
            kind=item["kind"],
            identifier=item["identifier"],
            relative_path=item["relative_path"],
            digest=item["digest"],
        )
        for item in parsed["objects"]
    }
    return ExperimentInventory(
        root=experiment_root,
        experiment_id=experiment_id,
        exists=bool(parsed["exists"]),
        objects=objects,
    )


def ensure_local_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_remote_dir(remote_host: str, path: str) -> None:
    run_remote_python(remote_host, REMOTE_ENSURE_DIR_SCRIPT, path)


def remove_local_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def remove_remote_path(remote_host: str, path: str) -> None:
    run_remote_python(remote_host, REMOTE_REMOVE_PATH_SCRIPT, path)


def rename_local_path(source: Path, destination: Path) -> None:
    if destination.exists():
        raise SyncError(f"Destination already exists: {destination}")
    ensure_local_dir(destination.parent)
    source.rename(destination)


def rename_remote_path(remote_host: str, source: str, destination: str) -> None:
    run_remote_python(remote_host, REMOTE_RENAME_PATH_SCRIPT, source, destination)


def rewrite_local_metadata(path: Path, kind: ObjectKind, experiment_id: str, artifact_path: str) -> None:
    text = path.read_text()
    if kind == "experiment_meta":
        text = rewrite_experiment_meta_text(text, experiment_id, artifact_path)
    else:
        text = rewrite_run_meta_text(text, experiment_id, artifact_path)
    path.write_text(text)


def rewrite_local_tree_paths(path: Path, source_prefix: str, destination_prefix: str) -> None:
    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())

    for file_path in candidates:
        try:
            text = file_path.read_text()
        except UnicodeDecodeError:
            continue

        rewritten = text.replace(source_prefix, destination_prefix)
        if rewritten != text:
            file_path.write_text(rewritten)


def rewrite_remote_metadata(
    remote_host: str,
    path: str,
    kind: ObjectKind,
    experiment_id: str,
    artifact_path: str,
) -> None:
    meta_kind = "experiment_meta" if kind == "experiment_meta" else "run"
    run_remote_python(
        remote_host,
        REMOTE_REWRITE_META_SCRIPT,
        meta_kind,
        path,
        experiment_id,
        artifact_path,
    )


def rewrite_remote_tree_paths(
    remote_host: str,
    path: str,
    source_prefix: str,
    destination_prefix: str,
) -> None:
    run_remote_python(
        remote_host,
        REMOTE_REWRITE_TREE_PATHS_SCRIPT,
        path,
        source_prefix,
        destination_prefix,
    )


def make_local_temp_path(destination: Path) -> Path:
    return destination.parent / f"{TEMP_PREFIX}{destination.name}_{uuid.uuid4().hex}"


def make_remote_temp_path(destination: str) -> str:
    parent = posixpath.dirname(destination)
    name = posixpath.basename(destination)
    return posixpath.join(parent, f"{TEMP_PREFIX}{name}_{uuid.uuid4().hex}")


def rsync_to_remote(local_source: Path, remote_host: str, remote_destination: str, *, source_is_dir: bool) -> None:
    source_arg = f"{local_source}/" if source_is_dir else str(local_source)
    destination_arg = f"{remote_host}:{remote_destination}/" if source_is_dir else f"{remote_host}:{remote_destination}"
    run_command(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o ClearAllForwardings=yes",
            source_arg,
            destination_arg,
        ]
    )


def rsync_from_remote(remote_host: str, remote_source: str, local_destination: Path, *, source_is_dir: bool) -> None:
    source_arg = f"{remote_host}:{remote_source}/" if source_is_dir else f"{remote_host}:{remote_source}"
    destination_arg = f"{local_destination}/" if source_is_dir else str(local_destination)
    run_command(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o ClearAllForwardings=yes",
            source_arg,
            destination_arg,
        ]
    )


def local_object_source_path(local_experiment_root: Path, obj: SyncObject) -> Path:
    return local_experiment_root / obj.relative_path


def remote_object_source_path(remote_experiment_root: str, obj: SyncObject) -> str:
    if obj.kind == "experiment_meta":
        return posixpath.join(remote_experiment_root, "meta.yaml")
    if obj.kind == "run":
        return posixpath.join(remote_experiment_root, obj.identifier)
    return posixpath.join(remote_experiment_root, "datasets", obj.identifier)


def sync_object_push(
    obj: SyncObject,
    *,
    local_experiment_root: Path,
    remote_host: str,
    remote_experiment_root: str,
    remote_experiment_id: str,
    local_home: str,
    remote_home: str,
) -> None:
    source = local_object_source_path(local_experiment_root, obj)
    destination = remote_object_source_path(remote_experiment_root, obj)
    ensure_remote_dir(remote_host, posixpath.dirname(destination))
    temp_destination = make_remote_temp_path(destination)

    try:
        if obj.kind == "experiment_meta":
            rsync_to_remote(source, remote_host, temp_destination, source_is_dir=False)
        else:
            ensure_remote_dir(remote_host, temp_destination)
            rsync_to_remote(source, remote_host, temp_destination, source_is_dir=True)
        rewrite_remote_tree_paths(remote_host, temp_destination, local_home, remote_home)
        if obj.kind == "experiment_meta":
            rewrite_remote_metadata(
                remote_host,
                temp_destination,
                obj.kind,
                remote_experiment_id,
                remote_experiment_root,
            )
        elif obj.kind == "run":
            rewrite_remote_metadata(
                remote_host,
                posixpath.join(temp_destination, "meta.yaml"),
                obj.kind,
                remote_experiment_id,
                posixpath.join(remote_experiment_root, obj.identifier, "artifacts"),
            )
        rename_remote_path(remote_host, temp_destination, destination)
    except Exception:
        try:
            remove_remote_path(remote_host, temp_destination)
        except SyncError:
            pass
        raise


def sync_object_pull(
    obj: SyncObject,
    *,
    local_experiment_root: Path,
    local_experiment_id: str,
    remote_host: str,
    remote_experiment_root: str,
    local_home: str,
    remote_home: str,
) -> None:
    source = remote_object_source_path(remote_experiment_root, obj)
    destination = local_object_source_path(local_experiment_root, obj)
    ensure_local_dir(destination.parent)
    temp_destination = make_local_temp_path(destination)

    try:
        if obj.kind == "experiment_meta":
            rsync_from_remote(remote_host, source, temp_destination, source_is_dir=False)
        else:
            ensure_local_dir(temp_destination)
            rsync_from_remote(remote_host, source, temp_destination, source_is_dir=True)
        rewrite_local_tree_paths(temp_destination, remote_home, local_home)
        if obj.kind == "experiment_meta":
            rewrite_local_metadata(temp_destination, obj.kind, local_experiment_id, str(local_experiment_root))
        elif obj.kind == "run":
            rewrite_local_metadata(
                temp_destination / "meta.yaml",
                obj.kind,
                local_experiment_id,
                str(local_experiment_root / obj.identifier / "artifacts"),
            )
        rename_local_path(temp_destination, destination)
    except Exception:
        remove_local_path(temp_destination)
        raise


def sync_experiment(
    *,
    local_root: str,
    remote_root: str,
    local_experiment_id: str,
    remote_experiment_id: str,
    remote_host: str,
    direction: SyncDirection = "both",
    dry_run: bool = False,
    exclude_run_ids: Sequence[str] | None = None,
) -> SyncResult:
    local_root_path = Path(local_root).expanduser().resolve()
    local_home = str(Path.home())
    remote_home = resolve_remote_home(remote_host)
    resolved_remote_root = run_remote_python(remote_host, REMOTE_RESOLVE_PATH_SCRIPT, remote_root)

    local_inventory = collect_local_inventory(local_root_path, local_experiment_id, exclude_run_ids=exclude_run_ids)
    remote_inventory = collect_remote_inventory(
        remote_host,
        resolved_remote_root,
        remote_experiment_id,
        exclude_run_ids=exclude_run_ids,
    )
    plan = build_sync_plan(local_inventory, remote_inventory, direction, exclude_run_ids=exclude_run_ids)

    result = SyncResult(
        local_root=local_root_path,
        remote_root=resolved_remote_root,
        local_experiment_id=local_experiment_id,
        remote_experiment_id=remote_experiment_id,
        remote_host=remote_host,
        plan=plan,
        executed=False,
    )

    if plan.conflicts or dry_run:
        return result

    local_experiment_root = local_root_path / local_experiment_id
    remote_experiment_root = posixpath.join(resolved_remote_root, remote_experiment_id)

    for obj in plan.push:
        sync_object_push(
            obj,
            local_experiment_root=local_experiment_root,
            remote_host=remote_host,
            remote_experiment_root=remote_experiment_root,
            remote_experiment_id=remote_experiment_id,
            local_home=local_home,
            remote_home=remote_home,
        )

    for obj in plan.pull:
        sync_object_pull(
            obj,
            local_experiment_root=local_experiment_root,
            local_experiment_id=local_experiment_id,
            remote_host=remote_host,
            remote_experiment_root=remote_experiment_root,
            local_home=local_home,
            remote_home=remote_home,
        )

    result.executed = True
    return result
