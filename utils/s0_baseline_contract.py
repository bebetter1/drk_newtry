"""Pure helpers for the S0 baseline evidence contract.

This module deliberately avoids importing torch or the renderer at import time so
that its contract tests can run on CPU-only machines. GPU/runtime collection is
performed by the command-line scripts under ``scripts/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PLAN_ID = "S0_BASELINE_V1"
METRIC_NAMES = ("l1", "psnr", "ssim", "lpips")
_RUN_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}_[A-Za-z0-9][A-Za-z0-9_.-]*$")
CRITICAL_SOURCE_PATHS = (
    "train.py",
    "arguments/__init__.py",
    "gaussian_renderer/__init__.py",
    "scene/gaussian_model.py",
    "scene/cameras.py",
    "scene/dataset_readers.py",
    "utils/metric.py",
    "submodules/drk_splatting/setup.py",
    "submodules/drk_splatting/cuda_rasterizer/config.h",
    "submodules/drk_splatting/cuda_rasterizer/forward.cu",
    "submodules/drk_splatting/cuda_rasterizer/backward.cu",
)
S0_ALLOWED_TOOL_PATHS = (
    "scripts/s0_baseline_contract.py",
    "scripts/run_s0_baseline_parity.py",
    "utils/s0_baseline_contract.py",
    "tests/test_s0_baseline_contract.py",
    "tests/test_s0_baseline_cli.py",
    "tests/test_s0_baseline_parity.py",
)


class ContractError(ValueError):
    """Raised when evidence would violate the frozen Stage 0 contract."""


def validate_run_id(value: str) -> str:
    """Validate a non-path run identifier suitable for an exclusive directory."""

    if not isinstance(value, str) or not _RUN_ID_PATTERN.fullmatch(value):
        raise ContractError(
            "run_id must match YYYYMMDD-HHMMSS_<label> and contain no path separators"
        )
    return value


def create_evidence_run(evidence_root: Path, run_id: str) -> Path:
    """Create the immutable Stage 0 directory skeleton."""

    run_id = validate_run_id(run_id)
    run_dir = Path(evidence_root).expanduser().resolve() / PLAN_ID / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractError("evidence run already exists: {}".format(run_dir)) from error
    (run_dir / "logs").mkdir()
    (run_dir / "artifacts").mkdir()
    return run_dir


def split_colmap_names(names: Iterable[str], llffhold: int = 8) -> Dict[str, List[str]]:
    """Reproduce the repository's sorted COLMAP ``eval`` split."""

    if not isinstance(llffhold, int) or llffhold < 2:
        raise ContractError("llffhold must be an integer >= 2")
    ordered = sorted(str(name) for name in names)
    if not ordered:
        raise ContractError("dataset image list is empty")
    if len(set(ordered)) != len(ordered):
        raise ContractError("duplicate image names are not allowed")
    return {
        "all": ordered,
        "train": [name for index, name in enumerate(ordered) if index % llffhold != 0],
        "test": [name for index, name in enumerate(ordered) if index % llffhold == 0],
        "val": [],
        "rule": "sort image_name lexicographically; test iff zero-based index % 8 == 0",
        "llffhold": llffhold,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_baseline_id(binding: Mapping[str, Any]) -> str:
    """Return the compact identifier for a complete, canonical binding."""

    return "S0-" + stable_json_hash(binding)[:16]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git(source_root: Path, *arguments: str) -> str:
    command = [
        "git",
        "-c",
        "safe.directory={}".format(source_root.as_posix()),
        "-C",
        str(source_root),
    ]
    command.extend(arguments)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ContractError(
            "git command failed ({}): {}".format(" ".join(arguments), completed.stderr.strip())
        )
    return completed.stdout


def collect_source_state(
    source_root: Path,
    expected_commit: str = "",
    critical_paths: Sequence[str] = CRITICAL_SOURCE_PATHS,
    allowed_paths: Sequence[str] = (),
) -> Dict[str, Any]:
    """Bind the local commit, dirty paths, and result-affecting file hashes."""

    source_root = Path(source_root).expanduser().resolve()
    execution_head = _git(source_root, "rev-parse", "HEAD").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", execution_head):
        raise ContractError("git HEAD is not a full commit hash: {}".format(execution_head))
    baseline_commit = expected_commit.lower() if expected_commit else execution_head
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_commit):
        raise ContractError("expected baseline commit is not a full hash: {}".format(baseline_commit))
    ancestor = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory={}".format(source_root.as_posix()),
            "-C",
            str(source_root),
            "merge-base",
            "--is-ancestor",
            baseline_commit,
            execution_head,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ContractError(
            "expected baseline commit {} is not an ancestor of execution HEAD {}".format(
                baseline_commit, execution_head
            )
        )

    raw_status = _git(source_root, "status", "--porcelain=v1", "--untracked-files=all", "-z")
    tokens = [token for token in raw_status.split("\0") if token]
    changed: List[Dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if len(token) < 4:
            raise ContractError("could not parse git status entry: {!r}".format(token))
        status = token[:2]
        relative = token[3:].replace("\\", "/")
        row: Dict[str, Any] = {"status": status, "path": relative}
        changed.append(row)
        if status[0] in {"R", "C"} and index + 1 < len(tokens):
            index += 1
            row["source_path"] = tokens[index].replace("\\", "/")
        index += 1

    critical_hashes: Dict[str, str] = {}
    for relative in critical_paths:
        path = source_root / relative
        if not path.is_file():
            raise ContractError("critical source file does not exist: {}".format(path))
        current_hash = sha256_file(path)
        critical_hashes[relative] = current_hash
        if expected_commit:
            baseline_object = _git(source_root, "rev-parse", "{}:{}".format(baseline_commit, relative)).strip()
            current_object = _git(source_root, "hash-object", "--path", relative, str(path)).strip()
            if current_object != baseline_object:
                raise ContractError(
                    "critical source file differs from baseline commit {}: {}".format(
                        baseline_commit, relative
                    )
                )

    committed_differences = {
        line.strip().replace("\\", "/")
        for line in _git(source_root, "diff", "--name-only", baseline_commit + ".." + execution_head).splitlines()
        if line.strip()
    }
    working_differences = {str(row["path"]) for row in changed}
    differences = sorted(committed_differences | working_differences)
    allowed = {str(path).replace("\\", "/") for path in allowed_paths}
    unauthorized = sorted(set(differences) - allowed)
    if allowed_paths and unauthorized:
        raise ContractError(
            "source differs from the frozen baseline outside authorized Stage 0 tools: {}".format(
                unauthorized
            )
        )
    return {
        "source_root": str(source_root),
        "commit": baseline_commit,
        "baseline_commit": baseline_commit,
        "execution_head": execution_head,
        "head_matches_baseline_commit": execution_head == baseline_commit,
        "dirty": bool(changed),
        "changed_files": changed,
        "diagnostic_files_since_baseline": differences,
        "authorized_diagnostic_files": sorted(allowed),
        "critical_file_sha256": critical_hashes,
    }


def write_json(path: Path, value: Any, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolved_camera_size(width: int, height: int, resolution: int = -1) -> Tuple[int, int]:
    """Mirror ``utils.camera_utils.loadCam`` at resolution_scale=1."""

    if width <= 0 or height <= 0:
        raise ContractError("image dimensions must be positive")
    if resolution in (1, 2, 4, 8):
        return round(width / resolution), round(height / resolution)
    if resolution == -1:
        global_down = width / 1600.0 if width > 1600 else 1.0
    elif resolution > 0:
        global_down = width / float(resolution)
    else:
        raise ContractError("resolution must be -1, a positive target width, or one of 1/2/4/8")
    return int(width / global_down), int(height / global_down)


def collect_dataset_manifest(
    dataset_root: Path, images_directory: str = "images", resolution: int = -1
) -> Dict[str, Any]:
    """Freeze the image inventory, resolution, COLMAP inputs, and eval split."""

    from PIL import Image

    dataset_root = Path(dataset_root).expanduser().resolve()
    image_root = dataset_root / images_directory
    sparse_root = dataset_root / "sparse" / "0"
    if not image_root.is_dir():
        raise ContractError("image directory does not exist: {}".format(image_root))
    if not sparse_root.is_dir():
        raise ContractError("COLMAP sparse/0 directory does not exist: {}".format(sparse_root))

    supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    image_paths = sorted(
        (path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in supported),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ContractError("no supported images found in {}".format(image_root))

    image_rows: List[Dict[str, Any]] = []
    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size
        resolved_width, resolved_height = resolved_camera_size(width, height, resolution)
        image_rows.append(
            {
                "file_name": path.name,
                "image_name": path.stem,
                "bytes": path.stat().st_size,
                "source_size": [width, height],
                "resolved_size": [resolved_width, resolved_height],
            }
        )

    sparse_rows: List[Dict[str, Any]] = []
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = sparse_root / name
        if not path.is_file():
            raise ContractError("required COLMAP file does not exist: {}".format(path))
        sparse_rows.append(
            {"relative_path": "sparse/0/" + name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    split = split_colmap_names((row["image_name"] for row in image_rows), llffhold=8)
    identity_payload = {
        "images_directory": images_directory,
        "resolution": resolution,
        "images": image_rows,
        "sparse": sparse_rows,
        "split": split,
    }
    return {
        "dataset_path": str(dataset_root),
        "images_directory": images_directory,
        "resolution": resolution,
        "image_count": len(image_rows),
        "images": image_rows,
        "sparse": sparse_rows,
        "split": split,
        "dataset_identity_sha256": stable_json_hash(identity_payload),
    }


def build_replay_commands(
    source_root: Path,
    dataset_root: Path,
    smoke_model_base: Path,
    full_model_base: Path,
    run_dir: Path,
    python_executable: str = "python",
    gpu_id: int = 0,
    images_directory: str = "images",
    resolution: int = -1,
    smoke_iterations: int = 10,
    iterations: int = 35000,
) -> str:
    """Build the exact, reviewable Bash sequence for Stage 0 on the server."""

    if smoke_iterations <= 0 or iterations <= smoke_iterations:
        raise ContractError("iterations must be greater than positive smoke_iterations")
    if gpu_id < 0:
        raise ContractError("gpu_id must be non-negative")

    def as_posix(value: Any) -> str:
        if hasattr(value, "as_posix"):
            return value.as_posix()
        return str(value).replace("\\", "/")

    quote = shlex.quote
    source_value = as_posix(source_root)
    dataset_value = as_posix(dataset_root)
    smoke_base_value = as_posix(smoke_model_base)
    full_base_value = as_posix(full_model_base)
    run_value = as_posix(run_dir)
    source = quote(source_value)
    dataset = quote(dataset_value)
    smoke_base = quote(smoke_base_value)
    if full_base_value.endswith("_DRK"):
        raise ContractError("full_model_base must not include train.py's automatic _DRK suffix")
    full_base = quote(full_base_value)
    full_actual_value = full_base_value + "_DRK"
    full_actual = quote(full_actual_value)
    run = quote(run_value)
    python = quote(str(python_executable))
    images = quote(images_directory)
    common = (
        "--eval --gs_type DRK --kernel_density dense --cache_sort --is_unbounded "
        "--images {} --resolution {}".format(images, int(resolution))
    )
    build_targets = (
        "depth-diff-gaussian-rasterization",
        "drk_splatting",
        "simple-knn",
    )
    lines = [
        "set -euo pipefail",
        "export CUDA_VISIBLE_DEVICES={}".format(gpu_id),
        "cd {}".format(source),
    ]
    for target in build_targets:
        log = quote(str(PurePosixPath(run_value) / "logs" / ("build_" + target + ".log")))
        lines.append(
            "(cd {} && {} setup.py install && {} -m pip install .) 2>&1 | tee {}".format(
                quote("submodules/" + target), python, python, log
            )
        )
    lines.extend(
        [
            "{} scripts/s0_baseline_contract.py capture-environment --run-dir {}".format(python, run),
            (
                "/usr/bin/time -v -o {smoke_time} {python} train.py -s {dataset} -m {smoke} "
                "{common} --iterations {smoke_iter} --test_iterations {smoke_iter} "
                "--save_iterations {smoke_iter} --checkpoint_iterations {smoke_iter} "
                "2>&1 | tee {smoke_log}"
            ).format(
                smoke_time=quote(str(PurePosixPath(run_value) / "logs" / "smoke_time.txt")),
                python=python,
                dataset=dataset,
                smoke=smoke_base,
                common=common,
                smoke_iter=smoke_iterations,
                smoke_log=quote(str(PurePosixPath(run_value) / "logs" / "smoke.log")),
            ),
            (
                "/usr/bin/time -v -o {train_time} {python} train.py -s {dataset} -m {full} "
                "{common} --iterations {iterations} --checkpoint_iterations {iterations} "
                "2>&1 | tee {train_log}"
            ).format(
                train_time=quote(str(PurePosixPath(run_value) / "logs" / "train_time.txt")),
                python=python,
                dataset=dataset,
                full=full_base,
                common=common,
                iterations=iterations,
                train_log=quote(str(PurePosixPath(run_value) / "logs" / "train.log")),
            ),
        ]
    )
    parity_base = (
        "{python} scripts/run_s0_baseline_parity.py --source-path {dataset} "
        "--model-path {model} --images {images} --resolution {resolution} "
        "--load-iteration {iterations} --gs-type DRK --kernel-density dense "
        "--cache-sort --is-unbounded --evidence-dir {run}"
    ).format(
        python=python,
        dataset=dataset,
        model=full_actual,
        images=images,
        resolution=int(resolution),
        iterations=iterations,
        run=run,
    )
    for repeat_index in (1, 2):
        lines.append(
            "{} --repeat-index {} 2>&1 | tee {}".format(
                parity_base,
                repeat_index,
                quote(str(PurePosixPath(run_value) / "logs" / "parity_repeat_{}.log".format(repeat_index))),
            )
        )
    lines.append(
        (
            "{python} scripts/s0_baseline_contract.py finalize --run-dir {run} "
            "--checkpoint {checkpoint} --train-stats {train_stats}"
        ).format(
            python=python,
            run=run,
            checkpoint=quote(full_actual_value + "/chkpnt{}.pth".format(iterations)),
            train_stats=quote(full_actual_value + "/train_stats.json"),
        )
    )
    return "\n".join(lines) + "\n"


def _index_views(repeat: Mapping[str, Any]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    views = repeat.get("views")
    if not isinstance(views, list) or not views:
        raise ContractError("each metric repeat must contain a non-empty views list")
    indexed: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in views:
        if not isinstance(row, Mapping):
            raise ContractError("metric view rows must be objects")
        key = (str(row.get("split", "")), str(row.get("image_name", "")))
        if not all(key):
            raise ContractError("each metric view requires split and image_name")
        if key in indexed:
            raise ContractError("duplicate metric view: {}/{}".format(*key))
        indexed[key] = row
    return indexed


def _finite_metric(row: Mapping[str, Any], metric: str, key: Tuple[str, str]) -> float:
    if metric not in row:
        raise ContractError("missing {} for {}/{}".format(metric, *key))
    value = float(row[metric])
    if not math.isfinite(value):
        raise ContractError("non-finite {} for {}/{}".format(metric, *key))
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ContractError("cannot aggregate an empty metric sequence")
    return float(sum(values) / len(values))


def summarize_metric_repeats(
    repeat_1: Mapping[str, Any], repeat_2: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pair two independent evaluations and deterministically aggregate noise."""

    first = _index_views(repeat_1)
    second = _index_views(repeat_2)
    if set(first) != set(second):
        missing_1 = sorted(set(second) - set(first))
        missing_2 = sorted(set(first) - set(second))
        raise ContractError(
            "metric repeat view sets differ; missing_from_repeat_1={}, "
            "missing_from_repeat_2={}".format(missing_1, missing_2)
        )

    per_view: List[Dict[str, Any]] = []
    split_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key in sorted(first):
        row_1 = first[key]
        row_2 = second[key]
        metrics_1 = {name: _finite_metric(row_1, name, key) for name in METRIC_NAMES}
        metrics_2 = {name: _finite_metric(row_2, name, key) for name in METRIC_NAMES}
        deltas = {name: abs(metrics_1[name] - metrics_2[name]) for name in METRIC_NAMES}
        paired = {
            "split": key[0],
            "image_name": key[1],
            "repeat_1": metrics_1,
            "repeat_2": metrics_2,
            "repeat_abs_delta": deltas,
        }
        per_view.append(paired)
        split_rows[key[0]].append(paired)

    all_deltas = {name: [row["repeat_abs_delta"][name] for row in per_view] for name in METRIC_NAMES}
    summary: Dict[str, Any] = {
        "plan_id": PLAN_ID,
        "repeatability": {
            "view_count": len(per_view),
            "max_abs_delta": {name: max(values) for name, values in all_deltas.items()},
            "mean_abs_delta": {name: _mean(values) for name, values in all_deltas.items()},
        },
        "splits": {},
    }
    for split, rows in sorted(split_rows.items()):
        split_summary: Dict[str, Any] = {"view_count": len(rows)}
        for repeat_key in ("repeat_1", "repeat_2"):
            split_summary[repeat_key] = {
                name: _mean([row[repeat_key][name] for row in rows]) for name in METRIC_NAMES
            }
        split_summary["abs_delta_of_means"] = {
            name: abs(split_summary["repeat_1"][name] - split_summary["repeat_2"][name])
            for name in METRIC_NAMES
        }
        split_summary["mean_of_repeats"] = {
            name: 0.5 * (split_summary["repeat_1"][name] + split_summary["repeat_2"][name])
            for name in METRIC_NAMES
        }
        summary["splits"][split] = split_summary
    return per_view, summary


def _required_json(run_dir: Path, name: str) -> Mapping[str, Any]:
    path = run_dir / name
    if not path.is_file():
        raise ContractError("required evidence file is missing: {}".format(path))
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ContractError("required evidence file must contain a JSON object: {}".format(path))
    return value


def finalize_evidence(
    run_dir: Path,
    checkpoint_path: Path,
    loaded_iteration: int,
    primitive_count: int,
    train_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind immutable Stage 0 evidence after two independent parity runs."""

    run_dir = Path(run_dir).expanduser().resolve()
    validate_run_id(run_dir.name)
    if not run_dir.is_dir() or run_dir.parent.name != PLAN_ID:
        raise ContractError("run_dir must be output/{}/<run_id>".format(PLAN_ID))
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ContractError("checkpoint does not exist: {}".format(checkpoint_path))
    if loaded_iteration <= 0 or primitive_count <= 0:
        raise ContractError("loaded_iteration and primitive_count must be positive")
    if not isinstance(train_stats, Mapping):
        raise ContractError("train_stats must be a JSON object")
    command_path = run_dir / "command.txt"
    if not command_path.is_file() or not command_path.read_text(encoding="utf-8").strip():
        raise ContractError("command.txt is missing or empty")

    config = _required_json(run_dir, "config.json")
    environment = _required_json(run_dir, "environment.json")
    source_state = _required_json(run_dir, "source_state.json")
    dataset_split = _required_json(run_dir, "dataset_split.json")
    repeat_1 = _required_json(run_dir, "metrics_repeat_1.json")
    repeat_2 = _required_json(run_dir, "metrics_repeat_2.json")
    if int(repeat_1.get("repeat_index", -1)) != 1 or int(repeat_2.get("repeat_index", -1)) != 2:
        raise ContractError("metric repeat files must have repeat_index 1 and 2")
    for repeat in (repeat_1, repeat_2):
        if int(repeat.get("loaded_iteration", -1)) != loaded_iteration:
            raise ContractError("metric repeat loaded_iteration does not match finalization")
        if int(repeat.get("primitive_count", -1)) != primitive_count:
            raise ContractError("metric repeat primitive_count does not match finalization")
    point_cloud_1 = repeat_1.get("point_cloud")
    point_cloud_2 = repeat_2.get("point_cloud")
    if not isinstance(point_cloud_1, Mapping) or not isinstance(point_cloud_2, Mapping):
        raise ContractError("metric repeats must identify the loaded point cloud")
    point_cloud_path = Path(str(point_cloud_1.get("path", ""))).expanduser().resolve()
    if point_cloud_path != Path(str(point_cloud_2.get("path", ""))).expanduser().resolve():
        raise ContractError("metric repeats loaded different point-cloud files")
    if not point_cloud_path.is_file():
        raise ContractError("loaded point cloud is unavailable: {}".format(point_cloud_path))
    point_cloud_bytes = point_cloud_path.stat().st_size
    if any(int(row.get("bytes", -1)) != point_cloud_bytes for row in (point_cloud_1, point_cloud_2)):
        raise ContractError("loaded point-cloud byte size differs from metric evidence")

    targets = [
        run_dir / "baseline_id.txt",
        run_dir / "metrics_per_view.json",
        run_dir / "metrics_summary.json",
        run_dir / "runtime.json",
        run_dir / "manifest.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise ContractError("refusing to overwrite finalized evidence: {}".format(existing))

    per_view, metrics_summary = summarize_metric_repeats(repeat_1, repeat_2)
    checkpoint_identity = {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "loaded_iteration": int(loaded_iteration),
        "primitive_count": int(primitive_count),
    }
    point_cloud_identity = {
        "path": str(point_cloud_path),
        "bytes": point_cloud_bytes,
        "sha256": sha256_file(point_cloud_path),
    }
    binding = {
        "plan_id": PLAN_ID,
        "source": source_state,
        "dataset_identity_sha256": dataset_split.get("dataset_identity_sha256"),
        "dataset_split_sha256": stable_json_hash(dataset_split.get("split", dataset_split)),
        "configuration": config,
        "environment": environment,
        "checkpoint": {
            "bytes": checkpoint_identity["bytes"],
            "sha256": checkpoint_identity["sha256"],
            "loaded_iteration": checkpoint_identity["loaded_iteration"],
            "primitive_count": checkpoint_identity["primitive_count"],
        },
        "point_cloud": {
            "bytes": point_cloud_identity["bytes"],
            "sha256": point_cloud_identity["sha256"],
        },
    }
    baseline_id = make_baseline_id(binding)
    runtime = {
        "training": dict(train_stats),
        "metric_repeat_1": repeat_1.get("runtime", {}),
        "metric_repeat_2": repeat_2.get("runtime", {}),
    }
    metrics_summary["quality_guard_defaults"] = {
        "max_psnr_drop_db": 0.10,
        "max_lpips_increase": 0.005,
        "status": "PENDING_RESEARCH_REVIEW",
        "note": "Research owner must compare these defaults with measured repeatability noise before variants.",
    }

    write_json(run_dir / "metrics_per_view.json", per_view, exclusive=True)
    write_json(run_dir / "metrics_summary.json", metrics_summary, exclusive=True)
    write_json(run_dir / "runtime.json", runtime, exclusive=True)
    with (run_dir / "baseline_id.txt").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(baseline_id + "\n")
    manifest: Dict[str, Any] = {
        "plan_id": PLAN_ID,
        "baseline_id": baseline_id,
        "decision": "PENDING_RESEARCH_REVIEW",
        "binding": binding,
        "checkpoint": checkpoint_identity,
        "point_cloud": point_cloud_identity,
        "evidence": {
            "command": "command.txt",
            "config": "config.json",
            "environment": "environment.json",
            "source_state": "source_state.json",
            "dataset_split": "dataset_split.json",
            "metrics_per_view": "metrics_per_view.json",
            "metrics_summary": "metrics_summary.json",
            "runtime": "runtime.json",
            "metric_repeats": ["metrics_repeat_1.json", "metrics_repeat_2.json"],
        },
    }
    write_json(run_dir / "manifest.json", manifest, exclusive=True)
    return manifest


def verify_evidence(run_dir: Path) -> Dict[str, Any]:
    """Recompute all deterministic Stage 0 bindings and reject tampering/drift."""

    run_dir = Path(run_dir).expanduser().resolve()
    validate_run_id(run_dir.name)
    manifest = _required_json(run_dir, "manifest.json")
    if manifest.get("plan_id") != PLAN_ID:
        raise ContractError("manifest plan_id does not match {}".format(PLAN_ID))
    baseline_path = run_dir / "baseline_id.txt"
    if not baseline_path.is_file():
        raise ContractError("baseline_id.txt is missing")
    recorded_id = baseline_path.read_text(encoding="utf-8").strip()
    if recorded_id != manifest.get("baseline_id"):
        raise ContractError("baseline_id.txt differs from manifest baseline_id")

    config = _required_json(run_dir, "config.json")
    environment = _required_json(run_dir, "environment.json")
    source_state = _required_json(run_dir, "source_state.json")
    dataset_split = _required_json(run_dir, "dataset_split.json")
    repeat_1 = _required_json(run_dir, "metrics_repeat_1.json")
    repeat_2 = _required_json(run_dir, "metrics_repeat_2.json")
    recorded_per_view = read_json(run_dir / "metrics_per_view.json")
    recorded_summary = read_json(run_dir / "metrics_summary.json")
    expected_per_view, expected_summary = summarize_metric_repeats(repeat_1, repeat_2)
    expected_summary["quality_guard_defaults"] = {
        "max_psnr_drop_db": 0.10,
        "max_lpips_increase": 0.005,
        "status": "PENDING_RESEARCH_REVIEW",
        "note": "Research owner must compare these defaults with measured repeatability noise before variants.",
    }
    if stable_json_hash(recorded_per_view) != stable_json_hash(expected_per_view):
        raise ContractError("metrics_per_view.json does not match the two repeat files")
    if stable_json_hash(recorded_summary) != stable_json_hash(expected_summary):
        raise ContractError("metrics_summary.json does not match the two repeat files")

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ContractError("manifest checkpoint identity is missing")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ContractError("bound checkpoint is unavailable: {}".format(checkpoint_path))
    if checkpoint_path.stat().st_size != int(checkpoint.get("bytes", -1)):
        raise ContractError("bound checkpoint byte size changed")
    if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise ContractError("bound checkpoint SHA-256 changed")
    point_cloud = manifest.get("point_cloud")
    if not isinstance(point_cloud, Mapping):
        raise ContractError("manifest point-cloud identity is missing")
    point_cloud_path = Path(str(point_cloud.get("path", ""))).expanduser().resolve()
    if not point_cloud_path.is_file():
        raise ContractError("bound point cloud is unavailable: {}".format(point_cloud_path))
    if point_cloud_path.stat().st_size != int(point_cloud.get("bytes", -1)):
        raise ContractError("bound point-cloud byte size changed")
    if sha256_file(point_cloud_path) != point_cloud.get("sha256"):
        raise ContractError("bound point-cloud SHA-256 changed")

    binding = {
        "plan_id": PLAN_ID,
        "source": source_state,
        "dataset_identity_sha256": dataset_split.get("dataset_identity_sha256"),
        "dataset_split_sha256": stable_json_hash(dataset_split.get("split", dataset_split)),
        "configuration": config,
        "environment": environment,
        "checkpoint": {
            "bytes": int(checkpoint["bytes"]),
            "sha256": checkpoint["sha256"],
            "loaded_iteration": int(checkpoint["loaded_iteration"]),
            "primitive_count": int(checkpoint["primitive_count"]),
        },
        "point_cloud": {
            "bytes": int(point_cloud["bytes"]),
            "sha256": point_cloud["sha256"],
        },
    }
    recomputed_id = make_baseline_id(binding)
    if recomputed_id != recorded_id:
        raise ContractError(
            "baseline_id no longer matches current evidence: recorded {}, recomputed {}".format(
                recorded_id, recomputed_id
            )
        )

    frozen_split = dataset_split.get("split", {})
    expected_keys = {("test", str(name)) for name in frozen_split.get("test", [])}
    actual_keys = {(str(row["split"]), str(row["image_name"])) for row in recorded_per_view}
    if actual_keys != expected_keys:
        raise ContractError("metric view set differs from the frozen test split")
    decision = str(manifest.get("decision", ""))
    allowed_decisions = {"PENDING_RESEARCH_REVIEW", "PROCEED", "REVISE_ONCE", "STOP"}
    if decision not in allowed_decisions:
        raise ContractError("invalid Stage 0 decision: {}".format(decision))
    return {
        "status": "VALID",
        "plan_id": PLAN_ID,
        "baseline_id": recorded_id,
        "decision": decision,
        "view_count": len(recorded_per_view),
        "checkpoint_sha256": checkpoint["sha256"],
    }
