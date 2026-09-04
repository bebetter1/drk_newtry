#!/usr/bin/env python3
"""Prepare and finalize immutable evidence for ``S0_BASELINE_V1``.

The script never starts training by itself. ``prepare`` writes a reviewable
``command.txt``; the user executes those commands explicitly on the server.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.s0_baseline_contract import (  # noqa: E402
    ContractError,
    PLAN_ID,
    S0_ALLOWED_TOOL_PATHS,
    build_replay_commands,
    collect_dataset_manifest,
    collect_source_state,
    create_evidence_run,
    finalize_evidence,
    read_json,
    sha256_file,
    validate_run_id,
    verify_evidence,
    write_json,
)


EXPECTED_BASELINE_COMMIT = "e9a3f557abde43e28ece3b648a337822d183dcb6"
SUPPORTED_TORCH_RELEASE = "2.2.2"
SUPPORTED_TORCH_CUDA = {"11.8", "12.1"}


def _run_text(command: Sequence[str]) -> Dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def validate_runtime_versions(
    torch_version: str, torch_cuda: Optional[str], nvcc_output: str
) -> Dict[str, str]:
    """Validate a supported PyTorch build and a compatible CUDA compiler."""

    torch_release = str(torch_version).split("+", 1)[0]
    if torch_release != SUPPORTED_TORCH_RELEASE:
        raise ContractError(
            "Stage 0 requires torch {}; found {}".format(
                SUPPORTED_TORCH_RELEASE, torch_version
            )
        )
    if torch_cuda not in SUPPORTED_TORCH_CUDA:
        raise ContractError(
            "Stage 0 supports torch CUDA builds {}; found {}".format(
                sorted(SUPPORTED_TORCH_CUDA), torch_cuda
            )
        )
    match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
    if match is None:
        raise ContractError("could not parse the CUDA version from nvcc --version")
    nvcc_cuda = match.group(1)
    if nvcc_cuda.split(".", 1)[0] != torch_cuda.split(".", 1)[0]:
        raise ContractError(
            "nvcc CUDA {} and torch CUDA {} have different major versions".format(
                nvcc_cuda, torch_cuda
            )
        )
    return {
        "torch_release": torch_release,
        "torch_cuda": torch_cuda,
        "nvcc_cuda": nvcc_cuda,
    }


def _module_identity(name: str, hash_binary: bool = False) -> Dict[str, Any]:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    row: Dict[str, Any] = {
        "module": name,
        "file": str(path) if path else None,
    }
    extension_spec = importlib.util.find_spec(name + "._C")
    if extension_spec is None or not extension_spec.origin:
        raise ContractError("compiled extension is not importable: {}._C".format(name))
    extension_path = Path(extension_spec.origin).resolve()
    compiled_extension = {
        "file": str(extension_path),
        "bytes": extension_path.stat().st_size,
    }
    if hash_binary:
        compiled_extension["sha256"] = sha256_file(extension_path)
    row["compiled_extension"] = compiled_extension
    return row


def _resolved_training_args(
    dataset_root: Path,
    model_base: str,
    images: str,
    resolution: int,
    iterations: int,
) -> Dict[str, Any]:
    """Resolve the same defaults and explicit flags used by ``train.py``."""

    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--W", type=int, default=800)
    parser.add_argument("--H", type=int, default=800)
    parser.add_argument("--elevation", type=float, default=0)
    parser.add_argument("--radius", type=float, default=5)
    parser.add_argument("--fovy", type=float, default=50)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 24000, 30000, 35000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 24000, 30000, 35000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--load_iteration", type=int, default=-1)
    resolved = parser.parse_args(
        [
            "-s",
            str(dataset_root),
            "-m",
            model_base,
            "--eval",
            "--gs_type",
            "DRK",
            "--kernel_density",
            "dense",
            "--cache_sort",
            "--is_unbounded",
            "--images",
            images,
            "--resolution",
            str(resolution),
            "--iterations",
            str(iterations),
            "--checkpoint_iterations",
            str(iterations),
        ]
    )
    resolved.save_iterations.append(resolved.iterations)
    resolved.model_path = resolved.model_path + "_DRK"
    return dict(vars(resolved))


def collect_environment(source_root: Path) -> Dict[str, Any]:
    """Capture and validate the documented Python/PyTorch/CUDA runtime."""

    if sys.version_info[:2] != (3, 9):
        raise ContractError("Stage 0 requires Python 3.9; found {}".format(platform.python_version()))
    import torch

    if not torch.cuda.is_available():
        raise ContractError("torch.cuda.is_available() is false")
    nvcc = _run_text(["nvcc", "--version"])
    if nvcc["returncode"] != 0:
        raise ContractError("nvcc --version failed: {}".format(nvcc["stderr"]))
    runtime_versions = validate_runtime_versions(
        torch.__version__, torch.version.cuda, nvcc["stdout"]
    )

    extension_names = (
        "diff_gaussian_rasterization",
        "drk_splatting",
        "simple_knn",
    )
    extensions = {
        name: _module_identity(name, hash_binary=(name == "drk_splatting"))
        for name in extension_names
    }
    drk_module = importlib.import_module("drk_splatting")
    required_drk_symbols = (
        "make_rasterize_gaussians_forward_args",
        "rasterize_gaussians_forward_from_args",
    )
    missing_symbols = [name for name in required_drk_symbols if not hasattr(drk_module, name)]
    if missing_symbols:
        raise ContractError("drk_splatting is missing required symbols: {}".format(missing_symbols))

    source_root = Path(source_root).expanduser().resolve()
    build_stamps = []
    for path in sorted((source_root / "submodules" / "drk_splatting").glob("build/**/drk_build_config.txt")):
        build_stamps.append(
            {
                "path": str(path.resolve()),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    if not build_stamps:
        raise ContractError("DRK build configuration stamp was not found; rebuild the extension from this source")

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    nvidia_smi = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia_smi["returncode"] != 0:
        raise ContractError("nvidia-smi query failed: {}".format(nvidia_smi["stderr"]))

    package_names = (
        "numpy",
        "torchvision",
        "piq",
        "Pillow",
        "scipy",
    )
    packages: Dict[str, Optional[str]] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    drk_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("DRK_") or key in {"CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST"}
    }
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpu": {
            "visible_index": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "nvcc": nvcc,
        "runtime_compatibility": runtime_versions,
        "nvidia_smi": nvidia_smi,
        "packages": packages,
        "extensions": extensions,
        "drk_build_stamps": build_stamps,
        "compile_environment": drk_environment,
    }


def _prepare(arguments: argparse.Namespace) -> int:
    source_root = Path(arguments.source_root).expanduser().resolve()
    dataset_root = Path(arguments.dataset).expanduser().resolve()
    validate_run_id(arguments.run_id)
    run_dir = create_evidence_run(Path(arguments.evidence_root), arguments.run_id)
    try:
        source_state = collect_source_state(
            source_root,
            expected_commit=arguments.expected_baseline_commit,
            allowed_paths=S0_ALLOWED_TOOL_PATHS,
        )
        dataset = collect_dataset_manifest(
            dataset_root,
            images_directory=arguments.images,
            resolution=arguments.resolution,
        )
        config = {
            "plan_id": PLAN_ID,
            "seed": 0,
            "source_root": str(source_root),
            "dataset": str(dataset_root),
            "images": arguments.images,
            "resolution": arguments.resolution,
            "eval": True,
            "gs_type": "DRK",
            "kernel_density": "dense",
            "cache_sort": True,
            "tile_culling": False,
            "is_unbounded": True,
            "pose_refine": False,
            "white_background": False,
            "smoke_iterations": arguments.smoke_iterations,
            "iterations": arguments.iterations,
            "gpu_id": arguments.gpu_id,
            "requested_smoke_model_path": arguments.smoke_model_base,
            "resolved_smoke_model_path": arguments.smoke_model_base + "_DRK",
            "requested_model_path": arguments.full_model_base,
            "resolved_model_path": arguments.full_model_base + "_DRK",
            "expected_baseline_commit": arguments.expected_baseline_commit,
            "resolved_training_args": _resolved_training_args(
                dataset_root,
                arguments.full_model_base,
                arguments.images,
                arguments.resolution,
                arguments.iterations,
            ),
        }
        commands = build_replay_commands(
            source_root=Path(arguments.source_root),
            dataset_root=Path(arguments.dataset),
            smoke_model_base=Path(arguments.smoke_model_base),
            full_model_base=Path(arguments.full_model_base),
            run_dir=Path(run_dir.as_posix()),
            python_executable=arguments.python,
            gpu_id=arguments.gpu_id,
            images_directory=arguments.images,
            resolution=arguments.resolution,
            smoke_iterations=arguments.smoke_iterations,
            iterations=arguments.iterations,
        )
        write_json(run_dir / "config.json", config, exclusive=True)
        write_json(run_dir / "source_state.json", source_state, exclusive=True)
        write_json(run_dir / "dataset_split.json", dataset, exclusive=True)
        with (run_dir / "command.txt").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(commands)
    except Exception:
        (run_dir / "PREPARE_FAILED").write_text("Preparation failed; preserve this directory for diagnosis.\n", encoding="utf-8")
        raise
    print(run_dir)
    return 0


def _capture_environment(arguments: argparse.Namespace) -> int:
    run_dir = Path(arguments.run_dir).expanduser().resolve()
    validate_run_id(run_dir.name)
    source_state = read_json(run_dir / "source_state.json")
    environment_path = run_dir / "environment.json"
    if environment_path.exists():
        raise ContractError("refusing to overwrite {}".format(environment_path))
    environment = collect_environment(Path(source_state["source_root"]))
    write_json(environment_path, environment, exclusive=True)
    print(environment_path)
    return 0


def _checkpoint_shape(checkpoint_path: Path) -> Dict[str, int]:
    import torch

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ContractError("checkpoint must contain (model_params, iteration)")
    model_params, iteration = payload
    if not isinstance(model_params, (tuple, list)) or len(model_params) < 2:
        raise ContractError("checkpoint model_params payload is invalid")
    xyz = model_params[1]
    if not hasattr(xyz, "shape") or len(xyz.shape) != 2 or int(xyz.shape[1]) != 3:
        raise ContractError("checkpoint xyz tensor must have shape [N,3]")
    return {"loaded_iteration": int(iteration), "primitive_count": int(xyz.shape[0])}


def _finalize(arguments: argparse.Namespace) -> int:
    run_dir = Path(arguments.run_dir).expanduser().resolve()
    checkpoint = Path(arguments.checkpoint).expanduser().resolve()
    train_stats = read_json(Path(arguments.train_stats).expanduser().resolve())
    shape = _checkpoint_shape(checkpoint)
    manifest = finalize_evidence(
        run_dir,
        checkpoint,
        loaded_iteration=shape["loaded_iteration"],
        primitive_count=shape["primitive_count"],
        train_stats=train_stats,
    )
    print(json.dumps({"baseline_id": manifest["baseline_id"], "decision": manifest["decision"]}, sort_keys=True))
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    verification = verify_evidence(Path(arguments.run_dir))
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create a new immutable evidence directory")
    prepare.add_argument("--source-root", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--evidence-root", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--smoke-model-base", required=True)
    prepare.add_argument("--full-model-base", required=True)
    prepare.add_argument("--expected-baseline-commit", default=EXPECTED_BASELINE_COMMIT)
    prepare.add_argument("--python", default="python")
    prepare.add_argument("--gpu-id", type=int, default=0)
    prepare.add_argument("--images", default="images")
    prepare.add_argument("--resolution", type=int, default=-1)
    prepare.add_argument("--smoke-iterations", type=int, default=10)
    prepare.add_argument("--iterations", type=int, default=35000)
    prepare.set_defaults(handler=_prepare)

    capture = subparsers.add_parser("capture-environment", help="record the rebuilt CUDA runtime")
    capture.add_argument("--run-dir", required=True)
    capture.set_defaults(handler=_capture_environment)

    finalize = subparsers.add_parser("finalize", help="bind checkpoint and repeated metrics")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--checkpoint", required=True)
    finalize.add_argument("--train-stats", required=True)
    finalize.set_defaults(handler=_finalize)

    verify = subparsers.add_parser("verify", help="recompute hashes and validate finalized evidence")
    verify.add_argument("--run-dir", required=True)
    verify.set_defaults(handler=_verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except ContractError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
