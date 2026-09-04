#!/usr/bin/env python3
"""Run one immutable Stage 0 baseline metric repeat.

Invoke this script twice with ``--repeat-index 1`` and ``2``. It loads the same
saved DRK point cloud, reuses the production render and metric implementations,
and writes atomic per-view evidence without changing training or renderer code.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.s0_baseline_contract import (  # noqa: E402
    ContractError,
    PLAN_ID,
    read_json,
    validate_run_id,
    write_json,
)


def validate_split_contract(
    train_names: Sequence[str], test_names: Sequence[str], dataset_manifest: Mapping[str, Any]
) -> None:
    frozen = dataset_manifest.get("split")
    if not isinstance(frozen, Mapping):
        raise ContractError("dataset_split.json does not contain a split object")
    actual_train = sorted(str(name) for name in train_names)
    actual_test = sorted(str(name) for name in test_names)
    expected_train = sorted(str(name) for name in frozen.get("train", []))
    expected_test = sorted(str(name) for name in frozen.get("test", []))
    if actual_train != expected_train:
        raise ContractError("loaded train split differs from frozen dataset manifest")
    if actual_test != expected_test:
        raise ContractError("loaded test split differs from frozen dataset manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--model-path", required=True, help="resolved directory including the _DRK suffix")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--repeat-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--load-iteration", type=int, default=35000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--kernel-k", type=int, default=8)
    parser.add_argument("--gs-type", default="DRK")
    parser.add_argument("--kernel-density", default="dense")
    parser.add_argument("--cache-sort", action="store_true")
    parser.add_argument("--tile-culling", action="store_true")
    parser.add_argument("--is-unbounded", action="store_true")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--metric-masked", action="store_true")
    return parser


def _assert_frozen_arguments(arguments: argparse.Namespace, config: Mapping[str, Any]) -> None:
    checks = {
        "images": arguments.images,
        "resolution": arguments.resolution,
        "gs_type": arguments.gs_type,
        "kernel_density": arguments.kernel_density,
        "cache_sort": arguments.cache_sort,
        "tile_culling": arguments.tile_culling,
        "is_unbounded": arguments.is_unbounded,
        "white_background": arguments.white_background,
    }
    for key, actual in checks.items():
        if config.get(key) != actual:
            raise ContractError(
                "parity argument {}={!r} differs from frozen config {!r}".format(
                    key, actual, config.get(key)
                )
            )
    if config.get("pose_refine") is not False:
        raise ContractError("Stage 0 parity requires pose_refine=false")
    source_path = Path(arguments.source_path).expanduser().resolve()
    if source_path != Path(str(config.get("dataset", ""))).expanduser().resolve():
        raise ContractError("parity source path differs from frozen dataset path")
    model_path = Path(arguments.model_path).expanduser().resolve()
    if model_path != Path(str(config.get("resolved_model_path", ""))).expanduser().resolve():
        raise ContractError("parity model path differs from frozen resolved model path")
    if arguments.load_iteration != int(config.get("iterations", -1)):
        raise ContractError("parity load iteration differs from frozen iteration budget")


def _finite_scalar(value: Any, label: str) -> float:
    scalar = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
    if not math.isfinite(scalar):
        raise ContractError("non-finite metric {}".format(label))
    return scalar


def _render_split(
    split: str,
    cameras: Sequence[Any],
    gaussians: Any,
    pipeline: Any,
    background: Any,
    metric_function: Any,
    l1_function: Any,
    torch: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for camera in cameras:
        render_package = gaussians.render_func(camera, gaussians, pipeline, background)
        raw = render_package["render"]
        nonfinite = int((~torch.isfinite(raw)).sum().item())
        if nonfinite:
            raise ContractError("render contains {} non-finite values for {}".format(nonfinite, camera.image_name))
        image = torch.clamp(raw, 0.0, 1.0)
        ground_truth = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        psnr_value, ssim_value, lpips_value = metric_function(ground_truth, image)
        rows.append(
            {
                "split": split,
                "image_name": str(camera.image_name),
                "width": int(camera.image_width),
                "height": int(camera.image_height),
                "l1": _finite_scalar(l1_function(image, ground_truth), "l1"),
                "psnr": _finite_scalar(psnr_value, "psnr"),
                "ssim": _finite_scalar(ssim_value, "ssim"),
                "lpips": _finite_scalar(lpips_value, "lpips"),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        evidence_dir = Path(arguments.evidence_dir).expanduser().resolve()
        validate_run_id(evidence_dir.name)
        if evidence_dir.parent.name != PLAN_ID:
            raise ContractError("evidence-dir must be output/{}/<run_id>".format(PLAN_ID))
        destination = evidence_dir / "metrics_repeat_{}.json".format(arguments.repeat_index)
        if destination.exists():
            raise ContractError("refusing to overwrite {}".format(destination))
        config = read_json(evidence_dir / "config.json")
        dataset_manifest = read_json(evidence_dir / "dataset_split.json")
        _assert_frozen_arguments(arguments, config)
        if not (evidence_dir / "environment.json").is_file():
            raise ContractError("environment.json must be captured after rebuilding extensions")

        import torch
        from scene import Scene
        from scene.gaussian_model import DRKModel
        from utils.general_utils import safe_state
        from utils.loss_utils import l1_loss
        from utils.metric import metric

        if not torch.cuda.is_available():
            raise ContractError("CUDA is unavailable")
        safe_state(True)
        torch.cuda.reset_peak_memory_stats()
        dataset = Namespace(
            sh_degree=arguments.sh_degree,
            source_path=str(Path(arguments.source_path).expanduser().resolve()),
            model_path=str(Path(arguments.model_path).expanduser().resolve()),
            images=arguments.images,
            resolution=arguments.resolution,
            white_background=arguments.white_background,
            data_device="cuda",
            eval=True,
            val_as_train=False,
            gs_type="DRK",
            metric_masked=arguments.metric_masked,
        )
        pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        gaussians = DRKModel(arguments.sh_degree, kernel_K=arguments.kernel_k)
        scene = Scene(dataset, gaussians, load_iteration=arguments.load_iteration, shuffle=False)
        gaussians.cache_sort = arguments.cache_sort
        gaussians.tile_culling = arguments.tile_culling
        gaussians.update(arguments.load_iteration)
        gaussians.eval()
        if getattr(gaussians, "pose_refine", None) is not None:
            raise ContractError("loaded model unexpectedly contains pose refinement")

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        validate_split_contract(
            [camera.image_name for camera in train_cameras],
            [camera.image_name for camera in test_cameras],
            dataset_manifest,
        )
        background = torch.tensor(
            [1.0, 1.0, 1.0] if arguments.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            views = _render_split(
                "test", test_cameras, gaussians, pipeline, background, metric, l1_loss, torch
            )
        torch.cuda.synchronize()
        wall_time = time.perf_counter() - started
        point_cloud_path = (
            Path(arguments.model_path)
            / "point_cloud"
            / "iteration_{}".format(arguments.load_iteration)
            / "point_cloud.ply"
        ).resolve()
        if not point_cloud_path.is_file():
            raise ContractError("loaded point cloud does not exist: {}".format(point_cloud_path))
        payload = {
            "plan_id": PLAN_ID,
            "repeat_index": arguments.repeat_index,
            "loaded_iteration": arguments.load_iteration,
            "primitive_count": int(gaussians.get_xyz.shape[0]),
            "point_cloud": {
                "path": str(point_cloud_path),
                "bytes": point_cloud_path.stat().st_size,
            },
            "views": views,
            "runtime": {
                "wall_time_sec": wall_time,
                "view_count": len(views),
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        write_json(destination, payload, exclusive=True)
        print(destination)
        return 0
    except ContractError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
