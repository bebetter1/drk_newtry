import tempfile
import unittest
import subprocess
from pathlib import Path

from PIL import Image

from utils.s0_baseline_contract import (
    ContractError,
    build_replay_commands,
    collect_dataset_manifest,
    collect_source_state,
    create_evidence_run,
    finalize_evidence,
    make_baseline_id,
    resolved_camera_size,
    split_colmap_names,
    summarize_metric_repeats,
    validate_run_id,
    verify_evidence,
    write_json,
)


class RunIdTests(unittest.TestCase):
    def test_accepts_frozen_run_id_shape(self):
        self.assertEqual(
            validate_run_id("20260904-231500_garden_baseline_seed0"),
            "20260904-231500_garden_baseline_seed0",
        )

    def test_rejects_path_traversal_and_separators(self):
        for value in ("../run", "run/name", r"run\name", "", "."):
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_run_id(value)


class DatasetSplitTests(unittest.TestCase):
    def test_uses_sorted_llff_holdout_eight(self):
        names = [f"frame_{index:03d}" for index in reversed(range(17))]
        split = split_colmap_names(names, llffhold=8)

        self.assertEqual(split["all"], [f"frame_{index:03d}" for index in range(17)])
        self.assertEqual(split["test"], ["frame_000", "frame_008", "frame_016"])
        self.assertEqual(
            split["train"],
            [f"frame_{index:03d}" for index in range(17) if index % 8 != 0],
        )

    def test_rejects_duplicate_names(self):
        with self.assertRaisesRegex(ContractError, "duplicate"):
            split_colmap_names(["a", "a"], llffhold=8)

    def test_resolved_size_matches_repository_auto_downsample(self):
        self.assertEqual(resolved_camera_size(2400, 1600, -1), (1600, 1066))
        self.assertEqual(resolved_camera_size(1200, 800, -1), (1200, 800))
        self.assertEqual(resolved_camera_size(2400, 1600, 4), (600, 400))


class EvidenceDirectoryTests(unittest.TestCase):
    def test_creates_new_run_layout_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = create_evidence_run(root, "20260904-231500_garden_baseline_seed0")

            self.assertTrue((run_dir / "logs").is_dir())
            self.assertTrue((run_dir / "artifacts").is_dir())
            with self.assertRaisesRegex(ContractError, "already exists"):
                create_evidence_run(root, "20260904-231500_garden_baseline_seed0")

    def test_collects_lightweight_colmap_identity_and_frozen_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "garden"
            images = dataset / "images"
            sparse = dataset / "sparse" / "0"
            images.mkdir(parents=True)
            sparse.mkdir(parents=True)
            for name, size in (("b.JPG", (2400, 1600)), ("a.JPG", (1200, 800))):
                Image.new("RGB", size, color=(10, 20, 30)).save(images / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (sparse / name).write_bytes(name.encode("ascii"))

            manifest = collect_dataset_manifest(dataset, "images", resolution=-1)

            self.assertEqual(manifest["image_count"], 2)
            self.assertEqual(manifest["split"]["test"], ["a"])
            self.assertEqual(manifest["split"]["train"], ["b"])
            rows = {row["image_name"]: row for row in manifest["images"]}
            self.assertEqual(rows["b"]["source_size"], [2400, 1600])
            self.assertEqual(rows["b"]["resolved_size"], [1600, 1066])
            self.assertNotIn("sha256", rows["b"])
            self.assertTrue(all("sha256" in row for row in manifest["sparse"]))
            self.assertRegex(manifest["dataset_identity_sha256"], r"^[0-9a-f]{64}$")

    def test_finalize_writes_bound_manifest_without_granting_proceed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = create_evidence_run(
                Path(temporary), "20260904-231500_garden_baseline_seed0"
            )
            write_json(run_dir / "config.json", {"seed": 0, "iterations": 35000})
            write_json(
                run_dir / "environment.json",
                {"python": "3.9.24", "torch": "2.2.2+cu118", "extensions": {"drk": "x"}},
            )
            write_json(
                run_dir / "source_state.json",
                {"commit": "e9a3", "dirty": True, "critical_file_sha256": {"train.py": "x"}},
            )
            write_json(
                run_dir / "dataset_split.json",
                {"dataset_identity_sha256": "dataset", "split": {"train": ["b"], "test": ["a"]}},
            )
            (run_dir / "command.txt").write_text("python train.py\n", encoding="utf-8")
            checkpoint = Path(temporary) / "chkpnt35000.pth"
            checkpoint.write_bytes(b"checkpoint")
            point_cloud = Path(temporary) / "point_cloud.ply"
            point_cloud.write_bytes(b"ply-data")
            common = {
                "loaded_iteration": 35000,
                "primitive_count": 12,
                "point_cloud": {
                    "path": str(point_cloud),
                    "bytes": point_cloud.stat().st_size,
                },
                "views": [
                    {
                        "split": "test",
                        "image_name": "a",
                        "l1": 0.1,
                        "psnr": 20.0,
                        "ssim": 0.8,
                        "lpips": 0.2,
                    },
                ],
            }
            write_json(run_dir / "metrics_repeat_1.json", dict(common, repeat_index=1))
            write_json(run_dir / "metrics_repeat_2.json", dict(common, repeat_index=2))

            manifest = finalize_evidence(
                run_dir,
                checkpoint,
                loaded_iteration=35000,
                primitive_count=12,
                train_stats={"wall_time_sec": 123.0},
            )

            self.assertEqual(manifest["decision"], "PENDING_RESEARCH_REVIEW")
            self.assertRegex(manifest["point_cloud"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["baseline_id"], r"^S0-[0-9a-f]{16}$")
            self.assertEqual(
                (run_dir / "baseline_id.txt").read_text(encoding="utf-8").strip(),
                manifest["baseline_id"],
            )
            self.assertTrue((run_dir / "metrics_per_view.json").is_file())
            self.assertTrue((run_dir / "metrics_summary.json").is_file())
            self.assertTrue((run_dir / "runtime.json").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())

            verification = verify_evidence(run_dir)
            self.assertEqual(verification["status"], "VALID")
            write_json(run_dir / "config.json", {"seed": 999, "iterations": 35000})
            with self.assertRaisesRegex(ContractError, "baseline_id"):
                verify_evidence(run_dir)

    def test_source_state_binds_commit_dirty_files_and_critical_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            (source / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "baseline"], check=True)
            (source / "train.py").write_text("print('changed')\n", encoding="utf-8")
            (source / "new.py").write_text("pass\n", encoding="utf-8")

            state = collect_source_state(source, critical_paths=("train.py",))

            self.assertTrue(state["dirty"])
            self.assertRegex(state["commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(state["changed_files"][0]["path"], "train.py")
            self.assertIn("train.py", state["critical_file_sha256"])

    def test_source_state_allows_committed_tools_above_unchanged_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            (source / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "baseline"], check=True)
            baseline = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (source / "tool.py").write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(source), "add", "tool.py"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "tool"], check=True)

            state = collect_source_state(
                source,
                expected_commit=baseline,
                critical_paths=("train.py",),
                allowed_paths=("tool.py",),
            )

            self.assertEqual(state["commit"], baseline)
            self.assertNotEqual(state["execution_head"], baseline)
            self.assertEqual(state["diagnostic_files_since_baseline"], ["tool.py"])


class BaselineIdTests(unittest.TestCase):
    def test_is_stable_under_mapping_order(self):
        left = {"commit": "abc", "dataset": {"name": "garden", "count": 185}}
        right = {"dataset": {"count": 185, "name": "garden"}, "commit": "abc"}

        self.assertEqual(make_baseline_id(left), make_baseline_id(right))
        self.assertRegex(make_baseline_id(left), r"^S0-[0-9a-f]{16}$")


class ReplayCommandTests(unittest.TestCase):
    def test_freezes_required_baseline_flags_and_two_independent_repeats(self):
        commands = build_replay_commands(
            source_root=Path("/srv/DRK_NEW_TRY"),
            dataset_root=Path("/data/garden"),
            smoke_model_base=Path("/runs/smoke"),
            full_model_base=Path("/runs/garden_baseline"),
            run_dir=Path("/evidence/S0_BASELINE_V1/run"),
            python_executable="python",
            gpu_id=3,
            images_directory="images",
            resolution=-1,
            smoke_iterations=10,
            iterations=35000,
        )

        self.assertIn("export CUDA_VISIBLE_DEVICES=3", commands)
        self.assertIn("--gs_type DRK --kernel_density dense --cache_sort --is_unbounded", commands)
        self.assertIn("--checkpoint_iterations 35000", commands)
        self.assertNotIn("--metric --load_iteration 35000", commands)
        self.assertNotIn("metric_official.log", commands)
        self.assertIn("--model-path /runs/garden_baseline_DRK", commands)
        self.assertEqual(commands.count("--repeat-index"), 2)
        self.assertNotIn(" verify --run-dir", commands)
        self.assertNotIn("--pose_refine", commands)


class MetricRepeatTests(unittest.TestCase):
    def test_pairs_views_and_reports_repeatability_noise(self):
        repeat_1 = {
            "repeat_index": 1,
            "views": [
                {
                    "split": "test",
                    "image_name": "a",
                    "l1": 0.10,
                    "psnr": 20.0,
                    "ssim": 0.80,
                    "lpips": 0.20,
                },
                {
                    "split": "train",
                    "image_name": "b",
                    "l1": 0.05,
                    "psnr": 25.0,
                    "ssim": 0.90,
                    "lpips": 0.10,
                },
            ],
        }
        repeat_2 = {
            "repeat_index": 2,
            "views": [
                {
                    "split": "test",
                    "image_name": "a",
                    "l1": 0.11,
                    "psnr": 19.9,
                    "ssim": 0.79,
                    "lpips": 0.21,
                },
                {
                    "split": "train",
                    "image_name": "b",
                    "l1": 0.05,
                    "psnr": 25.0,
                    "ssim": 0.90,
                    "lpips": 0.10,
                },
            ],
        }

        per_view, summary = summarize_metric_repeats(repeat_1, repeat_2)

        self.assertEqual(len(per_view), 2)
        test_view = next(row for row in per_view if row["split"] == "test")
        self.assertAlmostEqual(test_view["repeat_abs_delta"]["psnr"], 0.1)
        self.assertNotIn("render_sha256", test_view)
        self.assertNotIn("render_bitwise_equal", test_view)
        self.assertNotIn("bitwise_equal_views", summary["repeatability"])
        self.assertAlmostEqual(summary["repeatability"]["max_abs_delta"]["lpips"], 0.01)
        self.assertEqual(summary["splits"]["test"]["view_count"], 1)

    def test_rejects_mismatched_view_sets(self):
        repeat_1 = {
            "repeat_index": 1,
            "views": [{"split": "test", "image_name": "a"}],
        }
        repeat_2 = {
            "repeat_index": 2,
            "views": [{"split": "test", "image_name": "b"}],
        }

        with self.assertRaisesRegex(ContractError, "view sets"):
            summarize_metric_repeats(repeat_1, repeat_2)


if __name__ == "__main__":
    unittest.main()
