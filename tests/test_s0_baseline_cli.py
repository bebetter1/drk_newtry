import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

from scripts.s0_baseline_contract import build_parser, main, validate_runtime_versions


class RuntimeVersionTests(unittest.TestCase):
    def test_accepts_server_torch_cu121_with_matching_cuda_toolkit(self):
        runtime = validate_runtime_versions(
            "2.2.2+cu121",
            "12.1",
            "Cuda compilation tools, release 12.1, V12.1.105",
        )

        self.assertEqual(runtime["torch_release"], "2.2.2")
        self.assertEqual(runtime["torch_cuda"], "12.1")
        self.assertEqual(runtime["nvcc_cuda"], "12.1")

    def test_rejects_cuda_toolkit_major_mismatch(self):
        with self.assertRaisesRegex(ValueError, "major version"):
            validate_runtime_versions(
                "2.2.2+cu121",
                "12.1",
                "Cuda compilation tools, release 11.8, V11.8.89",
            )


class PrepareCliTests(unittest.TestCase):
    def test_verify_subcommand_accepts_finalized_run(self):
        args = build_parser().parse_args(["verify", "--run-dir", "/evidence/S0_BASELINE_V1/run"])
        self.assertEqual(args.command, "verify")

    def test_prepare_creates_static_evidence_and_replay_commands(self):
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "garden"
            (dataset / "images").mkdir(parents=True)
            (dataset / "sparse" / "0").mkdir(parents=True)
            Image.new("RGB", (32, 24), color=(1, 2, 3)).save(dataset / "images" / "a.png")
            Image.new("RGB", (32, 24), color=(4, 5, 6)).save(dataset / "images" / "b.png")
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (dataset / "sparse" / "0" / name).write_bytes(name.encode("ascii"))

            exit_code = main(
                [
                    "prepare",
                    "--source-root",
                    str(source_root),
                    "--dataset",
                    str(dataset),
                    "--evidence-root",
                    str(root / "evidence"),
                    "--run-id",
                    "20260904-231500_garden_baseline_seed0",
                    "--smoke-model-base",
                    "/runs/smoke",
                    "--full-model-base",
                    "/runs/full",
                ]
            )

            self.assertEqual(exit_code, 0)
            run_dir = (
                root
                / "evidence"
                / "S0_BASELINE_V1"
                / "20260904-231500_garden_baseline_seed0"
            )
            for name in ("command.txt", "config.json", "source_state.json", "dataset_split.json"):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertFalse((run_dir / "environment.json").exists())
            commands = (run_dir / "command.txt").read_text(encoding="utf-8")
            self.assertIn("capture-environment", commands)
            self.assertIn("--repeat-index 1", commands)
            self.assertIn("--repeat-index 2", commands)
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            resolved = config["resolved_training_args"]
            self.assertEqual(resolved["iterations"], 35000)
            self.assertEqual(resolved["model_path"], "/runs/full_DRK")
            self.assertTrue(resolved["cache_sort"])
            self.assertTrue(resolved["is_unbounded"])
            self.assertFalse(resolved["pose_refine"])


if __name__ == "__main__":
    unittest.main()
