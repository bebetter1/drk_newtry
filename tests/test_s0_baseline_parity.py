import unittest

from scripts.run_s0_baseline_parity import build_parser, validate_split_contract
from utils.s0_baseline_contract import ContractError


class ParityContractTests(unittest.TestCase):
    def test_parser_requires_explicit_repeat_and_evidence_directory(self):
        args = build_parser().parse_args(
            [
                "--source-path",
                "/data/garden",
                "--model-path",
                "/runs/garden_DRK",
                "--evidence-dir",
                "/evidence/run",
                "--repeat-index",
                "2",
                "--cache-sort",
                "--is-unbounded",
            ]
        )

        self.assertEqual(args.repeat_index, 2)
        self.assertTrue(args.cache_sort)
        self.assertTrue(args.is_unbounded)
        self.assertEqual(args.gs_type, "DRK")
        self.assertEqual(args.kernel_density, "dense")

    def test_accepts_exact_scene_split_and_rejects_drift(self):
        frozen = {"split": {"train": ["b", "c"], "test": ["a"], "val": []}}
        validate_split_contract(["c", "b"], ["a"], frozen)

        with self.assertRaisesRegex(ContractError, "train split"):
            validate_split_contract(["b"], ["a"], frozen)


if __name__ == "__main__":
    unittest.main()
