from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.reconstruct_alpaca_farm_gold_rm import (
    read_expected_model_sum,
    reconstruct_tuned_model,
    require_empty_output_root,
    verify_hf_checkpoint_layout,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], value: float) -> None:
        self.shape = shape
        self.value = value

    def size(self) -> tuple[int, ...]:
        return self.shape

    def add_(self, other: "FakeTensor") -> None:
        self.value += other.value


class FakeModel:
    def __init__(self, state: dict[str, FakeTensor]) -> None:
        self.state = state

    def state_dict(self) -> dict[str, FakeTensor]:
        return self.state


class ReconstructionInputTests(unittest.TestCase):
    def test_model_sum_must_be_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_sum.txt"
            path.write_text("12.5\n", encoding="utf-8")
            self.assertEqual(read_expected_model_sum(path), 12.5)
            path.write_text("nan\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                read_expected_model_sum(path)

    def test_nonempty_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            require_empty_output_root(root)
            (root / "partial-artifact").write_text("evidence", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                require_empty_output_root(root)

    def test_checkpoint_layout_requires_every_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytorch_model.bin.index.json").write_text(
                '{"weight_map":{"a":"pytorch_model-00001-of-00002.bin",'
                '"b":"pytorch_model-00002-of-00002.bin"}}',
                encoding="utf-8",
            )
            (root / "pytorch_model-00001-of-00002.bin").write_bytes(b"first")
            with self.assertRaisesRegex(FileNotFoundError, "Missing or empty indexed"):
                verify_hf_checkpoint_layout(root)
            (root / "pytorch_model-00002-of-00002.bin").write_bytes(b"second")
            self.assertEqual(len(verify_hf_checkpoint_layout(root)), 2)

    def test_local_addition_and_shape_mismatch_rule(self) -> None:
        tuned = FakeModel(
            {
                "weight": FakeTensor((2, 2), 2.0),
                "resized_embedding": FakeTensor((3, 2), 7.0),
            }
        )
        raw = FakeModel(
            {
                "weight": FakeTensor((2, 2), 5.0),
                "resized_embedding": FakeTensor((2, 2), 11.0),
            }
        )
        reconstruct_tuned_model(tuned, raw, is_reward_model=False)
        self.assertEqual(tuned.state["weight"].value, 7.0)
        self.assertEqual(tuned.state["resized_embedding"].value, 7.0)

    def test_reward_model_base_keys_are_nested(self) -> None:
        tuned = FakeModel({"backbone_model.weight": FakeTensor((1,), 2.0)})
        raw = FakeModel({"weight": FakeTensor((1,), 3.0)})
        reconstruct_tuned_model(tuned, raw, is_reward_model=True)
        self.assertEqual(tuned.state["backbone_model.weight"].value, 5.0)


if __name__ == "__main__":
    unittest.main()
