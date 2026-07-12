import importlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import mrcfile
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BACKWARD_DIR = ROOT / "core" / "backward"
sys.path.insert(0, str(BACKWARD_DIR))

best_model_selection = importlib.import_module("best_model_selection")
train = importlib.import_module("train")


class BestModelTests(unittest.TestCase):
    def test_selects_lowest_stable_post_20_epoch_loss(self):
        losses = [{"epoch": epoch, "s2_loss": 1.0} for epoch in range(1, 42)]
        for epoch in range(21, 32):
            losses[epoch - 1]["s2_loss"] = 0.5 + (epoch - 21) * 0.002
        for epoch in range(32, 42):
            losses[epoch - 1]["s2_loss"] = 0.7
        selected = best_model_selection.select_best_epoch(losses)
        self.assertEqual(selected["epoch"], 21)

    def test_rejects_large_jump_and_requires_ten_following_epochs(self):
        losses = [{"epoch": epoch, "s2_loss": 0.5} for epoch in range(1, 31)]
        self.assertIsNone(best_model_selection.select_best_epoch(losses))
        losses.extend({"epoch": epoch, "s2_loss": 0.5} for epoch in range(31, 33))
        losses[25]["s2_loss"] = 0.8
        self.assertIsNone(best_model_selection.select_best_epoch(losses))

    def test_equal_loss_prefers_earlier_epoch(self):
        losses = [{"epoch": epoch, "s2_loss": 0.5} for epoch in range(1, 33)]
        self.assertEqual(best_model_selection.select_best_epoch(losses)["epoch"], 21)

    def test_nonfinite_loss_is_not_a_candidate(self):
        losses = [{"epoch": epoch, "s2_loss": 0.5} for epoch in range(1, 33)]
        losses[20]["s2_loss"] = float("nan")
        self.assertEqual(best_model_selection.select_best_epoch(losses)["epoch"], 22)

    def test_materializes_selected_epoch_and_does_not_fake_best(self):
        stable = [{"epoch": epoch, "s2_loss": 0.5} for epoch in range(1, 32)]
        unstable = stable[:25]
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "21.pth").write_bytes(b"epoch-21")
            selected = best_model_selection.materialize_best_model(stable, model_dir)
            self.assertEqual(selected["epoch"], 21)
            self.assertEqual((model_dir / "best_model.pth").read_bytes(), b"epoch-21")
            (model_dir / "best_model.pth").unlink()
            self.assertIsNone(best_model_selection.materialize_best_model(unstable, model_dir))
            self.assertFalse((model_dir / "best_model.pth").exists())


class TrainingTests(unittest.TestCase):
    def _write_stack(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        mrcfile.write(path, np.asarray(values, dtype=np.float32), overwrite=True)

    def _make_dataset(self, root):
        stack = np.stack([np.zeros((8, 8), dtype=np.float32), np.ones((8, 8), dtype=np.float32)])
        self._write_stack(root / "s1" / "particles.mrcs", stack)
        self._write_stack(root / "s2" / "input.mrcs", stack)
        self._write_stack(root / "s2" / "label.mrcs", stack)
        self._write_stack(root / "s3" / "noise.mrcs", stack)
        self._write_stack(root / "val" / "input.mrcs", stack)
        self._write_stack(root / "val" / "label.mrcs", stack)

    def test_missing_or_mismatched_training_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Missing required training file"):
                train.validate_training_data(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_dataset(root)
            self._write_stack(root / "s2" / "label.mrcs", np.zeros((1, 8, 8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "same number"):
                train.validate_training_data(root)

    def test_scheduler_steps_once_per_epoch_with_tiny_model(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv2d(1, 1, 1)

            def forward(self, value):
                return self.layer(value)

        class CountingScheduler:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.steps = 0
                self.__class__.instances.append(self)

            def step(self):
                self.steps += 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            model_dir = root / "models"
            log_dir = root / "logs"
            model_dir.mkdir()
            (model_dir / "best_model.pth").write_bytes(b"stale")
            self._make_dataset(data_dir)
            with mock.patch.object(train.unet2d, "UDenoiseNet", TinyModel), \
                    mock.patch.object(train, "StepLR", CountingScheduler), \
                    mock.patch.object(train.torch, "save", side_effect=lambda _value, path: Path(path).write_bytes(b"model")), \
                    mock.patch.object(train.torch.cuda, "is_available", return_value=False):
                train.main(data_dir, model_dir, "0", 1, log_dir, epochs=2, seed=42)
            self.assertFalse((model_dir / "best_model.pth").exists())
        self.assertEqual(CountingScheduler.instances[-1].steps, 2)

    def test_train_cli_defaults_to_101_epochs(self):
        parser = train.build_parser()
        args = parser.parse_args(["-i", "input", "-o", "output"])
        self.assertEqual(args.epochs, 101)

    def test_nonfinite_training_loss_fails_instead_of_succeeding(self):
        class NaNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, value):
                return value * self.weight * float("nan")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            model_dir = root / "models"
            log_dir = root / "logs"
            model_dir.mkdir()
            self._make_dataset(data_dir)
            with mock.patch.object(train.unet2d, "UDenoiseNet", NaNModel), \
                    mock.patch.object(train.torch, "save", side_effect=lambda _value, path: Path(path).write_bytes(b"unexpected")), \
                    mock.patch.object(train.torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    train.main(data_dir, model_dir, "0", 1, log_dir, epochs=1, seed=42)
            self.assertFalse((model_dir / "1.pth").exists())


if __name__ == "__main__":
    unittest.main()
