import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import mrcfile
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BACKWARD_DIR = ROOT / "core" / "backward"
sys.path.insert(0, str(BACKWARD_DIR))
predict = importlib.import_module("predict")


class PredictTests(unittest.TestCase):
    def test_constant_normalization_is_finite_zero(self):
        image = np.full((8, 8), 3.0, dtype=np.float32)
        result = predict.normal(image)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, np.zeros_like(image))

    def test_nonfinite_normalization_input_is_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "NaN or Inf"):
                predict.normal(np.array([[0.0, bad]], dtype=np.float32))

    def test_memory_batch_limit_uses_20_percent_and_cap_64(self):
        self.assertEqual(predict.calculate_mrc_batch_size(1000, 100, 100), 2)
        self.assertEqual(predict.calculate_mrc_batch_size(10**9, 100, 100), 64)
        self.assertEqual(predict.calculate_mrc_batch_size(1, 100, 100), 1)
        self.assertEqual(predict.calculate_mrc_batch_size(10**9, 100, 3), 3)

    def test_memory_batches_use_each_micrograph_size(self):
        batches = predict.plan_mrc_batches(
            ["small.mrc", "large.mrc", "tail.mrc"],
            available_memory=1000,
            size_lookup={"small.mrc": 100, "large.mrc": 900, "tail.mrc": 100}.__getitem__,
        )
        self.assertEqual(batches, [["small.mrc"], ["large.mrc"], ["tail.mrc"]])

    def test_predict_cli_default_batch_is_16_and_aliases_work(self):
        parser = predict.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.batch_size, 16)
        self.assertEqual(parser.parse_args(["--particle_diamater", "180"]).particle_diameter, 180)
        self.assertEqual(parser.parse_args(["--particle_diameter", "200"]).particle_diameter, 200)

    def test_mixed_micrograph_shapes_are_preserved_and_model_is_eval(self):
        class IdentityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.eval_called = False

            def eval(self):
                self.eval_called = True
                return super().eval()

            def forward(self, value):
                return value

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            output = root / "output"
            logs = root / "logs"
            raw.mkdir()
            output.mkdir()
            logs.mkdir()
            mrcfile.write(raw / "a.mrc", np.arange(64, dtype=np.float32).reshape(8, 8), overwrite=True)
            mrcfile.write(raw / "b.mrc", np.arange(120, dtype=np.float32).reshape(12, 10), overwrite=True)
            model = IdentityModel()
            with mock.patch.object(predict.torch, "load", return_value=model), \
                    mock.patch.object(predict.torch, "device", return_value=torch.device("cpu")), \
                    mock.patch.object(predict.psutil, "virtual_memory", return_value=SimpleNamespace(available=1000)):
                predict.main(raw, output, root / "model.pth", "0", 256, logs, batch_size=2)
            self.assertTrue(model.eval_called)
            self.assertEqual(mrcfile.read(output / "a.mrc").shape, (8, 8))
            self.assertEqual(mrcfile.read(output / "b.mrc").shape, (12, 10))

    def test_padding_leaves_a_centre_for_every_window_size(self):
        self.assertEqual(predict.prediction_padding(128), 32)
        self.assertEqual(predict.prediction_padding(256), 64)
        self.assertEqual(predict.prediction_padding(384), 64)

    def test_small_particles_128_window_predicts_without_error(self):
        # particle diameter <= 85 px gives a 128 px window; padding 64 used to divide by zero
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, output, logs = root / "raw", root / "output", root / "logs"
            for folder in (raw, output, logs):
                folder.mkdir()
            image = np.random.default_rng(0).normal(size=(300, 200)).astype(np.float32)
            mrcfile.write(raw / "a.mrc", image, overwrite=True)
            identity = torch.nn.Identity()
            with mock.patch.object(predict.torch, "load", return_value=identity), \
                    mock.patch.object(predict.torch, "device", return_value=torch.device("cpu")):
                predict.main(raw, output, root / "model.pth", "0", 128, logs, batch_size=4)
            result = mrcfile.read(output / "a.mrc")
            self.assertEqual(result.shape, (300, 200))
            np.testing.assert_allclose(result, predict.normal(image), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
