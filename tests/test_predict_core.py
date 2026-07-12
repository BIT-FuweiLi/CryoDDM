import importlib
from pathlib import Path
import sys
import unittest

import numpy as np


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

    def test_predict_cli_default_batch_is_16_and_aliases_work(self):
        parser = predict.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.batch_size, 16)
        self.assertEqual(parser.parse_args(["--particle_diamater", "180"]).particle_diameter, 180)
        self.assertEqual(parser.parse_args(["--particle_diameter", "200"]).particle_diameter, 200)

    def test_model_is_switched_to_eval_mode(self):
        source = (BACKWARD_DIR / "predict.py").read_text(encoding="utf-8")
        self.assertIn("model.eval()", source)


if __name__ == "__main__":
    unittest.main()
