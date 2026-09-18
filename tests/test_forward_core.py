import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import mrcfile
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORWARD_DIR = ROOT / "core" / "forward"
sys.path.insert(0, str(FORWARD_DIR))

coordinates = importlib.import_module("coordinates")
forward = importlib.import_module("forward")


class CoordinateTests(unittest.TestCase):
    def test_star_columns_are_read_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "particles.star"
            path.write_text(
                "data_\n\nloop_\n"
                "_rlnCoordinateY #1\n"
                "_rlnMicrographName #2\n"
                "_rlnCoordinateX #3\n"
                "7.0 nested/micrograph.mrc 11.0\n",
                encoding="utf-8",
            )
            records = coordinates.read_coordinate_file(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filename, "micrograph.mrc")
        self.assertEqual((records[0].x, records[0].y), (11.0, 7.0))

    def test_origin_auto_and_explicit_override(self):
        record = coordinates.CoordinateRecord("micrograph.mrc", 11, 7)
        self.assertEqual(coordinates.resolve_coordinate_origin("plain.star", "auto"), "top-left")
        # cs2star's invert.star already undoes pyem's flip, so it is top-left like everything else.
        self.assertEqual(coordinates.resolve_coordinate_origin("INVERT.STAR", "auto"), "top-left")
        self.assertEqual(coordinates.resolve_coordinate_origin("invert.star", "bottom-left"), "bottom-left")
        self.assertEqual(coordinates.normalize_xy(record, (100, 120), "top-left"), (7, 11))
        self.assertEqual(coordinates.normalize_xy(record, (100, 120), "bottom-left"), (93, 11))
        self.assertEqual(coordinates.resolve_coordinate_origin("invert.star", "top-left"), "top-left")


class ForwardTests(unittest.TestCase):
    def _write_mrc(self, path, data):
        mrcfile.write(path, np.asarray(data, dtype=np.float32), overwrite=True)

    def test_noise_coordinate_single_row_supports_multiple_box_sizes(self):
        for shape in (128, 256, 512):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = np.arange((shape + 32) ** 2, dtype=np.float32).reshape(shape + 32, shape + 32)
                self._write_mrc(root / "micrograph.mrc", image)
                coords = root / "noise.txt"
                coords.write_text(f"micrograph.mrc {shape // 2 + 16} {shape // 2 + 16}\n", encoding="utf-8")
                patches = forward.get_noise(coords, root, shape)
                self.assertEqual(patches.shape, (1, shape, shape))

    def test_empty_or_out_of_bounds_coordinates_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_mrc(root / "micrograph.mrc", np.zeros((128, 128), dtype=np.float32))
            coords = root / "coords.txt"
            coords.write_text("micrograph.mrc 1 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No valid particle patches"):
                forward.get_particles_csstyle(root, coords, 128, jitter_fraction=0)

    def test_diffusion_parameters_are_validated(self):
        invalid = [
            dict(beta=0, total_steps=5, start=2),
            dict(beta=0.2, total_steps=5, start=2),
            dict(beta=0.1, total_steps=1, start=1),
            dict(beta=0.1, total_steps=5, start=0),
            dict(beta=0.1, total_steps=5, start=5),
            dict(beta=np.nan, total_steps=5, start=2),
            dict(beta=np.inf, total_steps=5, start=2),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                forward.validate_diffusion_params(**kwargs)

    def test_saved_pairs_keep_source_code4_cleaner_to_noisier_layout(self):
        clean = np.ones((2, 8, 8), dtype=np.float32)
        noise = np.zeros_like(clean)
        states = forward.generate_diffusion_states(clean, noise, beta=0.1, total_steps=5)
        train_input, train_label, val_input, val_label = forward.build_training_pairs(states, start=2)

        self.assertEqual(train_input.shape[0], 6)
        self.assertEqual(train_label.shape, train_input.shape)
        np.testing.assert_allclose(train_input.reshape(3, 2, 8, 8), [states[1], states[2], states[3]])
        np.testing.assert_allclose(train_label.reshape(3, 2, 8, 8), [states[2], states[3], states[4]])
        np.testing.assert_allclose(val_input, states[4])
        np.testing.assert_allclose(val_label, states[5])
        for input_patch, label_patch in zip(train_input, train_label):
            self.assertLess(np.mean((input_patch - clean[0]) ** 2), np.mean((label_patch - clean[0]) ** 2))
        self.assertLess(np.mean((val_input - clean[0]) ** 2), np.mean((val_label - clean[0]) ** 2))

    def test_seed_makes_coordinate_jitter_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
            self._write_mrc(root / "micrograph.mrc", image)
            coords = root / "coords.txt"
            coords.write_text("micrograph.mrc 128 128\n", encoding="utf-8")
            forward.set_random_seed(42)
            first = forward.get_particles_csstyle(root, coords, 64)
            forward.set_random_seed(42)
            second = forward.get_particles_csstyle(root, coords, 64)
            np.testing.assert_array_equal(first, second)

    def test_micrograph_lookup_handles_cryosparc_uid_prefix_both_ways(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain"
            prefixed = Path(tmp) / "prefixed"
            plain.mkdir()
            prefixed.mkdir()
            (plain / "FoilHole_1.mrc").touch()
            (plain / "0001_mic.mrc").touch()
            (prefixed / "006642101566427281036_FoilHole_1.mrc").touch()
            cases = [
                # CryoDDM-picked coordinates: name as on disk
                (plain, "FoilHole_1.mrc", "FoilHole_1.mrc"),
                # imported cryoSPARC STAR, micrographs stored without the UID
                (plain, "J2/imported/006642101566427281036_FoilHole_1.mrc", "FoilHole_1.mrc"),
                # cleaned STAR without UID, micrographs taken from cryoSPARC's imported folder
                (prefixed, "FoilHole_1.mrc", "006642101566427281036_FoilHole_1.mrc"),
                # a micrograph whose own name starts with digits is used as-is
                (plain, "0001_mic.mrc", "0001_mic.mrc"),
            ]
            for folder, name, expected in cases:
                with self.subTest(name=name):
                    self.assertEqual(Path(forward.resolve_micrograph_path(folder, name)).name, expected)
            with self.assertRaises(FileNotFoundError):
                forward.resolve_micrograph_path(plain, "missing.mrc")

    def test_particles_from_uid_prefixed_star_are_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
            self._write_mrc(root / "FoilHole_1.mrc", image)
            coords = root / "particle_selected.star"
            coords.write_text("J2/imported/006642101566427281036_FoilHole_1.mrc 128 100\n", encoding="utf-8")
            with mock.patch.object(forward.np.random, "randint", return_value=0):
                patches = forward.get_particles_csstyle(root, coords, 64, coord_origin="bottom-left")
            self.assertEqual(patches.shape, (1, 64, 64))
            self.assertEqual(patches[0, 32, 32], image[256 - 100, 128])

    def test_forward_cli_keeps_legacy_spelling(self):
        parser = forward.build_parser()
        legacy = parser.parse_args(["--particle_diamater", "180"])
        corrected = parser.parse_args(["--particle_diameter", "220"])
        short = parser.parse_args(["-pd", "256"])
        self.assertEqual(legacy.particle_diameter, 180)
        self.assertEqual(corrected.particle_diameter, 220)
        self.assertEqual(short.particle_diameter, 256)
        self.assertEqual(parser.parse_args([]).seed, 42)


if __name__ == "__main__":
    unittest.main()
