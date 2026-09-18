import os
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "core" / "cryo2star" / "pipelineV2.sh"
CLEAN_PATH = ROOT / "core" / "cryo2star" / "clean.py"


class Cryo2StarPipelineTests(unittest.TestCase):
    def test_clean_handles_repeated_whitespace(self):
        spec = importlib.util.spec_from_file_location("cryoddm_clean", CLEAN_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cleaned = module.clean_filenames("000001@J12/extract/prefix_stack.mrcs    J12/imported/prefix_micrograph.mrc   1 2\n")
        self.assertEqual(cleaned, "stack.mrcs micrograph.mrc 1 2\n")

    def _prepare_job(self, root):
        job = root / "J100"
        job.mkdir()
        (job / "J100_001_particles.cs").write_bytes(b"particles")
        (job / "J100_passthrough_particles.cs").write_bytes(b"passthrough")
        return job

    def _write_executable(self, path, body):
        path.write_text("#!/bin/bash\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)

    def test_converter_failure_propagates_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self._prepare_job(root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            self._write_executable(bin_dir / "csparc2star.py", "exit 7\n")
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
            result = subprocess.run(["bash", str(PIPELINE), str(job), str(root / "out"), "100"], env=env)
            self.assertNotEqual(result.returncode, 0)

    def _assert_invalid_num_projects(self, invalid):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self._prepare_job(root)
            result = subprocess.run(
                ["bash", str(PIPELINE), str(job), str(root / "out"), "100", invalid],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("positive integer", result.stderr)

    def test_num_projects_rejects_zero(self):
        self._assert_invalid_num_projects("0")

    def test_num_projects_rejects_non_numeric_value(self):
        self._assert_invalid_num_projects("abc")

    def test_successful_converter_clean_and_invert_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self._prepare_job(root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            self._write_executable(
                bin_dir / "csparc2star.py",
                "printf 'data_\\n' > \"${@: -1}\"\n",
            )
            self._write_executable(
                bin_dir / "fake-python",
                """
                case "$1" in
                  *clean.py) printf 'header\\n%.0s' {1..13} > cleaned_particles_relion.star ;;
                  *invert_coordinateY.py) touch "$3" ;;
                  *) exit 9 ;;
                esac
                """,
            )
            env = dict(
                os.environ,
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                PYTHON=str(bin_dir / "fake-python"),
            )
            result = subprocess.run(["bash", str(PIPELINE), str(job), str(root / "out"), "100"], env=env)
            self.assertEqual(result.returncode, 0)
            self.assertTrue((root / "out" / "class0" / "invert.star").exists())

    def test_partial_class_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self._prepare_job(root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            self._write_executable(bin_dir / "csparc2star.py", "touch \"${@: -1}\"\n")
            self._write_executable(
                bin_dir / "fake-python",
                """
                case "$1" in
                  *clean.py) printf 'header\\n%.0s' {1..13} > cleaned_particles_relion.star ;;
                  *invert_coordinateY.py) touch "$3" ;;
                esac
                """,
            )
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=str(bin_dir / "fake-python"))
            result = subprocess.run(["bash", str(PIPELINE), str(job), str(root / "out"), "100", "2"], env=env)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
