import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "core" / "cryo2star" / "pipelineV2.sh"


class Cryo2StarPipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
