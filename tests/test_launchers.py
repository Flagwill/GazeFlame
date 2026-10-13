"""Shell launcher regression tests; fake conda/python, no camera or models."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("run_demo.sh", "run.sh", "run_webcam.sh")


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "env path with spaces" / "gaze_flame_rocm"
        self.python = self.prefix / "bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text('#!/bin/bash\nprintf "PY=%s\\n" "$0"\nprintf "ARG=%s\\n" "$@"\n')
        self.python.chmod(0o755)
        self.mock_bin = self.root / "mock_bin"
        self.mock_bin.mkdir()
        conda = self.mock_bin / "conda"
        conda.write_text('#!/bin/bash\nif [[ "${MOCK_NO_CONDA:-}" == 1 ]]; then\n'
                         '  echo "ERROR: conda should not be called" >&2; exit 88\n'
                         'fi\nprintf "%s\\n" "$MOCK_CONDA_LIST"\n')
        conda.chmod(0o755)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("BASH_FUNC_"):
                self.env.pop(key)
        self.env.update(PATH=str(self.mock_bin) + ":" + self.env.get("PATH", ""),
                        CONDA_DEFAULT_ENV="", CONDA_PREFIX="", GAZE_PYTHON="",
                        MOCK_NO_CONDA="", MOCK_CONDA_LIST="")

    def launch(self, script, **variables):
        return subprocess.run(["/bin/bash", str(ROOT / script), "--help"],
                              env={**self.env, **variables}, text=True,
                              capture_output=True, timeout=10)

    def assert_correct_python(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PY={self.python}\n", result.stdout)
        self.assertIn("ARG=--help\n", result.stdout)

    def test_conda_list_markers_and_paths_with_spaces(self):
        p = str(self.prefix)
        for row in (f"gaze_flame_rocm   {p}", f"gaze_flame_rocm * {p}",
                    f"gaze_flame_rocm * + {p}", f"* gaze_flame_rocm {p}",
                    f"gaze_flame_rocm {p} *"):
            for script in LAUNCHERS:
                with self.subTest(script=script, row=row):
                    result = self.launch(script, MOCK_CONDA_LIST="# conda environments:\n" + row)
                    self.assert_correct_python(result)

    def test_active_environment_skips_conda_startup(self):
        for script in LAUNCHERS:
            with self.subTest(script=script):
                result = self.launch(script, CONDA_DEFAULT_ENV="gaze_flame_rocm",
                                     CONDA_PREFIX=str(self.prefix), MOCK_NO_CONDA="1")
                self.assert_correct_python(result)

    def test_explicit_python_override(self):
        for script in LAUNCHERS:
            with self.subTest(script=script):
                result = self.launch(script, GAZE_PYTHON=str(self.python), MOCK_NO_CONDA="1")
                self.assert_correct_python(result)

    def test_missing_environment_fails_clearly(self):
        result = self.launch("run_demo.sh", MOCK_CONDA_LIST="base /opt/conda")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gaze_flame_rocm 不存在", result.stderr)

    def test_nonexecutable_python_fails_clearly(self):
        result = self.launch("run_demo.sh", GAZE_PYTHON=str(self.root / "missing_python"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 不可执行", result.stderr)


if __name__ == "__main__":
    unittest.main()
