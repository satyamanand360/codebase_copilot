"""Run generated pytest code in a separate process.

This is process isolation, not a security sandbox: the code runs on the host
with access to its filesystem and network. Mitigations here are a temp
working directory, a timeout, and an environment stripped of secrets such as
API keys. For untrusted use, run it in a container instead, e.g.
``docker run --rm --network none --memory 512m --cpus 1 -v <dir>:/work:ro ...``.
That is why TestAgent only calls this when ENABLE_TEST_SANDBOX=1.
"""
import os
import subprocess
import sys
import tempfile

SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR")


class SandboxExecutor:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def execute_test(self, test_code: str) -> dict:
        """Returns {"passed": bool, "output": str}."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_generated.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_code)

            env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
            env["PYTHONPATH"] = self.project_root  # so tests can import project modules

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-q", "-p", "no:cacheprovider"],
                    cwd=temp_dir, capture_output=True, text=True, env=env, timeout=self.timeout,
                )
                return {"passed": result.returncode == 0, "output": (result.stdout + "\n" + result.stderr).strip()}
            except subprocess.TimeoutExpired:
                return {"passed": False, "output": f"Execution timed out after {self.timeout} seconds."}
            except Exception as e:
                return {"passed": False, "output": f"Execution failed to start: {e}"}
