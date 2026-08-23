import subprocess
import tempfile
import os

class SandboxExecutor:
    def __init__(self):
        # We assume local execution environment using pytest for the MVP.
        # In a real environment, we would use docker run --rm ...
        pass
        
    def execute_test(self, test_code: str) -> dict:
        """
        Executes the given test code in a temporary file using pytest.
        Returns a dict with {"passed": bool, "output": str}.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # We need to write the test code to a file starting with test_
            test_file_path = os.path.join(temp_dir, "test_generated.py")
            
            # Since the dummy repo files are in the main project directory,
            # we need to ensure the test can import them. The easiest way for our
            # local mock sandbox is to set PYTHONPATH to the project root.
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write(test_code)
                
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root
            
            try:
                # Run pytest on the temporary file
                result = subprocess.run(
                    ["pytest", test_file_path, "-v"],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10 # Prevent infinite loops
                )
                
                passed = (result.returncode == 0)
                output = result.stdout + "\n" + result.stderr
                return {"passed": passed, "output": output}
                
            except subprocess.TimeoutExpired:
                return {"passed": False, "output": "Execution timed out after 10 seconds."}
            except Exception as e:
                return {"passed": False, "output": f"Execution failed to start: {str(e)}"}
