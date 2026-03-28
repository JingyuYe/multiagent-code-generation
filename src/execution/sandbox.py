import sys
import contextlib
import io
import traceback
import subprocess
import tempfile
import os
from typing import List, Tuple

def run_tests_in_sandbox(code: str, tests: List[str]) -> Tuple[bool, str]:
    """
    A naive and insecure execution sandbox. In a real environment, 
    use Docker or a strict subprocess with timeouts.
    Returns:
        bool: True if all tests passed, False otherwise.
        str: Aggregated execution traces, stdout, and tracebacks.
    """
    full_code = code + "\n\n" + "\n".join(tests)
    passed = False
    trace = ""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name
        
    try:
        # Run in a subprocess to gracefully catch infinite loops and segfaults
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            passed = True
            trace = result.stdout
        else:
            trace = result.stdout + "\n" + result.stderr
            
    except subprocess.TimeoutExpired:
        trace = "Execution timed out (possible infinite loop)."
    except Exception as e:
        trace = f"Uncaught execution error: {e}"
    finally:
        os.remove(temp_path)
            
    if passed:
        trace += "\nAll tests passed successfully."
    else:
        trace += "\nTests failed during execution."
        
    return passed, trace
