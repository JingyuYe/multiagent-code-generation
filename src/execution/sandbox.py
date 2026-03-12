import sys
import contextlib
import io
import traceback
from typing import List, Tuple

def run_tests_in_sandbox(code: str, tests: List[str]) -> Tuple[bool, str]:
    """
    A naive and insecure execution sandbox. In a real environment, 
    use Docker or a strict subprocess with timeouts.
    Returns:
        bool: True if all tests passed, False otherwise.
        str: Aggregated execution traces, stdout, and tracebacks.
    """
    output = io.StringIO()
    passed = True
    trace = ""
    
    # We combine the code and the tests
    full_code = code + "\n" + "\n".join(tests)
    
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            # We use an empty dict for global/local to somewhat isolate
            exec(full_code, {})
        except Exception as e:
            passed = False
            traceback.print_exc(file=output)
            
    trace = output.getvalue()
    
    if passed:
        trace += "\nAll tests passed successfully."
    else:
        trace += "\nTests failed during execution."
        
    return passed, trace
