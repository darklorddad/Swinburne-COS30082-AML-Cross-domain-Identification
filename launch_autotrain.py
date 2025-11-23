import uvicorn
import sys
import os
import time

if __name__ == "__main__":
    # This script relies on PYTHONPATH to find the 'autotrain' package.
    # It no longer takes command-line arguments.
    try:
        from autotrain.app.app import app
    except ImportError:
        print("Error: Could not import 'autotrain.app.app'.", file=sys.stderr)
        print("Please ensure that the parent directory of the 'autotrain' package is in your PYTHONPATH.", file=sys.stderr)
        time.sleep(10)
        sys.exit(1)

    start_ts = time.time()
    try:
        uvicorn.run(app, host="localhost", port=7861)
    except Exception as e:
        print(f"Error running uvicorn: {e}")
    
    if time.time() - start_ts < 5:
        print("Process exited too quickly. Pausing for 10 seconds...")
        time.sleep(10)
