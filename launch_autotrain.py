import uvicorn
import sys
import os

if __name__ == "__main__":
    # This script relies on PYTHONPATH to find the 'autotrain' package.
    # It no longer takes command-line arguments.
    try:
        from autotrain.app.app import app
    except ImportError:
        print("Error: Could not import 'autotrain.app.app'.", file=sys.stderr)
        print("Please ensure that the parent directory of the 'autotrain' package is in your PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

    uvicorn.run(app, host="localhost", port=7861)
