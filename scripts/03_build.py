"""Run the 'build' stage: equivalent to `python -m tzbank run build`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "build"]))
