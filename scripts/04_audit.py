"""Run the 'audit' stage: equivalent to `python -m tzbank run audit`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "audit"]))
