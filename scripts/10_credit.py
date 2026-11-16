"""Run the 'credit' stage (chapter 8): equivalent to `python -m tzbank run credit`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "credit"]))
