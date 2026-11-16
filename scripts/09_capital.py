"""Run the 'capital' stage (chapter 7): equivalent to `python -m tzbank run capital`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "capital"]))
