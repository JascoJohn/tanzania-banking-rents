"""Run the 'persistence' stage (chapter 5): equivalent to `python -m tzbank run persistence`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "persistence"]))
