"""Run the 'synthesis' stage (Part II synthesis): equivalent to `python -m tzbank run synthesis`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "synthesis"]))
