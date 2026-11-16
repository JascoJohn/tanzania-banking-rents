"""Run the 'review' stage: equivalent to `python -m tzbank run review`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "review"]))
