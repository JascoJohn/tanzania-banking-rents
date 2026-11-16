"""Run the 'funding' stage (chapter 6): equivalent to `python -m tzbank run funding`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "funding"]))
