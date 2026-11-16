"""Run the 'usd' stage (chapter 12): `python -m tzbank run usd`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "usd"]))
