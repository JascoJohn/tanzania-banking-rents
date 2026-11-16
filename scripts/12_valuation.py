"""Run the 'valuation' stage (prompt 09a dry run and proposed register): `python -m tzbank run valuation`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "valuation"]))
