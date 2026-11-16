"""Run the 'valuation_results' stage (chapters 9-11): `python -m tzbank run valuation_results`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "valuation_results"]))
