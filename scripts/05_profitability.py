"""Run the 'profitability' stage: equivalent to `python -m tzbank run profitability`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "profitability"]))
