"""Run the 'market_ext' stage (market behaviour and valuation extensions, prompt 14): `python -m tzbank run market_ext`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "market_ext"]))
