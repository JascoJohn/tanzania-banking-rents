"""Collect DSE daily trading history (D71): `python scripts/19_dse_collect.py initial|update|build [--ticker CRDB]`.

Network step, run by hand (not part of `run all`); the offline build also runs inside the ingest stage.
"""

import sys

from tzbank.ingest.dse_trading_history import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
