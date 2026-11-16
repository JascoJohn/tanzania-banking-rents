"""Run the 'ingest' stage: equivalent to `python -m tzbank run ingest`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "ingest"]))
