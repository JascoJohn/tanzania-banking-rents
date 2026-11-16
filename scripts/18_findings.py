"""Run the 'findings' stage (findings pack): `python -m tzbank run findings`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "findings"]))
