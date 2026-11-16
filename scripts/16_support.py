"""Run the 'support' stage (report support, prompt 13): `python -m tzbank run support`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "support"]))
