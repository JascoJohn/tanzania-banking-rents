"""Run the 'excel_model' stage (Excel companion model, recalculated and checked): `python -m tzbank run excel_model`."""

import sys

from tzbank.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "excel_model"]))
