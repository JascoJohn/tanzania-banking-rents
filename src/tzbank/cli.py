"""Command line: ``python -m tzbank run <stage>`` and ``python -m tzbank info`` (CLAUDE.md section 4)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from tzbank import __version__


def _info() -> int:
    from tzbank.config import find_root, path, settings, valuation_dates
    from tzbank.ingest.manifest import verify_manifest
    from tzbank.io import git_commit, sha256

    root = find_root()
    s = settings(root)
    print(f"tzbank {__version__}  |  Python {sys.version.split()[0]}")
    print(f"repository root : {root}")
    print(f"git commit      : {git_commit(root)}")
    print("valuation dates : " + ", ".join(f"{k} {v.date()}" for k, v in valuation_dates(root).items()))
    print(f"random seed     : {s['random_seed']}")
    print("inputs (sha256):")
    for key, rel in s["inputs"].items():
        f = root / rel
        h = sha256(f)[:16] + "..." if f.exists() and f.is_file() else "MISSING"
        print(f"  {key:<24} {rel}  {h}")
    if path("manifest", root).exists():
        st = verify_manifest(root)["status"].value_counts().to_dict()
        print(f"MANIFEST check  : {json.dumps(st)}")
    print("config:")
    print(json.dumps({k: v for k, v in s.items() if k not in {"paths", "inputs"}}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    from tzbank.pipeline import STAGES

    p = argparse.ArgumentParser(prog="python -m tzbank", description="tanzania-banking-rents pipeline")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a pipeline stage")
    r.add_argument("stage", choices=[*STAGES, "all"])
    r.add_argument("--skip-slow", action="store_true",
                   help="reuse cached PDF-stage outputs (inspection, manual list, code extraction) when their inputs "
                        "are unchanged (input fingerprint)")
    sub.add_parser("info", help="print config, valuation dates and input hashes")
    mp = sub.add_parser("manifest", help="(re)write data/raw/MANIFEST.csv after adding or updating a raw file")
    mp.add_argument("--note", nargs=2, action="append", metavar=("FILE", "DESCRIPTION"), default=[],
                    help="description of the change for a file (path relative to data/raw)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "info":
        return _info()
    if args.cmd == "manifest":
        from tzbank.ingest.manifest import write_manifest

        print(write_manifest(notes=dict(args.note)))
        return 0
    from tzbank.pipeline import StageFailed, run_stage

    try:
        files = run_stage(args.stage, skip_slow=args.skip_slow)
    except StageFailed as exc:
        logging.getLogger("tzbank").error("FAILED: %s", exc)
        return 1
    print(f"done: {len(files)} files written")
    return 0
