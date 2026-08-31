from __future__ import annotations

import argparse
import json
from pathlib import Path

from .binary_io import compare_steps, write_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two C++-schema 3-D step files")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = compare_steps(args.candidate, args.reference)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        write_comparison(report, args.json)
    raise SystemExit(0 if report["file_bitwise_identical"] else 1)


if __name__ == "__main__":
    main()
