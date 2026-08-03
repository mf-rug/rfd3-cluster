"""Cluster-side entry point: `python -m rfd3_tools <subcommand>`.

The package is shipped inside every job directory, so the job script can call
these without anything being installed on the cluster.

Subcommands:
  log <job_dir>   parse slurm-*.out into a clean summary (run from the exit trap)
  mpnn-config     build MPNN's batch --config_json from a directory of backbones
"""

from __future__ import annotations

import sys
from typing import List, Optional

USAGE = """usage: python -m rfd3_tools <command> [args]

commands:
  log <job_dir> [--epilog-marker STR]   write <job>_<STATUS>_<id>.log
  mpnn-config --structures DIR --out F --out-directory D [...]
"""


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]

    if command == "log":
        from .logparse import summarize

        job_dir = rest[0] if rest and not rest[0].startswith("-") else "."
        marker = ""
        if "--epilog-marker" in rest:
            idx = rest.index("--epilog-marker")
            if idx + 1 < len(rest):
                marker = rest[idx + 1]
        path = summarize(job_dir, marker)
        if path:
            print(f"wrote {path}")
        return 0

    if command in ("mpnn-config", "mpnn_config"):
        from .mpnn_config import main as mpnn_main

        return mpnn_main(rest)

    print(f"unknown command: {command}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
