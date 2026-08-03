"""Build MPNN's batch config JSON from a directory of backbones.

MPNN's CLI takes either one `--structure_path`, or `--config_json` pointing at
a file of the shape::

    {
      "model_type": "protein_mpnn",
      "checkpoint_path": "...",
      "is_legacy_weights": true,
      "out_directory": "...",
      "inputs": [ {"structure_path": "...", "name": "...", ...}, ... ]
    }

The batch form matters on a cluster: one process loads the model once and walks
every backbone, instead of paying model-load latency per structure.

This runs *on the cluster*, inside the job, because RFD3's output filenames
don't exist when the job script is generated.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# RFD3 writes **gzipped mmCIF** (`.cif.gz`), never PDB. Plain .cif/.pdb are
# accepted too so this helper also works on hand-staged inputs.
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".cif.gz")
# RFD3 writes diffusion trajectories next to the designs when dump_trajectories
# is on; they are multi-model files and must never be fed to MPNN.
EXCLUDE_MARKERS = ("_traj", "_trajectory", "_noisy", "_denoised")


def _matches_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in STRUCTURE_SUFFIXES)


def _stem(path: Path) -> str:
    """Filename without structure suffix (handles the double .cif.gz)."""
    name = path.name
    for suffix in (".cif.gz", ".cif", ".pdb"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_structures(root: str) -> List[Path]:
    """Collect design structures under `root` (a directory, file, or glob)."""
    path = Path(root)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(p for p in path.rglob("*") if p.is_file())
    else:  # treat as a glob relative to cwd
        candidates = sorted(Path().glob(root))

    out = []
    for p in candidates:
        if not _matches_suffix(p):
            continue
        if any(marker in _stem(p) for marker in EXCLUDE_MARKERS):
            continue
        out.append(p)
    return out


def stage_decompressed(structures: List[Path], dest: str) -> List[Path]:
    """Write plain .cif copies of any .cif.gz into `dest`.

    Whether MPNN reads .cif.gz directly is unverified upstream, so decompress
    rather than gamble a queued job on it. Non-gzipped inputs pass through
    untouched. Existing staged files are reused, keeping the step idempotent
    across resubmissions.
    """
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged: List[Path] = []
    for path in structures:
        if not path.name.lower().endswith(".gz"):
            staged.append(path)
            continue
        target = dest_dir / f"{_stem(path)}.cif"
        if not target.exists() or target.stat().st_size == 0:
            with gzip.open(path, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        staged.append(target)
    return staged


def resolve_checkpoint(
    explicit: Optional[str], ckpt_glob: Optional[str], ckpt_dir: Optional[str]
) -> str:
    """Find the MPNN weights.

    Prefers an explicit path; otherwise globs the job's checkpoint directory
    and anything in $FOUNDRY_CHECKPOINT_DIRS, since weight filenames carry
    version numbers that change between releases.
    """
    if explicit:
        return explicit
    if not ckpt_glob:
        raise SystemExit("need --checkpoint or --checkpoint-glob")

    search: List[Path] = []
    if ckpt_dir:
        search.append(Path(ckpt_dir))
    for entry in (os.environ.get("FOUNDRY_CHECKPOINT_DIRS") or "").split(":"):
        if entry:
            search.append(Path(entry))
    search.append(Path.home() / ".foundry/checkpoints")

    for directory in search:
        if not directory.is_dir():
            continue
        hits = sorted(directory.glob(ckpt_glob))
        if hits:
            # Absolute, so the path stays valid regardless of the job's cwd.
            return str(hits[-1].resolve())  # newest-sorting version wins
    raise SystemExit(
        f"no checkpoint matching {ckpt_glob!r} in: "
        + ", ".join(str(s) for s in search)
    )


def build_config(
    *,
    structures: List[Path],
    model_type: str,
    checkpoint_path: str,
    out_directory: str,
    batch_size: int,
    number_of_batches: int,
    temperature: Optional[float] = None,
    write_structures: bool = True,
    write_fasta: bool = True,
) -> Dict[str, Any]:
    inputs: List[Dict[str, Any]] = []
    for path in structures:
        entry: Dict[str, Any] = {
            "structure_path": str(path.resolve()),
            "name": _stem(path),
            "batch_size": batch_size,
            "number_of_batches": number_of_batches,
        }
        if temperature is not None:
            entry["temperature"] = temperature
        inputs.append(entry)

    return {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        # Required for every checkpoint in Foundry's default registry.
        "is_legacy_weights": True,
        "out_directory": out_directory,
        "write_fasta": write_fasta,
        "write_structures": write_structures,
        "inputs": inputs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rfd3_tools mpnn-config",
        description="Build an MPNN --config_json batch config from a directory "
        "of backbones.",
    )
    ap.add_argument("--structures", required=True,
                    help="Directory, file, or glob of backbone structures.")
    ap.add_argument("--out", required=True, help="Where to write the config JSON.")
    ap.add_argument("--model-type", default="protein_mpnn",
                    choices=["protein_mpnn", "ligand_mpnn"])
    ap.add_argument("--out-directory", required=True,
                    help="MPNN output directory (sequences + designed structures).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--number-of-batches", type=int, default=8)
    ap.add_argument("--checkpoint", default=None, help="Explicit checkpoint path.")
    ap.add_argument("--checkpoint-glob", default=None,
                    help="Glob used to locate the checkpoint (e.g. 'proteinmpnn*.pt').")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Directory searched first for --checkpoint-glob.")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-write-structures", action="store_true")
    ap.add_argument("--gunzip-to", default=None, metavar="DIR",
                    help="Stage decompressed .cif copies of any .cif.gz here "
                         "and point MPNN at those (RFD3 output is gzipped).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    structures = find_structures(args.structures)
    if not structures:
        print(f"error: no structures found under {args.structures!r}", file=sys.stderr)
        return 1

    n_gz = sum(1 for p in structures if p.name.lower().endswith(".gz"))
    if args.gunzip_to and n_gz:
        structures = stage_decompressed(structures, args.gunzip_to)

    checkpoint = resolve_checkpoint(
        args.checkpoint, args.checkpoint_glob, args.checkpoint_dir
    )
    config = build_config(
        structures=structures,
        model_type=args.model_type,
        checkpoint_path=checkpoint,
        out_directory=args.out_directory,
        batch_size=args.batch_size,
        number_of_batches=args.number_of_batches,
        temperature=args.temperature,
        write_structures=not args.no_write_structures,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2))

    if not args.quiet:
        staged = f", {n_gz} decompressed into {args.gunzip_to}" if (args.gunzip_to and n_gz) else ""
        print(
            f"wrote {out_path} -- {len(structures)} structure(s), "
            f"{args.batch_size * args.number_of_batches} sequence(s) each, "
            f"checkpoint {checkpoint}{staged}"
        )
    return 0
