"""Turn a job's `slurm-<id>.out` into a short human-readable summary.

Runs on the cluster from the job script's exit trap, so it must be defensive:
it parses whatever is present and never raises into the job.  The output is a
single `<jobname>_<STATUS>_<jobid>.log` next to the Slurm file, which is what
you actually read after pulling results back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STRUCTURE_SUFFIXES = (".pdb", ".cif")
EXCLUDE_MARKERS = ("_traj", "_trajectory", "_noisy", "_denoised")

ERROR_PATTERNS = (
    re.compile(r"^Traceback \(most recent call last\)", re.M),
    re.compile(r"^\s*(\w*Error|Exception|RuntimeError):", re.M),
    re.compile(r"CUDA out of memory", re.M),
    re.compile(r"slurmstepd:.*(CANCELLED|OOM|TIME LIMIT)", re.M),
    re.compile(r"driver .* too old|no kernel image is available", re.M),
)


def _latest_slurm_out(job_dir: Path) -> Optional[Path]:
    files = sorted(job_dir.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _job_id(slurm_out: Path) -> str:
    m = re.search(r"slurm-(\d+)\.out", slurm_out.name)
    return m.group(1) if m else "unknown"


def _count_structures(root: Path) -> Tuple[int, List[str]]:
    if not root.is_dir():
        return 0, []
    names = []
    for suffix in STRUCTURE_SUFFIXES:
        for p in sorted(root.rglob(f"*{suffix}")):
            if any(marker in p.stem for marker in EXCLUDE_MARKERS):
                continue
            names.append(p.name)
    return len(names), names


def _design_metadata(root: Path) -> List[Dict[str, Any]]:
    """RFD3 writes a sidecar JSON per design; pull the interesting bits.

    Key names differ between releases, so every lookup is best-effort.
    """
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        spec = data.get("input_specification") or {}
        entry: Dict[str, Any] = {"name": path.stem}
        for key in ("length", "contig", "partial_t"):
            if isinstance(spec, dict) and spec.get(key) is not None:
                entry[key] = spec[key]
        seq = data.get("sequence")
        if isinstance(seq, str) and seq:
            entry["seq_len"] = len(seq)
        out.append(entry)
    return out


def _mpnn_summary(root: Path) -> Dict[str, Any]:
    """Sequence count and recovery range from MPNN's pooled FASTA files."""
    if not root.is_dir():
        return {}
    n_seqs = 0
    recoveries: List[float] = []
    for fasta in sorted(root.rglob("*.fa")):
        try:
            text = fasta.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.startswith(">"):
                continue
            n_seqs += 1
            m = re.search(r"sequence_recovery=([0-9.]+)", line)
            if m:
                try:
                    recoveries.append(float(m.group(1)))
                except ValueError:
                    pass
    summary: Dict[str, Any] = {"n_sequences": n_seqs}
    if recoveries:
        summary["recovery_min"] = min(recoveries)
        summary["recovery_max"] = max(recoveries)
        summary["recovery_mean"] = sum(recoveries) / len(recoveries)
    return summary


def _extract_errors(text: str, max_blocks: int = 5, context: int = 12) -> List[str]:
    """Pull out distinct error blocks.

    Several patterns match the same failure (a traceback header *and* its
    trailing `SomeError:` line), so matches are deduplicated by line range,
    and each block stops at the job script's `====` banner or a blank line
    rather than running on into unrelated output.
    """
    lines = text.splitlines()
    starts: List[int] = []
    for pattern in ERROR_PATTERNS:
        for m in pattern.finditer(text):
            starts.append(text[: m.start()].count("\n"))
    starts.sort()

    blocks: List[str] = []
    covered_until = -1
    for start in starts:
        if start <= covered_until:
            continue  # already inside a block we captured
        chunk: List[str] = []
        for line in lines[start : start + context]:
            if line.startswith("====") or (chunk and not line.strip()):
                break
            chunk.append(line)
        covered_until = start + len(chunk) - 1
        block = "\n".join(chunk).strip()
        if block and block not in blocks:
            blocks.append(block)
        if len(blocks) >= max_blocks:
            break
    return blocks


def _strip_epilog(text: str, marker: str) -> str:
    if marker and marker in text:
        return text.split(marker, 1)[0]
    return text


def summarize(job_dir: str | Path, epilog_marker: str = "") -> Optional[Path]:
    """Write the summary log; return its path (or None if nothing to do)."""
    job_dir = Path(job_dir)
    slurm_out = _latest_slurm_out(job_dir)
    if slurm_out is None:
        return None

    try:
        raw = slurm_out.read_text(errors="replace")
    except OSError:
        return None
    text = _strip_epilog(raw, epilog_marker)

    job_id = _job_id(slurm_out)
    ok = "Job completed successfully" in text
    status = "COMPLETED" if ok else "FAILED"

    job_name = job_dir.name
    m = re.search(r"JobName=(\S+)", text)
    if m:
        job_name = m.group(1)

    rfd3_dir = job_dir / "output" / "rfd3"
    mpnn_dir = job_dir / "output" / "mpnn"
    n_designs, design_names = _count_structures(rfd3_dir)
    metadata = _design_metadata(rfd3_dir)
    mpnn = _mpnn_summary(mpnn_dir)

    lines: List[str] = []
    lines.append(f"job      : {job_name}")
    lines.append(f"job id   : {job_id}")
    lines.append(f"status   : {status}")

    for label, pattern in (
        ("runtime  ", r"Job (?:completed successfully|FAILED \(exit code \d+\)) (?:in|after) (\d+)s"),
        ("node     ", r"NodeList=(\S+)"),
        ("partition", r"Partition=(\S+)"),
    ):
        m = re.search(pattern, text)
        if m:
            value = m.group(1)
            if label.strip() == "runtime":
                value = f"{int(value)}s ({int(value) / 60:.1f} min)"
            lines.append(f"{label}: {value}")

    gpus = sorted(set(re.findall(r"\|\s+\d+\s+(NVIDIA [\w\- ]+?)\s+(?:On|Off)", text)))
    if gpus:
        lines.append(f"gpu      : {', '.join(gpus)}")

    lines.append("")
    lines.append(f"designs  : {n_designs} structure(s) in output/rfd3")
    for entry in metadata[:20]:
        bits = [f"{k}={v}" for k, v in entry.items() if k != "name"]
        lines.append(f"  - {entry['name']}" + (f"  ({', '.join(bits)})" if bits else ""))
    if len(metadata) > 20:
        lines.append(f"  ... and {len(metadata) - 20} more")
    if not metadata and design_names:
        for name in design_names[:20]:
            lines.append(f"  - {name}")

    if mpnn:
        lines.append("")
        lines.append(f"sequences: {mpnn.get('n_sequences', 0)} in output/mpnn")
        if "recovery_mean" in mpnn:
            lines.append(
                f"  recovery: mean {mpnn['recovery_mean']:.3f} "
                f"(min {mpnn['recovery_min']:.3f}, max {mpnn['recovery_max']:.3f})"
            )

    errors = _extract_errors(text)
    if errors:
        lines.append("")
        lines.append("errors:")
        for block in errors:
            lines.append("  " + block.replace("\n", "\n  "))
            lines.append("")

    out_path = job_dir / f"{job_name}_{status}_{job_id}.log"
    try:
        out_path.write_text("\n".join(lines) + "\n")
    except OSError:
        return None
    return out_path
