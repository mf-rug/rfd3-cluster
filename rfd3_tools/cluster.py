"""Cluster configuration for rfd3-setup-job.

Reads ``~/.config/rfd3-setup/config.yaml``, a multi-cluster registry of the
shape::

    default_cluster: habrok
    clusters:
      habrok:
        ssh_target: hpc
        python_module: ...
        venv: {path: /scratch/{user}/venvs/rfd3}
        checkpoint_dir: /scratch/{user}/foundry/checkpoints
        jobs_dir: /scratch/{user}/rfd3_jobs
        gpu_tiers: [...]
        partitions: [...]

Nothing cluster-specific is hardcoded in the Python: adding a cluster is a
config edit, selected with ``--cluster NAME``.

``{user}`` in any path is replaced with the *cluster* username, resolved from
the cluster's ``ssh_target`` via ``ssh -G`` (which reads ~/.ssh/config without
connecting).  Resolution is skipped entirely when no path needs it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(
    os.environ.get("RFD3_SETUP_CONFIG", Path.home() / ".config/rfd3-setup/config.yaml")
)

# Written on first run / `--init`.  Paths match HANDOFF.md; the GPU tiers and
# the walltime constant are PROVISIONAL until the cluster install reports back
# real timings (see HANDOFF.md §4.3).
DEFAULT_CONFIG = """\
# rfd3-setup cluster configuration (multi-cluster).
# Select a cluster with `rfd3-setup-job --cluster NAME`; without it,
# `default_cluster` is used. Machine-local: holds your paths/accounts.
#
# {user} is replaced with the cluster username (resolved via `ssh -G <ssh_target>`).

default_cluster: habrok

clusters:

  # ---- RUG / Habrok ------------------------------------------------------
  habrok:
    ssh_target: hpc
    python_module: Python/3.11.5-GCCcore-13.2.0
    venv:
      path: /scratch/{user}/venvs/rfd3
    checkpoint_dir: /scratch/{user}/foundry/checkpoints
    jobs_dir: /scratch/{user}/rfd3_jobs
    epilog_marker: "Habrok Cluster"
    # Tier chosen by the longest design length in the job (drives VRAM).
    gpu_tiers:
      - {max_length: 300,     gpu_sbatch: l40s:1,         mem: 32GB}
      - {max_length: 700,     gpu_sbatch: a100:1,         mem: 48GB}
      - {max_length: 9999999, gpu_sbatch: rtx_pro_6000:1, mem: 64GB, warn: true}
    partitions:
      - {name: gpushort,  max_hours: 4,  gpus: [v100:1, a100:1, l40s:1, rtx_pro_6000:1]}
      - {name: gpumedium, max_hours: 24, gpus: [v100:1, a100:1, rtx_pro_6000:1]}
      - {name: gpulong,   max_hours: 72, gpus: [v100:1, a100:1, rtx_pro_6000:1]}

  # ---- SURF / Snellius ---------------------------------------------------
  # Replace <PROJECT> with your project allocation id (e.g. prjs1234); these
  # paths are per-allocation, so they must be edited before first use.
  snellius:
    ssh_target: ss
    # Lmod year-stack must precede the module (one `module load` line).
    python_module: "2024 Python/3.12.3-GCCcore-13.3.0"
    venv:
      path: /projects/<PROJECT>/rfd3_venv
    checkpoint_dir: /projects/<PROJECT>/foundry/checkpoints
    jobs_dir: /projects/<PROJECT>/rfd3_jobs
    epilog_marker: ""
    gpu_tiers:
      - {max_length: 700,     gpu_sbatch: a100:1, mem: 48GB}
      - {max_length: 9999999, gpu_sbatch: h100:1, mem: 64GB}
    partitions:
      - {name: gpu_a100, max_hours: 120, gpus: [a100:1]}
      - {name: gpu_h100, max_hours: 120, gpus: [h100:1]}

# Walltime model (PROVISIONAL -- recalibrate from real cluster timings).
# Estimated seconds = designs * (length/100) * (timesteps/100) * sec_per_unit,
# then scaled by `safety_factor` and rounded up.
timing:
  sec_per_unit: 25
  safety_factor: 2.0
  min_minutes: 20
"""


class ConfigError(RuntimeError):
    pass


@dataclass
class Cluster:
    """One cluster's settings, with {user} already substituted."""

    name: str
    ssh_target: str = ""
    python_module: str = ""
    venv_path: str = ""
    checkpoint_dir: str = ""
    jobs_dir: str = ""
    epilog_marker: str = ""
    gpu_tiers: List[Dict[str, Any]] = field(default_factory=list)
    partitions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Timing:
    sec_per_unit: float = 25.0
    safety_factor: float = 2.0
    min_minutes: int = 20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            "PyYAML is required. Install it with: python3 -m pip install pyyaml"
        ) from exc
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def ensure_config(path: Path = CONFIG_PATH) -> Path:
    """Create the config file with annotated defaults if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG)
    return path


def resolve_cluster_user(ssh_target: str) -> str:
    """Cluster username for {user} substitution.

    `ssh -G <alias>` reads ~/.ssh/config and prints the effective config
    without connecting, so this is fast and offline.  Falls back to the local
    username.
    """
    if "@" in ssh_target:
        return ssh_target.split("@", 1)[0]
    if ssh_target and shutil.which("ssh"):
        try:
            out = subprocess.run(
                ["ssh", "-G", ssh_target],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in out.splitlines():
                if line.lower().startswith("user "):
                    return line.split(None, 1)[1].strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return os.environ.get("LOGNAME") or os.environ.get("USER") or ""


def load(cluster_name: Optional[str] = None, path: Path = CONFIG_PATH):
    """Return (Cluster, Timing) for `cluster_name` (or the default cluster)."""
    ensure_config(path)
    data = _load_yaml(path)

    clusters = data.get("clusters")
    if not isinstance(clusters, dict) or not clusters:
        # Tolerate a legacy flat config by wrapping it as a single cluster.
        clusters = {"default": data}
        data = {"default_cluster": "default", "clusters": clusters}

    name = cluster_name or data.get("default_cluster")
    if not name:
        raise ConfigError(f"{path}: no `default_cluster` set and no --cluster given")
    if name not in clusters:
        raise ConfigError(
            f"Unknown cluster {name!r}. Configured: {', '.join(sorted(clusters))}"
        )

    raw = clusters[name] or {}
    ssh_target = str(raw.get("ssh_target", "") or "")

    venv = raw.get("venv") or {}
    venv_path = str(venv.get("path", "") if isinstance(venv, dict) else venv or "")

    paths = {
        "venv_path": venv_path,
        "checkpoint_dir": str(raw.get("checkpoint_dir", "") or ""),
        "jobs_dir": str(raw.get("jobs_dir", "") or ""),
    }
    # Only pay for username resolution if something actually needs it.
    if any("{user}" in v for v in paths.values()):
        user = resolve_cluster_user(ssh_target)
        if not user:
            raise ConfigError(
                f"Cluster {name!r} uses {{user}} in its paths but the username "
                f"could not be resolved from ssh_target={ssh_target!r}."
            )
        paths = {k: v.replace("{user}", user) for k, v in paths.items()}

    cluster = Cluster(
        name=name,
        ssh_target=ssh_target,
        python_module=str(raw.get("python_module", "") or ""),
        epilog_marker=str(raw.get("epilog_marker", "") or ""),
        gpu_tiers=list(raw.get("gpu_tiers") or []),
        partitions=list(raw.get("partitions") or []),
        **paths,
    )

    t = data.get("timing") or {}
    timing = Timing(
        sec_per_unit=float(t.get("sec_per_unit", 25)),
        safety_factor=float(t.get("safety_factor", 2.0)),
        min_minutes=int(t.get("min_minutes", 20)),
    )
    return cluster, timing


def list_clusters(path: Path = CONFIG_PATH) -> List[str]:
    ensure_config(path)
    data = _load_yaml(path)
    clusters = data.get("clusters")
    if isinstance(clusters, dict):
        return sorted(clusters)
    return ["default"]


def default_cluster_name(path: Path = CONFIG_PATH) -> Optional[str]:
    ensure_config(path)
    return _load_yaml(path).get("default_cluster")
