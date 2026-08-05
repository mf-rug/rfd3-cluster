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

# Written on first run.  The Snellius block holds MEASURED values (install
# report, 2026-08-03); the Habrok block is still provisional -- its tiers are
# extrapolated from the same L^2 memory curve and must be re-measured.
DEFAULT_CONFIG = """\
# rfd3-setup cluster configuration (multi-cluster).
# Select a cluster with `rfd3-setup-job --cluster NAME`; without it,
# `default_cluster` is used. Machine-local: holds your paths/accounts.
#
# {user} is replaced with the cluster username (resolved via `ssh -G <ssh_target>`).

default_cluster: habrok

clusters:

  # ---- RUG / Habrok ------------------------------------------------------
  # No Foundry install here yet (venv absent as of 2026-08-05); Python
  # 3.12.3-GCCcore-13.3.0 exists, so the install is unblocked.
  habrok:
    ssh_target: hpc
    python_module: Python/3.12.3-GCCcore-13.3.0
    venv:
      path: /scratch/{user}/venvs/rfd3
    checkpoint_dir: /scratch/{user}/foundry/checkpoints
    jobs_dir: /scratch/{user}/rfd3_jobs
    epilog_marker: "Habrok Cluster"
    # Tier caps derived from the MEASURED A100 curve on Snellius
    # (mem ~= 4.04 + 1.372e-4 * L^2 GiB; that fit reproduces the measured
    # L=460/480 points to within 2%) solved against each card's VRAM at the
    # same ~82% occupancy the measured a100 cap sits at. VRAM per card from
    # the Habrok cluster description: v100 32GB, a100 40GB, l40s 48GB,
    # rtx_pro_6000 96GB. EXTRAPOLATED -- run verify/bisect.sh here to confirm.
    gpu_tiers:
      - {max_length: 390, gpu_sbatch: v100:1,         mem: 32GB}   # 32GB card
      - {max_length: 460, gpu_sbatch: a100:1,         mem: 64GB}   # 40GB, measured
      - {max_length: 500, gpu_sbatch: l40s:1,         mem: 48GB}   # 48GB card
      - {max_length: 730, gpu_sbatch: rtx_pro_6000:1, mem: 96GB, warn: true}
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
    # MEASURED peak VRAM: L100=3.4  L200=8.0  L400=26  L460=32.7  L480=35.5
    # L500=38.4 (97% of a 40GB A100)  L520=OOM  L600=57 (on H100).
    # max_length is a hard OOM cliff, not a soft limit, and is set BELOW the
    # measured cliff: memory follows token count, and a ligand or extra chain
    # makes a design wider than its stated length. `mem` is HOST RAM.
    gpu_tiers:
      - {max_length: 200, gpu_sbatch: a100_3g.20gb:1, mem: 48GB}   # MIG, cheapest
      - {max_length: 460, gpu_sbatch: a100:1,         mem: 64GB}   # 500 fits, 520 OOMs
      - {max_length: 700, gpu_sbatch: h100:1,         mem: 96GB}   # 600 fits at 57GB
    # cpus_per_gpu is a QOS ceiling: exceed it and sbatch rejects the job.
    # Nodes are charged per GPU, so the full allowance is taken by default.
    partitions:
      - {name: gpu_mig,  max_hours: 120, gpus: [a100_3g.20gb:1], cpus_per_gpu: 9}
      - {name: gpu_a100, max_hours: 120, gpus: [a100:1],         cpus_per_gpu: 18}
      - {name: gpu_h100, max_hours: 120, gpus: [h100:1],         cpus_per_gpu: 16}

# Walltime model, fitted to measured A100 runs at 200 timesteps:
#   seconds = (load + n_batches * n_specs * (L/100) *
#              (sec_per_batch_per_100aa + sec_per_design_per_100aa * batch))
#             * (timesteps/ref_timesteps) * safety_factor
# The per-batch term dominates the per-design term -- that is the measured
# result that raising diffusion_batch_size is nearly free.
#
# A partial-diffusion run (partial_t set) executes only the tail of the
# schedule, so the estimate is scaled by partial_t / partial_t_reference.
# This affects TIME only -- peak VRAM is unchanged, so tier selection must
# never use it.
timing:
  load_seconds: 30
  sec_per_batch_per_100aa: 12
  sec_per_design_per_100aa: 2.2
  ref_timesteps: 200
  partial_t_reference: 20
  # MPNN shares the job's walltime: one model load + per-sequence cost.
  mpnn_load_seconds: 20
  sec_per_sequence_per_100aa: 0.3
  safety_factor: 2.0
  min_minutes: 10
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
    """Wall-time cost model (see generate.estimate_seconds).

    Defaults are fitted to measured A100 runs at 200 timesteps: the per-batch
    term dominates the per-design term, which is why batching is nearly free.
    """

    load_seconds: float = 30.0
    sec_per_batch_per_100aa: float = 12.0
    sec_per_design_per_100aa: float = 2.2
    ref_timesteps: int = 200
    # Noise level at which a partial-diffusion run costs the full schedule.
    # `partial_t` below this scales the estimate down proportionally.
    partial_t_reference: float = 20.0
    # MPNN runs in the same job: one model load plus a per-sequence cost.
    mpnn_load_seconds: float = 20.0
    sec_per_sequence_per_100aa: float = 0.3
    safety_factor: float = 2.0
    # Over-requesting walltime worsens scheduling priority, so the floor is
    # kept modest; safety_factor already absorbs variance.
    min_minutes: int = 10


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
        load_seconds=float(t.get("load_seconds", 30)),
        sec_per_batch_per_100aa=float(t.get("sec_per_batch_per_100aa", 12)),
        sec_per_design_per_100aa=float(t.get("sec_per_design_per_100aa", 2.2)),
        ref_timesteps=int(t.get("ref_timesteps", 200)),
        partial_t_reference=float(t.get("partial_t_reference", 20.0)),
        mpnn_load_seconds=float(t.get("mpnn_load_seconds", 20.0)),
        sec_per_sequence_per_100aa=float(t.get("sec_per_sequence_per_100aa", 0.3)),
        safety_factor=float(t.get("safety_factor", 2.0)),
        min_minutes=int(t.get("min_minutes", 10)),
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
