"""GPU/walltime recommendation and Slurm job-script templating.

Nothing here knows a partition or GPU name: every choice comes from the
selected cluster's config block, which is what makes the same generator emit a
correct script for any Slurm cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .cluster import Cluster, Timing

# Default MPNN checkpoint *patterns*, resolved by glob on the cluster at job
# time rather than hardcoded here -- filenames carry version numbers
# (proteinmpnn_v_48_020.pt) that change between weight releases.
MPNN_CKPT_GLOB = {
    "protein_mpnn": "proteinmpnn*.pt",
    "ligand_mpnn": "ligandmpnn*.pt",
}


@dataclass
class SlurmParams:
    job_name: str
    time: str
    partition: str
    gpus_per_node: str
    mem: str
    cpus_per_task: int = 8
    nodes: int = 1


@dataclass
class Rfd3Params:
    inputs: str = "./input/designs.json"
    out_dir: str = "./output/rfd3"
    diffusion_batch_size: int = 8
    n_batches: int = 1
    num_timesteps: int = 200
    seed: Optional[int] = None
    ckpt_path: Optional[str] = None
    dump_trajectories: bool = False
    skip_existing: bool = False
    prevalidate_inputs: bool = True
    low_memory_mode: bool = False
    extra_overrides: Optional[List[str]] = None


@dataclass
class MpnnParams:
    enabled: bool = False
    model_type: str = "protein_mpnn"
    structures: str = "./output/rfd3"  # dir the job reads backbones from
    out_dir: str = "./output/mpnn"
    batch_size: int = 1
    number_of_batches: int = 8
    checkpoint: Optional[str] = None  # explicit path; else glob-resolved
    temperature: Optional[float] = None
    write_structures: bool = True
    extra_args: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend_gpu(cluster: Cluster, max_length: int) -> Tuple[str, str, bool]:
    """(gpu_sbatch, mem, warn) for the largest design in the job."""
    if not cluster.gpu_tiers:
        raise RuntimeError(
            f"Cluster {cluster.name!r} has no gpu_tiers configured -- add them "
            f"to ~/.config/rfd3-setup/config.yaml."
        )
    for tier in cluster.gpu_tiers:
        if max_length <= int(tier.get("max_length", 0)):
            return (
                str(tier.get("gpu_sbatch", "")),
                str(tier.get("mem", "32GB")),
                bool(tier.get("warn", False)),
            )
    last = cluster.gpu_tiers[-1]
    return (
        str(last.get("gpu_sbatch", "")),
        str(last.get("mem", "32GB")),
        bool(last.get("warn", False)),
    )


def estimate_seconds(
    *, designs: int, length: int, timesteps: int, timing: Timing
) -> int:
    """Provisional cost model -- linear in designs x length x timesteps.

    The constant lives in config (`timing.sec_per_unit`) precisely because it
    is a guess until the cluster reports measured timings; recalibrating is a
    one-line config edit, not a code change.
    """
    units = designs * (length / 100.0) * (timesteps / 100.0)
    seconds = units * timing.sec_per_unit * timing.safety_factor
    return max(int(math.ceil(seconds)), timing.min_minutes * 60)


def format_walltime(seconds: int) -> str:
    seconds = int(math.ceil(seconds / 60.0) * 60)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def recommend_partition(cluster: Cluster, gpu_sbatch: str, seconds: int) -> str:
    """Smallest partition that offers this GPU and is long enough."""
    hours = seconds / 3600.0
    candidates = [
        p
        for p in cluster.partitions
        if gpu_sbatch in [str(g) for g in (p.get("gpus") or [])]
    ]
    if not candidates:
        raise RuntimeError(
            f"No partition on {cluster.name!r} offers GPU {gpu_sbatch!r}. "
            f"Check `partitions` in ~/.config/rfd3-setup/config.yaml."
        )
    fitting = [p for p in candidates if float(p.get("max_hours", 0)) >= hours]
    if not fitting:
        longest = max(candidates, key=lambda p: float(p.get("max_hours", 0)))
        raise RuntimeError(
            f"Estimated walltime {hours:.1f} h exceeds the longest partition "
            f"offering {gpu_sbatch} on {cluster.name!r} "
            f"({longest.get('name')}: {longest.get('max_hours')} h). "
            f"Split the job (fewer designs per job) or pass --time/--partition."
        )
    return str(min(fitting, key=lambda p: float(p.get("max_hours", 0)))["name"])


def cap_walltime(cluster: Cluster, partition: str, seconds: int) -> int:
    for p in cluster.partitions:
        if str(p.get("name")) == partition:
            return min(seconds, int(float(p.get("max_hours", 0)) * 3600))
    return seconds


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------


def _rfd3_command(p: Rfd3Params) -> str:
    parts = [
        'python -c "from rfd3.cli import app; app()" design',
        f"    inputs={p.inputs}",
        f"    out_dir={p.out_dir}",
        f"    diffusion_batch_size={p.diffusion_batch_size}",
        f"    n_batches={p.n_batches}",
        f"    inference_sampler.num_timesteps={p.num_timesteps}",
        f"    skip_existing={p.skip_existing}",
        f"    prevalidate_inputs={p.prevalidate_inputs}",
    ]
    if p.seed is not None:
        parts.append(f"    seed={p.seed}")
    if p.ckpt_path:
        parts.append(f"    ckpt_path={p.ckpt_path}")
    if p.dump_trajectories:
        parts.append("    dump_trajectories=True")
    if p.low_memory_mode:
        parts.append("    low_memory_mode=True")
    for override in p.extra_overrides or []:
        parts.append(f"    {override}")
    return " \\\n".join(parts)


def _mpnn_stage(m: MpnnParams, checkpoint_dir: str) -> str:
    """Build the MPNN stage.

    The config JSON is built *on the cluster*, after RFD3 has run, because the
    backbone filenames don't exist at generation time.  One `mpnn` process then
    handles every backbone via its native batch interface (`--config_json` with
    an `inputs: [...]` list), so the model loads once instead of per structure.
    """
    ckpt_arg = (
        f"    --checkpoint {m.checkpoint} \\\n"
        if m.checkpoint
        else f"    --checkpoint-glob '{MPNN_CKPT_GLOB.get(m.model_type, '*.pt')}' \\\n"
        f"    --checkpoint-dir '{checkpoint_dir}' \\\n"
    )
    extra = ""
    if m.temperature is not None:
        extra += f"    --temperature {m.temperature} \\\n"
    if not m.write_structures:
        extra += "    --no-write-structures \\\n"
    for arg in m.extra_args or []:
        extra += f"    {arg} \\\n"

    return f"""
# --- stage 2: MPNN sequence design over every backbone (single process) ---
PYTHONPATH="$_job_dir" python -m rfd3_tools mpnn-config \\
    --structures '{m.structures}' \\
    --out ./input/mpnn.json \\
    --model-type {m.model_type} \\
    --out-directory '{m.out_dir}' \\
    --batch-size {m.batch_size} \\
    --number-of-batches {m.number_of_batches} \\
{ckpt_arg}{extra}    --quiet

python -c "from mpnn.inference import main; main()" --config_json ./input/mpnn.json
"""


def build_job_script(
    *,
    cluster: Cluster,
    slurm: SlurmParams,
    rfd3: Optional[Rfd3Params],
    mpnn: MpnnParams,
) -> str:
    """Emit a portable Slurm script for the selected cluster.

    Portability rules (learned from the boltz job scripts):
    - use `$SLURM_SUBMIT_DIR`, never a path baked in at generation time;
    - invoke via the venv's `python -c`, so a stray `rfd3`/`mpnn` binary
      earlier on `PATH` can't shadow the venv's interpreter;
    - never let the exit trap fail the job.
    """
    if not cluster.venv_path:
        raise RuntimeError(
            f"Cluster {cluster.name!r} has no venv.path configured. The Foundry "
            f"install must be completed first (see HANDOFF.md)."
        )

    stages = []
    if rfd3 is not None:
        stages.append(
            f"""
# --- stage 1: RFD3 backbone generation ---
{_rfd3_command(rfd3)}
"""
        )
    if mpnn.enabled:
        stages.append(_mpnn_stage(mpnn, cluster.checkpoint_dir))

    # What counts as "the job produced something".
    if rfd3 is not None:
        success_dir = rfd3.out_dir
        success_pat = r"-name '*.pdb' -o -name '*.cif'"
    else:
        success_dir = mpnn.out_dir
        success_pat = r"-name '*.fa' -o -name '*.cif'"

    checkpoint_export = (
        f'export FOUNDRY_CHECKPOINT_DIRS="{cluster.checkpoint_dir}"\n'
        if cluster.checkpoint_dir
        else ""
    )

    return f"""#!/bin/bash
#SBATCH --job-name={slurm.job_name}
#SBATCH --time={slurm.time}
#SBATCH --gpus-per-node={slurm.gpus_per_node}
#SBATCH --partition={slurm.partition}
#SBATCH --nodes={slurm.nodes}
#SBATCH --cpus-per-task={slurm.cpus_per_task}
#SBATCH --mem={slurm.mem}
#SBATCH --output=slurm-%j.out

# Resolve job directory: Slurm's submit dir, falling back to the script's own
# location so the script also runs outside Slurm.
_job_dir="${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}}"

scontrol show job $SLURM_JOB_ID 2>/dev/null || true

# Rename script to include the job ID (job.sh -> job_12345.sh)
_self="$(readlink -f "$0")"
_stem="${{_self%.sh}}"
mv "$_self" "${{_stem}}_${{SLURM_JOB_ID}}.sh" 2>/dev/null || true

set -x
set -e
PS4='$(date +%H:%M:%S) '

cleanup() {{
    local rc=$?
    set +x
    echo "========================================="
    if [ $rc -eq 0 ]; then
        echo "Job completed successfully in ${{SECONDS}}s"
    else
        echo "Job FAILED (exit code $rc) after ${{SECONDS}}s"
    fi
    echo "========================================="
    # Clean summary log; best-effort, never fails the job.
    # rfd3_tools/ ships alongside job.sh, so no cluster-side install is needed.
    PYTHONPATH="$_job_dir" python -m rfd3_tools log "$_job_dir" || true
}}
trap cleanup EXIT

echo "start $(date)"

module purge
module load {cluster.python_module}
source {cluster.venv_path}/bin/activate

{checkpoint_export}
nvidia-smi || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$_job_dir"
{"".join(stages)}
# Fail loudly if the job produced no output despite exiting 0.
find {success_dir} \\( {success_pat} \\) -print -quit 2>/dev/null | grep -q . || {{
    echo "no output produced in {success_dir}" >&2
    exit 1
}}

echo "done $(date)"
"""
