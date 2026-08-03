"""GPU/walltime recommendation and Slurm job-script templating.

Nothing here knows a partition or GPU name: every choice comes from the
selected cluster's config block, which is what makes the same generator emit a
correct script for any Slurm cluster.

Calibrated against measurements from the Snellius install (2026-08-03):
RFD3 VRAM grows ~L^2 with design length and is almost independent of
diffusion batch size, so length picks the GPU tier and batching is cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    # Design scope. Without one MPNN redesigns every residue in the backbone.
    # `designed_map` is the usual route: a JSON of spec name -> residue IDs,
    # derived from each spec's select_unfixed_sequence so the two cannot drift.
    designed_residues: Optional[str] = None
    designed_map: Optional[str] = None
    # Per-position amino-acid constraints (forcing/forbidding residue types).
    omit_map: Optional[str] = None
    extra_args: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend_gpu(cluster, max_length: int) -> Tuple[str, str, bool]:
    """(gpu_sbatch, mem, warn) for the largest design in the job.

    `max_length` is the selector because VRAM is set by design length (~L^2),
    not by how many designs are sampled at once. Tiers are a hard cliff: RFD3
    dies with a single large allocation in the pairwise embedding rather than
    degrading gracefully, so a job one residue over the limit OOMs in seconds.
    """
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
        bool(last.get("warn", True)),
    )


def exceeds_largest_tier(cluster, max_length: int) -> bool:
    """True when no configured tier covers this design length."""
    if not cluster.gpu_tiers:
        return False
    return max_length > max(int(t.get("max_length", 0)) for t in cluster.gpu_tiers)


def partial_step_fraction(partial_t: Optional[float], timing) -> float:
    """Fraction of the denoising schedule a partial-diffusion run executes.

    With `partial_t` set, sampling starts from a low noise level instead of
    pure noise, so only the tail of the schedule runs -- far cheaper than the
    timestep count implies (measured: 2 specs x 8 designs x 442 tokens at
    timesteps=200, partial_t=10, finished in 6.5 min including MPNN).

    Memory is *not* affected: peak VRAM is set by the pairwise activation in a
    single forward pass, which is identical at every timestep. This scales time
    only -- never feed it into tier selection.
    """
    if not partial_t or partial_t <= 0:
        return 1.0
    reference = max(getattr(timing, "partial_t_reference", 20.0), 1e-6)
    return min(1.0, float(partial_t) / reference)


def estimate_seconds(
    *,
    length: int,
    batch_size: int,
    n_batches: int,
    n_specs: int,
    timesteps: int,
    timing,
    gpu_speedup: float = 1.0,
    partial_t: Optional[float] = None,
    n_sequences: int = 0,
) -> int:
    """Wall-time estimate from the measured cost structure.

    Three components, all in config:

    - a fixed model-load cost (~20 s for the 2.5 GB checkpoint) paid once;
    - a per-batch cost that scales with length;
    - a per-design cost that scales with length.

    The per-design term is much smaller than the per-batch term, which is the
    measured result that batching is nearly free: at L=200 one design costs
    ~28 s of compute and eight cost ~58 s, not 8x.  Length scaling is taken as
    linear in wall time (memory is what grows quadratically).
    """
    length_factor = max(length, 1) / 100.0
    step_factor = (max(timesteps, 1) / max(timing.ref_timesteps, 1)) * partial_step_fraction(
        partial_t, timing
    )

    per_batch = length_factor * (
        timing.sec_per_batch_per_100aa + timing.sec_per_design_per_100aa * batch_size
    )
    compute = per_batch * n_batches * max(n_specs, 1) * step_factor
    total = timing.load_seconds + compute / max(gpu_speedup, 0.1)

    # MPNN runs in the same job and is not free: in the reference run it was
    # roughly half the wall time. One model load plus a per-sequence cost that
    # scales with chain length.
    if n_sequences:
        total += getattr(timing, "mpnn_load_seconds", 20.0)
        total += (
            getattr(timing, "sec_per_sequence_per_100aa", 0.3)
            * n_sequences
            * length_factor
        )

    seconds = total * timing.safety_factor
    return max(int(math.ceil(seconds)), timing.min_minutes * 60)


def format_walltime(seconds: int) -> str:
    seconds = int(math.ceil(seconds / 60.0) * 60)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_partition(cluster, gpu_sbatch: str, seconds: int) -> Dict[str, Any]:
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
    return min(fitting, key=lambda p: float(p.get("max_hours", 0)))


def partition_by_name(cluster, name: str) -> Optional[Dict[str, Any]]:
    for p in cluster.partitions:
        if str(p.get("name")) == name:
            return p
    return None


def cap_cpus(partition: Optional[Dict[str, Any]], requested: Optional[int]) -> int:
    """Respect the QOS core-per-GPU ceiling.

    Slurm rejects the job outright (`QOSMaxCpuPerJobLimit`) when a partition's
    per-GPU core share is exceeded -- e.g. 9 cores alongside one MIG slice.
    GPU nodes are charged per GPU, so when no explicit request is made we take
    the full allowance rather than leaving free cores on the table.
    """
    ceiling = None
    if partition and partition.get("cpus_per_gpu"):
        ceiling = int(partition["cpus_per_gpu"])
    if requested is None:
        return ceiling if ceiling else 8
    return min(requested, ceiling) if ceiling else requested


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------


def _hydra_override(item: str) -> str:
    """Prefix `+` where Hydra needs it to *add* a key rather than modify one.

    `specification` is declared as an empty mapping in
    `inference_engine/rfdiffusion3.yaml`, and OmegaConf runs in struct mode, so
    `specification.length=60` fails with `Key 'length' is not in struct`; it
    must be `+specification.length=60`. Every other key already exists in the
    config, where a `+` is equally an error. Two classes, not one.
    """
    if item.startswith(("+", "~")):
        return item
    if item.startswith("specification."):
        return "+" + item
    return item


def _rfd3_command(p: Rfd3Params, venv_bin: str) -> str:
    parts = [
        f"{venv_bin}/rfd3 design",
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
        parts.append(f"    {_hydra_override(override)}")
    return " \\\n".join(parts)


def _mpnn_stage(m: MpnnParams, checkpoint_dir: str, venv_bin: str) -> str:
    """Build the MPNN stage.

    The config JSON is built *on the cluster*, after RFD3 has run, because the
    backbone filenames don't exist at generation time.  One `mpnn` process then
    handles every backbone via its native batch interface (`--config_json` with
    an `inputs: [...]` list), so the model loads once instead of per structure.

    RFD3 writes gzipped mmCIF (`.cif.gz`).  Whether MPNN reads that directly is
    unverified, so the helper stages decompressed copies and points MPNN at
    those -- cheap, and it removes the unknown.
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
    if m.designed_residues:
        extra += f"    --designed-residues '{m.designed_residues}' \\\n"
    if m.designed_map:
        extra += f"    --designed-residues-map '{m.designed_map}' \\\n"
    if m.omit_map:
        extra += f"    --omit-per-residue-map '{m.omit_map}' \\\n"
    for arg in m.extra_args or []:
        extra += f"    {arg} \\\n"

    return f"""
# --- stage 2: MPNN sequence design over every backbone (single process) ---
PYTHONPATH="$_job_dir" {venv_bin}/python -m rfd3_tools mpnn-config \\
    --structures '{m.structures}' \\
    --out ./input/mpnn.json \\
    --model-type {m.model_type} \\
    --out-directory '{m.out_dir}' \\
    --gunzip-to ./output/staged_cif \\
    --batch-size {m.batch_size} \\
    --number-of-batches {m.number_of_batches} \\
{ckpt_arg}{extra}    --quiet

{venv_bin}/mpnn --config_json ./input/mpnn.json
"""


def build_job_script(
    *,
    cluster,
    slurm: SlurmParams,
    rfd3: Optional[Rfd3Params],
    mpnn: MpnnParams,
) -> str:
    """Emit a portable Slurm script for the selected cluster.

    Portability rules:
    - use `$SLURM_SUBMIT_DIR`, never a path baked in at generation time;
    - call binaries by **absolute venv path**, which is immune to `PATH`
      ordering and to a missing `module load`, and avoids `rfd3/cli.py`
      stripping literal `design`/`fold` tokens from the argument list as the
      `python -c` form does;
    - never let the exit trap fail the job.
    """
    if not cluster.venv_path:
        raise RuntimeError(
            f"Cluster {cluster.name!r} has no venv.path configured. The Foundry "
            f"install must be completed first (see HANDOFF.md)."
        )
    venv_bin = f"{cluster.venv_path.rstrip('/')}/bin"

    stages = []
    if rfd3 is not None:
        stages.append(
            f"""
# --- stage 1: RFD3 backbone generation ---
{_rfd3_command(rfd3, venv_bin)}
"""
        )
    if mpnn.enabled:
        stages.append(_mpnn_stage(mpnn, cluster.checkpoint_dir, venv_bin))

    # What counts as "the job produced something".  RFD3 emits gzipped mmCIF
    # (`*.cif.gz`), never PDB -- checking for `*.cif` alone would report every
    # successful run as a failure.
    if rfd3 is not None:
        success_dir = rfd3.out_dir
        success_pat = r"-name '*.cif.gz' -o -name '*.cif' -o -name '*.pdb'"
    else:
        success_dir = mpnn.out_dir
        success_pat = r"-name '*.fa' -o -name '*.cif'"

    module_line = f"module purge\nmodule load {cluster.python_module}\n" if cluster.python_module else ""
    checkpoint_export = (
        f'# No fallback: foundry writes nothing to $HOME, so a missing export\n'
        f'# means no checkpoint is discoverable at all.\n'
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
    PYTHONPATH="$_job_dir" {venv_bin}/python -m rfd3_tools log "$_job_dir" || true
}}
trap cleanup EXIT

echo "start $(date)"

{module_line}
{checkpoint_export}
nvidia-smi || true
# `PYTORCH_ALLOC_CONF` is the current spelling in torch 2.10; the legacy name
# is set too so the script behaves the same on older stacks.
export PYTORCH_ALLOC_CONF=expandable_segments:True
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
