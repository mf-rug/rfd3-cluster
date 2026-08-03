# verify/ — measure a cluster's real GPU ceiling

`gpu_tiers` in `~/.config/rfd3-setup/config.yaml` decides which GPU a design
gets, and `max_length` is a **hard OOM cliff**: RFD3 dies on a single large
allocation in the pairwise embedding, so a design slightly over the limit fails
in seconds rather than running slowly. Guessing that number wastes queued jobs
in one direction and burns bigger GPUs in the other.

`bisect.sh` measures it.

## Run

```bash
export RFD3_VENV=/path/to/rfd3_venv
export RFD3_CKPT_DIR=/path/to/foundry/checkpoints
export RFD3_MODULE="2024 Python/3.12.3-GCCcore-13.3.0"   # omit if not needed

# edit the two CHANGEME lines in the SBATCH header for your partition/GPU
sbatch bisect.sh
```

Results land in `bisect_results.tsv` next to the script.

## What it does

An ascending sweep with early stop, not a binary search: probes are seconds
each, an OOM aborts quickly, and the full curve lets the *other* tiers be set by
extrapolation instead of guessed.

Two things make it cheap and reliable:

- **`num_timesteps=2`.** Peak VRAM comes from the pairwise L×L×D activation
  inside a single forward pass and is identical at every timestep, so two steps
  measure the same peak as two hundred. Timesteps scale wall time, not memory.
- **One process per probe**, so an OOM cannot leave a poisoned CUDA context for
  the next length.

Memory is read with `torch.cuda.max_memory_allocated()` inside the run.
`nvidia-smi --query-gpu` is not usable here: on a MIG slice it reports the
parent device and includes other tenants' memory.

After the sweep it re-probes the last passing length at batch 1 and 4, because
batch independence had only ever been checked well below the limit.

## Reading the result

Set the tier **below** the measured cliff. Two reasons:

1. The tier keys on design *length*, but memory follows *token count* — a
   ligand or a second chain makes a design wider than its stated length.
2. The failure is abrupt and total, so the costs are asymmetric: too high loses
   a queued job, too low spends a larger GPU.

## Measured so far

Snellius, A100-40GB, batch 8, 200-timestep-equivalent peak:

| length | peak VRAM | verdict |
|---:|---:|---|
| 460 | 32.66 GiB | ok (82%) |
| 480 | 35.45 GiB | ok (89%) |
| 500 | 38.35 GiB | ok (97%) |
| 520 | — | **OOM** |

Batch dependence at L=500: batch 1 → 38.348 GiB, batch 4 → 38.349, batch 8 →
38.351. A 2.3 MB spread for 8× the designs, so the tier table stays
one-dimensional.

Tier set to **460**, not 500, per the headroom argument above.

Still unmeasured: the h100 cap (700, extrapolated from L=600 at 57 GB), the MIG
cap (200, inferred), and everything on Hábrók — whose whole table is
extrapolated from this curve against each card's VRAM.
