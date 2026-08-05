# HANDOFF — install the Foundry suite (RFD3 / MPNN / RF3) on a Slurm cluster

**Audience:** a Claude Code session running *on the cluster*.
**Author:** the Claude Code session on the local Mac that maintains
`rfd3-setup-job`, the generator that emits Slurm jobs against this install.

**Status of the fleet:**

| Cluster | State |
|---|---|
| **Snellius** (SURF) | ✅ installed and verified 2026-08-03 — used as the reference |
| **Hábrók** (RUG) | ⬜ not started — **this document is for you** |

Read this file top to bottom before running anything. §1 is the accumulated
list of things that have already gone wrong once; every item there cost the
Snellius session real time. The final section ("What to report back") is the
deliverable — the local generator's config is blocked on it.

---

## 0. Goal

A working per-cluster installation of the RosettaCommons **Foundry** suite so
that generated Slurm scripts can run:

- **RFD3** (`rfd3 design`) — de novo backbone generation. Primary use.
- **MPNN** (`mpnn`) — ProteinMPNN / LigandMPNN inverse folding on those backbones.
- **RF3** (`rf3 fold`) — structure prediction, for self-consistency validation.
  Comes with `[all]`; nice to have, but RFD3 + MPNN are what must work.

This mirrors an existing, working Boltz-2 setup on the same clusters. Follow
the same conventions — venv on scratch/project, heavy pip installs inside a
Slurm job. Do **not** modify, reuse, or upgrade anything under the existing
Boltz venvs or caches.

---

## 1. 🔴 Known traps — read before planning anything

Every item here is **measured on Snellius**, not guessed. Most are
cluster-independent and will bite on Hábrók too.

### 1.1 Python must be exactly 3.12.x — RESOLVED for Hábrók

`rc-foundry` 0.2.0 declares `Requires-Python: >=3.12,<3.13`. **3.11 cannot
resolve it at all, and 3.13 is equally excluded.** Hábrók's Boltz module is
Python 3.11.5 — **do not reuse it**.

**Checked 2026-08-05: Hábrók has `Python/3.12.3-GCCcore-13.3.0`** (the only
3.12 on offer; the others are 3.13.1 and 3.13.5, both excluded). No year-stack
prefix is needed, unlike Snellius. This was the biggest risk to the install and
it is cleared — use that module.

### 1.2 The RFD3 CLI spec-override syntax needs a `+`

```bash
rfd3 design inputs=null out_dir=... specification.length=60     # ✗ FAILS
rfd3 design inputs=null out_dir=... +specification.length=60    # ✓
```

`specification` is declared as an **empty mapping** in
`inference_engine/rfdiffusion3.yaml`, and OmegaConf runs in struct mode, so
without `+` Hydra reports `ConfigAttributeError: Key 'length' is not in struct`.
The rule:

> `specification.*` → **always** `+`.  Everything else → **never** `+`
> (`inputs`, `out_dir`, `diffusion_batch_size`, `n_batches`, `skip_existing`,
> `inference_sampler.num_timesteps`, `ckpt_path`, `global_prefix`, … all already
> exist; adding `+` to those is equally an error).

pydantic coerces a bare int, so `+specification.length=60` needs no quoting.
This affects **smoke tests and CLI-driven single designs**; the real pipeline
passes `inputs=<json>` where these fields come from the JSON instead.

### 1.3 RFD3 output is gzipped mmCIF, not PDB

Output for a design is `<prefix>_<batch>_model_<i>.cif.gz` plus a sibling
`.json` — **never `.pdb`, and never uncompressed**. Two consequences:

- Anything scanning for results must match `*.cif.gz`. (This silently broke the
  generator's success check until it was fixed.)
- With `inputs=null` the name starts with a **leading underscore**
  (`_0_model_0`), because the pattern is
  `{jsonfilebasename}_{jsonkey}_{batch}_{model}` and the first two parts are
  empty. Set `global_prefix` explicitly if you run that way.

**Keep the `.json`.** It carries free per-design QC — `n_chainbreaks`,
interresidue clash counts, `helix_fraction`, `radius_of_gyration`,
`num_residues`, and the `sampled_contig` — which is exactly what you filter on
before spending MPNN/RF3 time on a bad backbone.

### 1.4 Memory scales with **length**, not batch size

Measured peak VRAM at 200 timesteps: **L=100 → 3.4 GB, L=200 → 8.0 GB,
L=400 → 26 GB, L=600 → 57 GB** — roughly L². Meanwhile
`diffusion_batch_size` 1 → 8 costs **~1 % more VRAM** for 8× the designs, and
is far sublinear in wall time (at L=200 on H100: 36 s for 1 design, 55 s for 8).

Two consequences for the tier table you will report:
- `max_length` per GPU is a **hard OOM cliff**, not a soft limit. RFD3 dies on a
  single large allocation in the pairwise distance embedding (an L×L×D tensor),
  so a job one residue over the limit fails in ~45 s rather than running slowly.
  **Leave headroom.**
- Batch size should be maxed out; `n_batches` is for going beyond it.

### 1.5 The torch pin is safe — the brief used to overstate this risk

`rc-foundry` requires only `torch<3,>=2.2.0` and does **not** fight a pin.
Verified: torch identical before and after `pip install "rc-foundry[all]"`, and
`pip check` clean. The defensive re-pin (step 3 below) is a **no-op** and can be
dropped.

`torch==2.10.0+cu128` reports
`sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` — so **V100 (sm_70) is supported
by the wheel**, and Blackwell too. On Hábrók the real V100 question is
therefore **memory, not compute**. Hábrók's V100s are **32 GB** (per the
cluster description — not 16 GB, which an earlier draft of this file assumed),
so extrapolating the measured curve they cap out around **L≈390**. Expect V100
to be *usable but length-limited*, not broken.

### 1.6 `base-models` is 7.9 GB, not 5.5 GB

Five checkpoints, not four — there is an undocumented
`rfd3na_1190.ckpt` (2.51 GB), the nucleic-acid RFD3 variant:

| file | size |
|---|---|
| `rfd3_latest.ckpt` | 2.51 GB |
| `rfd3na_1190.ckpt` | 2.51 GB ← not in the original brief |
| `rf3_foundry_01_24_latest_remapped.ckpt` | 2.83 GB |
| `proteinmpnn_v_48_020.pt` | 0.01 GB |
| `ligandmpnn_v_32_010_25.pt` | 0.01 GB |

Check quota accordingly. There is no way to install a subset via
`base-models`; if space is tight, `rfd3na_1190.ckpt` is the one to delete
afterwards — nothing in the RFD3+MPNN pipeline uses it.

### 1.7 Assorted, each cheap to honour

- **`$FOUNDRY_CHECKPOINT_DIRS` has no fallback.** `foundry install
  --checkpoint-dir …` writes nothing to `$HOME` — no registry, nothing to go
  stale, but equally nothing discoverable if a job script forgets the export.
- **`PYTORCH_ALLOC_CONF` is the current spelling** in torch 2.10;
  `PYTORCH_CUDA_ALLOC_CONF` is legacy but still works. Set both.
- **Redirect `PIP_CACHE_DIR` to `$TMPDIR`** in the install job, or pip caches
  ~4 GB of CUDA wheels into `$HOME/.cache`.
- **`$TMPDIR` is deleted when the job ends.** Copy anything diagnostic back to
  shared storage *before* exiting, or you will re-run jobs just to read errors.
- **A `DeprecationWarning: torch.jit.script is deprecated` prints on every
  foundry entry point.** Cosmetic — don't let a log scraper call it a failure.
- **`wandb` never initialises during `rfd3 design`** despite being a hard
  dependency. `WANDB_MODE=offline` is optional, not required.
- **Don't assume one `$USER` across clusters** — the Hábrók account and the
  Snellius account differ.
- **Overlap the queue** by submitting verification jobs with
  `--dependency=afterok:<install_jobid>` before the install finishes. Costs
  nothing if the install fails; saves the whole install duration in queue time.

---

## 2. Target layout

Use these paths unless something forces a change (if you change one, say so in
the report — the generator's config must match exactly).

### Hábrók (RUG)

| Thing | Path |
|---|---|
| venv | `/scratch/$USER/venvs/rfd3` |
| checkpoints | `/scratch/$USER/foundry/checkpoints` |
| jobs dir | `/scratch/$USER/rfd3_jobs` (create it) |
| python module | **must be 3.12.x — see §1.1**, `module spider Python` |

### Snellius (SURF) — reference, already done

| Thing | Path |
|---|---|
| venv | `/projects/<PROJECT>/rfd3_venv` |
| checkpoints | `/projects/<PROJECT>/foundry/checkpoints` |
| jobs dir | `/projects/<PROJECT>/rfd3_jobs` |
| python module | `2024 Python/3.12.3-GCCcore-13.3.0` (Lmod year-stack first) |

Reusable scripts from that install, **cluster-agnostic apart from three path
variables at the top** — copy them rather than rewriting:
`/projects/<PROJECT>/rfd3_jobs/_install/install.sh`,
`_verify/verify.sh`, `_verify/scaling.sh`.

---

## 3. Install procedure

### 3.1 Create the venv (login node)

```bash
module purge
module load <python_3.12_module>       # §1.1 — verify the version first
python3 -m venv <venv_path>
source <venv_path>/bin/activate
pip install --upgrade pip
```

### 3.2 Install torch + foundry — inside a Slurm job

Use a **CPU** partition: the pip install needs no GPU, and holding a GPU
through a ~15 min download is waste. (On Snellius this ran on `rome` with 16
cores in 14:47.) The reason for using a job at all is the login node's
CPU/memory limits, not network.

```bash
export PIP_CACHE_DIR="$TMPDIR"        # §1.7

# 1. pinned CUDA 12.8 torch FIRST
pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128

# 2. foundry (plain PyPI; the Linux-only rf3/cuEquivariance extra installs here)
pip install "rc-foundry[all]"

# 3. OPTIONAL, and a confirmed no-op (§1.5) — drop it unless you want the belt
# pip install --force-reinstall 'torch==2.10.0' --index-url .../cu128
```

**Check Hábrók's driver first** (`nvidia-smi`) and confirm it supports CUDA
12.8. Report `pip check` output and the torch build actually installed.

### 3.3 Download model weights

```bash
foundry install base-models --checkpoint-dir <checkpoint_dir>   # 7.9 GB, §1.6
foundry list-installed
```

**Check whether Hábrók's compute nodes have outbound network** — Snellius's do
(40–500 MB/s), but do not assume it. If not, download from the login node; it
is network- not CPU-bound, so that is fine.

Every generated job exports:

```bash
export FOUNDRY_CHECKPOINT_DIRS=<checkpoint_dir>
```

Confirm `foundry list-installed` finds all five checkpoints with
`$HOME/.foundry` absent and `FOUNDRY_CHECKPOINT_DIRS` as the only signal —
that is exactly the condition job scripts run under.

---

## 4. Verification

Run all of these and paste the actual output into the report.

### 4.1 Imports

```bash
python -c "import torch, rfd3, rf3, mpnn; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch; print(torch._C._cuda_getArchFlags())"
```

### 4.2 MPNN smoke test (~15 s, model load dominates)

```bash
curl -sL https://files.rcsb.org/download/1L2Y.pdb -o "$TMPDIR/1L2Y.pdb"
mpnn --model_type protein_mpnn \
     --checkpoint_path <checkpoint_dir>/proteinmpnn_v_48_020.pt \
     --is_legacy_weights True \
     --structure_path "$TMPDIR/1L2Y.pdb" \
     --out_directory "$TMPDIR/mpnn_smoke" \
     --number_of_batches 1 --batch_size 1
```

`--is_legacy_weights True` is **required** for every checkpoint in the default
registry. Expect `1L2Y_b0_d0.cif` + pooled `1L2Y.fa`. Nearly silent on success.

**Also note whether a CCD/PDB mirror exists on Hábrók** (none on Snellius) —
it matters for ligand work later.

### 4.3 RFD3 smoke test

```bash
rfd3 design inputs=null out_dir="$TMPDIR/rfd3_smoke" \
    +specification.length=60 \
    diffusion_batch_size=1 n_batches=1 \
    inference_sampler.num_timesteps=20 \
    skip_existing=False
```

Note the `+` (§1.2). Expect `.cif.gz` + `.json` (§1.3). Time it.

### 4.4 The length × batch grid — the measurement that matters

Copy `_verify/scaling.sh` from Snellius. Run **L ∈ {100, 200, 400, 600} ×
B ∈ {1, 8}** at 200 timesteps on **each** GPU type Hábrók offers
(`v100`, `l40s`, `a100`, `rtx_pro_6000`), recording wall time and peak VRAM,
and letting the OOMs happen — the cliff location is the point.

Use `torch.cuda.max_memory_allocated()` in-process for the memory number if
`nvidia-smi --query-gpu` is unreliable (it is, on MIG-sliced cards).

This is what fills the `gpu_tiers` table; a single timing point is not enough.

### 4.5 Both invocation forms

Confirm `<venv>/bin/rfd3` works **without** `activate`, and that
`python -c "from rfd3.cli import app; app()" design …` also works. The
generator uses the **absolute venv-bin path** because it is immune to `PATH`
ordering and to a missing `module load`.

⚠️ If you use the `python -c` form, note that `rfd3/cli.py` filters the literal
tokens `design` and `fold` out of the argument list, so an override whose
*value* is exactly `design` or `fold` is silently dropped.

---

## 5. Constraints

- **Do not touch the Boltz venv, cache, or job directories.** Production.
- Everything must work **non-interactively** from a job script — no
  `conda activate`, no manual module juggling.
- Write the exact commands you ran to `<venv_path>/../rfd3_install_log.md`.
- If something here is wrong for the machine in front of you, **do the sensible
  thing and say so** — don't follow an instruction that clearly won't work.

---

## 6. What to report back

1. **Filled config block**, exactly this shape (note `cpus_per_gpu` — a QOS
   ceiling; exceeding it makes sbatch reject the job outright):

```yaml
  habrok:
    ssh_target: hpc
    python_module: "<module string actually used>"
    venv:
      path: <venv path>
    checkpoint_dir: <checkpoint dir>
    jobs_dir: <jobs dir>
    gpu_tiers:
      - {max_length: ???, gpu_sbatch: <type>:1, mem: ??GB}   # max_length = measured OOM cliff, with headroom
    partitions:
      - {name: <partition>, max_hours: ??, gpus: [<type>:1], cpus_per_gpu: ??}
```

2. **The length × batch grid** (§4.4) — wall time and peak VRAM per GPU type,
   including where each card OOMs. Say explicitly which `max_length` values are
   *measured at the boundary* and which are extrapolated.
3. **Versions**: `pip list` excerpt (`torch`, `rc-foundry`, `atomworks`,
   `cuequivariance*`, `nvidia-*`), plus `torch._C._cuda_getArchFlags()`,
   driver/CUDA version, and `pip check`.
4. **The Python 3.12 answer** (§1.1) — which module, or that none exists.
5. **Compute-node network** — yes/no, and where you downloaded weights.
6. **Any deviation** from this document, and why.
7. **Anything this document got wrong.** Snellius's report did exactly that and
   it was the most valuable part — three of the corrections in §1 came from it,
   and one of them was a live bug in the generator.

---

## Appendix — what the generator emits

For context, so the install matches its use. A generated job looks like this
(paths and module from the cluster's config block):

```bash
module purge && module load <python_module>
export FOUNDRY_CHECKPOINT_DIRS=<checkpoint_dir>
export PYTORCH_ALLOC_CONF=expandable_segments:True

# stage 1: backbones — absolute venv path, no activate needed
<venv>/bin/rfd3 design inputs=./input/designs.json out_dir=./output/rfd3 \
    diffusion_batch_size=8 n_batches=2 \
    inference_sampler.num_timesteps=200 skip_existing=False

# stage 2 (optional): sequences for every backbone, ONE process
<venv>/bin/python -m rfd3_tools mpnn-config --structures ./output/rfd3 \
    --out ./input/mpnn.json --gunzip-to ./output/staged_cif ...
<venv>/bin/mpnn --config_json ./input/mpnn.json
```

Stage 2 uses MPNN's batch interface: one JSON with global model settings plus
an `"inputs": [{structure_path, name, …}, …]` list, so the model loads once for
hundreds of backbones.

**Two things still unverified — please check if you get the chance:**
1. Does MPNN read `.cif.gz` directly? The generator currently stages
   decompressed copies (`--gunzip-to`) to avoid the question. If MPNN handles
   gzip natively, that step can go.
2. Does MPNN's batch-JSON path name outputs the same way as the
   single-`--structure_path` path? Only the single-path form has been exercised.
