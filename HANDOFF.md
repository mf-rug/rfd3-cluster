# HANDOFF — install the Foundry suite (RFD3 / MPNN / RF3) on the HPC clusters

**Audience:** a Claude Code session running *on the cluster* (Hábrók and/or Snellius).
**Author:** a Claude Code session on the local Mac, building the `rfd3-setup-job`
toolbox that will generate Slurm jobs against this install.

Read this file top to bottom before running anything. The final section
("What to report back") is the deliverable — the local toolbox is blocked on it.

---

## 1. Goal

Provide a working, per-cluster installation of the RosettaCommons **Foundry**
suite so that Slurm job scripts can run:

- **RFD3** (`rfd3 design`) — de novo backbone generation. Primary use.
- **MPNN** (`mpnn`) — ProteinMPNN / LigandMPNN inverse folding on those backbones.
- **RF3** (`rf3 fold`) — structure prediction, for self-consistency validation.
  Nice to have; install it (it comes with `[all]`) but RFD3 + MPNN are what
  must work.

This mirrors an existing, working Boltz-2 setup on the same clusters. **Follow
the same conventions** — same venv-on-scratch pattern, same "heavy pip installs
run inside a Slurm job" rule. Do **not** modify, reuse, or upgrade anything
under the existing Boltz venvs or caches.

## 2. Target layout

Use these paths unless something forces a change (if you change one, say so in
the report — the local config must match exactly).

### Hábrók (RUG)

| Thing | Path |
|---|---|
| venv | `/scratch/$USER/venvs/rfd3` |
| checkpoints | `/scratch/$USER/foundry/checkpoints` |
| jobs dir | `/scratch/$USER/rfd3_jobs` (create it; jobs land here later) |
| python module | `Python/3.11.5-GCCcore-13.2.0` (as used by Boltz) — **but see §3.1** |

### Snellius (SURF, project `<PROJECT>`)

| Thing | Path |
|---|---|
| venv | `/projects/<PROJECT>/rfd3_venv` |
| checkpoints | `/projects/<PROJECT>/foundry/checkpoints` |
| jobs dir | `/projects/<PROJECT>/rfd3_jobs` |
| python module | `2024 Python/3.12.3-GCCcore-13.3.0` (Lmod year-stack first, one `module load` line) |

**Snellius filesystem etiquette:** `$HOME` is small (200 GiB) and backed up —
nothing goes there. `/projects/<PROJECT>/` is persistent but **group-shared**, so
hitting quota hurts everyone; the ~5.5 GB of weights living there is deliberate
(shared benefit), but do not stage scratch data there. `/gpfs/scratch1/` is
8 TiB, not backed up, auto-purged.

## 3. Install procedure

### 3.1 Python version — check before you commit to a module

Foundry requires a reasonably modern Python. The Mac install runs **3.12**.
Hábrók's Boltz venv uses **3.11.5**, which *should* be fine, but verify that
`pip install "rc-foundry[all]"` resolves under whichever module you pick before
building the whole thing. If 3.11 fails to resolve, use the newest Python
module available (`module spider Python`) and report which one you used.

### 3.2 Create the venv (login node)

```bash
module purge
module load <python_module>            # see table above
python3 -m venv <venv_path>
source <venv_path>/bin/activate
pip install --upgrade pip
```

### 3.3 Install torch + foundry — **inside a Slurm job, not on the login node**

The login node kills large pip installs (this is exactly how the Boltz install
had to be done). Write a small job script, `sbatch` it, and check the log.

**Torch pinning — the important part.** Hábrók's driver supports up to CUDA
12.9; the current default PyPI torch is built against CUDA 13.x and fails on
every Hábrók node with *"The NVIDIA driver on your system is too old"*. The
Boltz install solved this with a pinned cu128 wheel, and `torch==2.10.0` was
verified across V100 / A100 / L40s / RTX Pro 6000 (newer wheels drop `sm_70`
and break V100).

So, install in this order:

```bash
# 1. pinned CUDA 12.8 torch FIRST
pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128

# 2. foundry (plain PyPI; the Linux-only rf3/cuEquivariance extra installs here,
#    unlike on macOS where it is skipped)
pip install "rc-foundry[all]"

# 3. defensive: re-pin torch in case foundry pulled a different build
pip install --force-reinstall 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
```

**Check after step 2 whether foundry actually replaced torch** — report what
`pip list | grep -iE '^(torch|rc-foundry|atomworks|nvidia-cu)'` shows before and
after step 3. If foundry hard-requires a torch newer than 2.10, do **not** fight
it silently: report the conflict, install what foundry needs, and note that V100
support is then likely lost.

On **Snellius** the GPUs are A100/H100 only (no V100), so the `sm_70`
constraint doesn't apply and a newer torch is acceptable *if* it's needed —
but still prefer a CUDA build the driver supports (`nvidia-smi` shows the max
CUDA version).

### 3.4 Download model weights

```bash
foundry install base-models --checkpoint-dir <checkpoint_dir>
foundry list-installed
```

`base-models` is ~5.5 GB: `rfd3_latest.ckpt` (2.5 GB),
`rf3_foundry_01_24_latest_remapped.ckpt` (2.8 GB), `proteinmpnn_v_48_020.pt`,
`ligandmpnn_v_32_010_25.pt`.

Foundry searches `~/.foundry/checkpoints` **plus** the colon-separated entries in
`$FOUNDRY_CHECKPOINT_DIRS`. Since we deliberately keep weights off `$HOME`,
every job script will export:

```bash
export FOUNDRY_CHECKPOINT_DIRS=<checkpoint_dir>
```

Confirm that `foundry list-installed` finds the checkpoints **with `$HOME/.foundry`
absent or empty** and only `FOUNDRY_CHECKPOINT_DIRS` set — that is the condition
the job scripts will run under. This is a real failure mode worth testing
explicitly.

The download is network-bound, not CPU-bound, so the login node is fine for it
if the compute nodes lack outbound network. Report which way worked.

## 4. Verification

Run all of these and paste the actual output into the report.

### 4.1 Imports

```bash
python -c "import torch, rfd3, rf3, mpnn; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch; print(torch._C._cuda_getArchFlags())"
```

The arch flags line tells us which GPU generations the wheel supports — we need
it to pick GPU tiers in the local config. Paste it verbatim.

### 4.2 MPNN smoke test (seconds, on GPU)

```bash
curl -sL https://files.rcsb.org/download/1L2Y.pdb -o /tmp/1L2Y.pdb
mpnn --model_type protein_mpnn \
     --checkpoint_path <checkpoint_dir>/proteinmpnn_v_48_020.pt \
     --is_legacy_weights True \
     --structure_path /tmp/1L2Y.pdb \
     --out_directory /tmp/mpnn_smoke \
     --number_of_batches 1 --batch_size 1
ls /tmp/mpnn_smoke
```

Note: `--is_legacy_weights True` is **required** for every checkpoint in the
default registry. Outputs are `<name>_b<batch>_d<design>.cif` plus a pooled
`<name>.fa`. The run is nearly silent on success — verify by `ls`.

Two `CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` "not set" warnings always print and
are harmless. **However** — if a CCD/PDB mirror already exists on the cluster,
note its path; it speeds up ligand work later.

### 4.3 RFD3 smoke test (the one that matters)

Small unconditional monomer, few timesteps:

```bash
mkdir -p /tmp/rfd3_smoke
rfd3 design \
    inputs=null \
    out_dir=/tmp/rfd3_smoke \
    specification.length=60 \
    diffusion_batch_size=1 \
    n_batches=1 \
    inference_sampler.num_timesteps=20 \
    skip_existing=False
ls /tmp/rfd3_smoke
```

Expect a `.pdb`/`.cif` plus a `.json` metadata file. **Time it** (`time ...`).

Then a realistic-size timing run — this calibrates the walltime/GPU
recommendation table in the local tool, so it is genuinely needed, not busywork:

```bash
time rfd3 design inputs=null out_dir=/tmp/rfd3_bench \
    specification.length=200 \
    diffusion_batch_size=8 n_batches=1 \
    inference_sampler.num_timesteps=200 \
    skip_existing=False
```

Report wall time, peak GPU memory (`nvidia-smi` during, or
`torch.cuda.max_memory_allocated()`), and which GPU type it ran on. If you can
cheaply repeat it on a second GPU type, do — a two-point curve is much more
useful than one.

### 4.4 GPU tier survey

For each GPU type available on the cluster (Hábrók: `v100`, `l40s`, `a100`,
`rtx_pro_6000`; Snellius: `a100`, `h100`), confirm RFD3 actually runs — a short
`num_timesteps=20`, `length=100` job is enough. We specifically need to know
whether **V100 (sm_70) works with RFD3**, since it's the cheapest queue.

Report a simple table: GPU → works / fails → error if it fails.

## 5. Constraints

- **Do not touch the Boltz venv, cache, or job directories.** They are in
  production use.
- Everything the local tooling needs must be reachable **without** interactive
  setup — no `conda activate`, no manual `module load` beyond what a job script
  can do itself.
- Keep the install reproducible: write the exact commands you ran into a file
  at `<venv_path>/../rfd3_install_log.md` (or the project equivalent) so it can
  be redone.
- If something in this handoff turns out to be wrong (path, module, torch pin),
  **do the sensible thing and say so** — do not silently follow an instruction
  that clearly won't work on the machine in front of you.

## 6. What to report back

The local toolbox is blocked on these. Return them as a single message.

1. **Filled config block, per cluster**, in exactly this shape:

```yaml
  <cluster_name>:
    ssh_target: <hpc|ss>
    python_module: "<module string used>"
    venv:
      path: <venv path>
    checkpoint_dir: <checkpoint dir>
    jobs_dir: <jobs dir>
    gpu_tiers:
      - {max_length: ???, gpu_sbatch: <type>:1, mem: ??GB}
    partitions:
      - {name: <partition>, max_hours: ??, gpus: [<type>:1]}
```

2. **Torch/foundry versions actually installed** (`pip list` excerpt) and the
   `torch._C._cuda_getArchFlags()` output.
3. **The GPU tier survey table** from §4.4.
4. **Timings** from §4.3 (length 60 / length 200), with GPU type and memory.
5. **Any deviation** from this document, and why.
6. **Exact invocation that works from a job script** — i.e. does `rfd3` resolve
   on `PATH` after `source <venv>/bin/activate`, or is
   `python -c "from rfd3.cli import app; app()" design ...` needed? (The Boltz
   job scripts use the `python -c` form to guarantee the venv's interpreter is
   used regardless of any binary earlier on `PATH`; we will likely do the same,
   but confirm both work.)

---

## Appendix — what the local side will generate

For context, so the install matches how it will be used. A generated job will
look approximately like this:

```bash
module purge && module load <python_module>
source <venv>/bin/activate
export FOUNDRY_CHECKPOINT_DIRS=<checkpoint_dir>

# stage 1: backbones
rfd3 design inputs=./input/designs.json out_dir=./output/rfd3 \
    diffusion_batch_size=8 n_batches=2 \
    inference_sampler.num_timesteps=200 skip_existing=False

# stage 2 (optional): sequences for every backbone, ONE process
mpnn --config_json ./input/mpnn.json
```

Stage 2 uses MPNN's native batch interface: a single JSON holding global model
settings plus an `"inputs": [{structure_path, name, batch_size, ...}, ...]`
list, so the model loads once for hundreds of backbones. If you discover that
this batch path behaves differently from the single-`--structure_path` CLI path
(e.g. different output naming), that is worth flagging.
