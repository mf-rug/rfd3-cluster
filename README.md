# rfd3-setup-job

Generate RFdiffusion3 (+ ProteinMPNN / LigandMPNN) Slurm jobs for an HPC
cluster, from a single local command. The RFD3/MPNN counterpart to
[`boltz-setup`](https://github.com/mf-rug/boltz-setup), and it uses the same
transport tool, [`hpcjob`](https://github.com/mf-rug/hpcjob), for
submit/status/pull.

| Step | Tool | What it does |
|------|------|--------------|
| 1. Generate | `rfd3-setup-job` | Writes the RFD3 design JSON + a **cluster-correct** `job.sh` locally |
| 2. Submit / fetch | `hpcjob` | `hpcjob submit` uploads + `sbatch`; `hpcjob pull` rsyncs results back |

Everything cluster-specific (modules, venv, checkpoint dir, partitions, GPU
tiers) lives in `~/.config/rfd3-setup/config.yaml`, so the same tool targets
any SLURM cluster: add a block, select it with `--cluster NAME`.

---

## Install

```bash
git clone <this repo> ~/Documents/code/rfd3_cluster
echo 'alias rfd3-setup-job="$HOME/Documents/code/rfd3_cluster/rfd3-setup-job"' >> ~/.bashrc
```

Requires Python 3.9+ with PyYAML locally. Nothing needs installing on the
cluster beyond the Foundry venv itself (see `HANDOFF.md`) — `rfd3_tools/` is
copied into every job directory.

First run writes an annotated `~/.config/rfd3-setup/config.yaml`. Check that
its paths match what was actually installed on the cluster.

## Usage

```bash
# 20 unconditional backbones, 100 residues
rfd3-setup-job --length 100 --num-designs 20 --name denovo --out-dir ./denovo/

# length screen — three independent specs, one job
rfd3-setup-job --length "80|120|160" --num-designs 8 --name lenscan --out-dir ./lenscan/

# binder design with hotspots, then sequences for every backbone
rfd3-setup-job --input target.pdb --contig "A1-100,50-150" \
    --hotspots "A45-50,A75" --num-designs 32 --mpnn \
    --name binder --out-dir ./binder/

# motif scaffolding
rfd3-setup-job --input site.pdb --unindex "A431,A572-573" --length "120-180" \
    --num-designs 16 --name motif --out-dir ./motif/

# partial diffusion around an existing structure
rfd3-setup-job --input start.cif --partial-t 10 --set 'unindex=A20-40' \
    --num-designs 8 --name partial --out-dir ./partial/

# MPNN on its own, over backbones already on the cluster
rfd3-setup-job --mpnn-only \
    --structures remote:/scratch/$USER/rfd3_jobs/denovo/output/rfd3 \
    --mpnn-seqs 8 --name reseq --out-dir ./reseq/
```

Then:

```bash
hpcjob submit ./denovo/job.sh --cluster habrok --jobs-dir /scratch/$USER/rfd3_jobs
hpcjob status <job_id> --cluster habrok
hpcjob pull denovo --cluster habrok --yes
```

`rfd3-setup-job` prints the exact `hpcjob submit` line, including `--jobs-dir`,
after generating a job.

### Anything the flags don't cover

RFD3's spec supports symmetry, per-atom h-bond conditioning, RASA burial,
sequence unfixing and more. Rather than growing a flag per field, hand-write
the inputs JSON and pass it through — it is validated exactly the same way:

```bash
rfd3-setup-job --spec-json my_specs.json --num-designs 8 --name custom --out-dir ./custom/
```

Or set any single field inline (values are parsed as JSON where possible):

```bash
rfd3-setup-job --length 100 --set plddt_enhanced=false \
    --set 'select_fixed_atoms={"A431":"TIP"}' --name x --out-dir ./x/
```

`rfd3-setup-job --print-fields` lists every accepted spec field.

## What gets written

```
<out-dir>/
  input/designs.json     RFD3 inputs (one entry per design spec)
  input/structures/      any local PDB/CIF a spec references
  job.sh                 cluster-correct Slurm script
  rfd3_tools/            shipped so the job needs no cluster-side install
```

After the run, `hpcjob pull` brings back:

```
<out-dir>/
  output/rfd3/           designed backbones (.cif.gz) + per-design .json
  output/staged_cif/     decompressed copies fed to MPNN
  output/mpnn/           sequences (.fa) + designed structures (.cif)
  <name>_COMPLETED_<id>.log   clean summary written by the job's exit trap
```

RFD3 writes **gzipped mmCIF**, never PDB. The sibling `.json` is worth keeping:
it holds free per-design QC (`n_chainbreaks`, interresidue clash counts,
`helix_fraction`, `radius_of_gyration`, `sampled_contig`) — a pre-filter before
spending MPNN or folding time on a bad backbone. The summary log reports how
many designs came out clean.

## How the job is structured

1. `module load` + venv activate + `FOUNDRY_CHECKPOINT_DIRS` export (weights
   live on scratch/project, not `$HOME`).
2. **RFD3**: `rfd3 design` over `input/designs.json`.
   `--num-designs N` is split into `diffusion_batch_size × n_batches`.
3. **MPNN** (optional): the backbone list doesn't exist until RFD3 finishes, so
   the job builds MPNN's batch config *on the cluster*
   (`python -m rfd3_tools mpnn-config`) and runs **one** `mpnn` process over
   every backbone via `--config_json`. The model loads once, not per structure.
4. Exit trap parses `slurm-*.out` into a readable summary.

## Resource recommendation

RFD3's VRAM grows roughly with the **square of design length** and is almost
independent of batch size — measured on Snellius at 200 timesteps:

| length | peak VRAM | batch 1 → 8 |
|---|---|---|
| 100 | 3.4 GB | +1 % VRAM, 8× the designs |
| 200 | 8.0 GB | 36 s → 55 s on H100 |
| 400 | 26 GB | |
| 600 | 57 GB | OOMs on a 40 GB A100 |

So **length picks the GPU tier and batching is nearly free**. `max_length` is a
hard cliff, not a soft limit: RFD3 dies on a single large allocation in the
pairwise embedding, so a job slightly over the limit fails in seconds rather
than running slowly. The generator warns when a design exceeds every tier.

```yaml
gpu_tiers:
  - {max_length: 200, gpu_sbatch: a100_3g.20gb:1, mem: 48GB}   # MIG, cheapest
  - {max_length: 400, gpu_sbatch: a100:1,         mem: 64GB}
timing:
  sec_per_batch_per_100aa: 12     # dominates...
  sec_per_design_per_100aa: 2.2   # ...this one — hence batching is cheap
  safety_factor: 2.0
```

Cores default to the partition's full QOS per-GPU allowance (`cpus_per_gpu`),
since GPU nodes are charged per GPU — asking for fewer saves nothing, and
asking for more makes Slurm reject the job.

Override per job with `--gpu`, `--mem`, `--time`, `--partition`, `--cpus`.

> Snellius values are measured; Hábrók's are extrapolated placeholders until
> that install happens. Recalibrating is a config edit, not a code change.

## See also

- `HANDOFF.md` — cluster-side Foundry install (venv, torch pinning, weights).
- RFD3 input spec: https://rosettacommons.github.io/foundry/models/rfd3/input.html
- Foundry: https://github.com/RosettaCommons/foundry
