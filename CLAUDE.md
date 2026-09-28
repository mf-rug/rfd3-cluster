# rfd3_cluster — developer notes

Local-side generator for RFD3 / MPNN Slurm jobs. Deliberately mirrors
`~/Documents/code/boltz-setup` so the two toolboxes behave the same way; read
that repo's `CLAUDE.md` when in doubt about a convention.

## Layout

```
rfd3-setup-job          bash entrypoint (PYTHONPATH shim, like boltz-setup-yaml)
rfd3_tools/
  cli.py                argparse CLI -> job directory
  spec.py               RFD3 design-JSON building + validation
  generate.py           GPU/walltime recommendation + job.sh template
  cluster.py            ~/.config/rfd3-setup/config.yaml loader (multi-cluster)
  mpnn_config.py        cluster-side: build MPNN --config_json from a dir of backbones
  logparse.py           cluster-side: slurm-*.out -> clean summary
  __main__.py           cluster-side dispatch: `python -m rfd3_tools {log,mpnn-config}`
HANDOFF.md              instructions for the Claude session doing the cluster install
```

`rfd3_tools/` is **copied into every generated job directory**, so the two
cluster-side entry points work with zero installation on the cluster. Keep them
**stdlib-only** — they run under the Foundry venv's Python but must not import
anything from it, and must never raise into the job (the log parser is called
from an exit trap).

## Key upstream facts (verified against the installed package, not docs)

- **RFD3 spec schema** is `rfd3.inference.input_parsing.DesignInputSpecification`,
  a pydantic model with `extra="forbid"` — an unknown key is a hard failure at
  runtime. `spec.SPEC_FIELDS` mirrors its fields and
  `spec.validate_spec()` re-implements its `validate_input_schema` rules, so
  mistakes surface locally instead of after a queue wait. **If the cluster's
  rfd3 version changes, re-check both against the installed package.**
- **RFD3 CLI** is Hydra-based: `rfd3 design inputs=… out_dir=… key=value`.
  Config keys live in `rfd3/configs/inference_engine/rfdiffusion3.yaml`
  (`@package _global_`, so `num_timesteps` is at `inference_sampler.num_timesteps`,
  not top level). Designs per spec = `diffusion_batch_size × n_batches`.
- **`specification.*` overrides need a leading `+`; everything else must not
  have one.** `specification` is an empty mapping in the config and OmegaConf
  runs in struct mode, so `specification.length=60` fails with
  `Key 'length' is not in struct`. `generate._hydra_override()` applies this
  rule to `--rfd3-set` passthroughs.
- **RFD3 writes gzipped mmCIF (`*.cif.gz`), never PDB**, plus a sibling `.json`
  holding per-design QC metrics (`n_chainbreaks`, interresidue clash counts,
  `helix_fraction`, `radius_of_gyration`, `sampled_contig`). Any code that
  globs for results must match `.cif.gz` — this was a live bug caught by the
  Snellius install report. The log parser surfaces the QC metrics as a
  clean/flagged count.
- **VRAM scales ~L² with design length and is ~independent of batch size**
  (measured: L100 3.4 GB, L200 8.0 GB, L400 26 GB, L600 57 GB; batch 1→8 costs
  ~1% more VRAM for 8× the designs). So length picks the GPU tier, batching is
  nearly free, and `max_length` is a hard OOM cliff — RFD3 dies on one big
  allocation in the pairwise embedding rather than degrading.
- **Invoke by absolute `<venv>/bin/…` path.** Immune to `PATH` ordering and to a
  missing `module load`, and it avoids `rfd3/cli.py` stripping literal
  `design`/`fold` tokens from the argument list (the `python -c` form's trap).
- **MPNN CLI** takes either one `--structure_path` or `--config_json`, the
  latter being `{globals…, "inputs": [ {structure_path, name, …}, … ]}`. The
  batch form is what the job uses: one process, model loaded once.
  `is_legacy_weights: true` is required for every checkpoint in the default
  registry. **MPNN has no design scope by default and redesigns every residue**
  ("Design scope - if all None, design all residues"); per-input
  `designed_residues` / `fixed_residues` take `["A801", …]` and are mutually
  exclusive. The generator derives the scope from `select_unfixed_sequence`.
- **Checkpoints** are found via `~/.foundry/checkpoints` **plus**
  `$FOUNDRY_CHECKPOINT_DIRS` (colon-separated). Since weights live on
  scratch/project, every job exports the latter.
- MPNN checkpoint filenames carry version numbers (`proteinmpnn_v_48_020.pt`),
  so the job resolves them by **glob at run time** rather than a hardcoded name.

## Conventions that matter

- **No cluster names, partitions, module names or paths in Python.** Everything
  comes from the selected cluster's config block. That is what makes
  `--cluster snellius` produce a correct script with no hand-editing.
- Job scripts use `$SLURM_SUBMIT_DIR`, never a path baked in at generation time.
- Models are invoked as `python -c "from rfd3.cli import app; app()"` (and the
  MPNN equivalent) so a stray binary earlier on `PATH` can't shadow the venv's
  interpreter — same reasoning as the boltz job scripts.
- Transport is **not** this repo's job: `hpcjob` handles submit/pull. Its
  `--jobs-dir` flag (pushed as `de49b1e`) lets RFD3 jobs land in `rfd3_jobs/`
  without duplicate cluster entries. An older installed `hpcjob` rejects it, so
  the printed submit line probes `hpcjob submit --help` and omits the flag with
  a warning when unsupported — it is not dropped silently, because the two
  `jobs_dir` values deliberately differ and dropping it would file RFD3 jobs
  under the boltz directory.

## State of calibration

- **Snellius: measured.** `gpu_tiers` and the `timing` constants come from the
  install report of 2026-08-03 (length × batch grid on A100 / H100 / A100-MIG),
  plus a boundary bisect on 2026-08-05: L=500 fits at 38.35 GiB (97% of a 40 GB
  A100), L=520 OOMs, so the a100 tier is **460** with headroom for token count
  exceeding stated length. Batch independence holds at the boundary (batch 1/4/8
  within 2.3 MB). The MIG `max_length: 200` and H100 `max_length: 700` are still
  *inferred* — `verify/bisect.sh` settles them.
- **Hábrók: no install yet, tiers extrapolated.** `python_module` is now the
  real `Python/3.12.3-GCCcore-13.3.0` (confirmed present 2026-08-05; it is the
  only 3.12 there, and `rc-foundry` needs `>=3.12,<3.13`, so the 3.11.5 Boltz
  module cannot be reused). Tier caps are solved from the measured Snellius
  curve against each card's real VRAM — note **V100 is 32 GB, not 16** — giving
  v100 390 / a100 460 / l40s 500 / rtx_pro_6000 730. No Foundry venv exists
  there yet; run `verify/bisect.sh` once it does.
- The walltime model is fitted to A100 at 200 timesteps and has a `safety_factor`
  of 2. H100 is ~1.4–1.6× faster; no per-tier speed factor is applied yet
  (`estimate_seconds` accepts `gpu_speedup` but the CLI never passes it).
- **`partial_t` scales time, never memory.** A partial-diffusion run executes
  only the tail of the schedule (measured: 2 specs × 8 designs × 442 tokens at
  `timesteps=200`, `partial_t=10` → 6.5 min including MPNN), so the estimate is
  multiplied by `partial_t / partial_t_reference`. Peak VRAM is set by the
  pairwise activation in a *single* forward pass and is identical at every
  timestep — tier selection must never use `partial_t`.
- **MPNN shares the job's walltime** and is modelled separately
  (`mpnn_load_seconds` + `sec_per_sequence_per_100aa` × sequences × length).
  In the reference run it was roughly half the total. For `--mpnn-only` the
  backbone count lives on the cluster, so the estimate under-counts — pass
  `--time` for large re-sequencing runs.
- **A100 evidence beyond the grid**: 442 tokens (440 res + NAD, 2 chains,
  3436 atoms) ran on an A100-**40 GB** without OOM. That is a pass/fail
  observation — no peak-VRAM number was captured — and the boundary has since
  been measured directly: the cliff sits between 500 and 520 (see above).

## Known provisional bits

- `spec.design_length()` resolves length in order of reliability: explicit
  `length` → contig arithmetic → CA count of the input structure → an assumed
  300. The assumption is now the last resort rather than the first, and the CLI
  prints which one applied (`length : 441 residues (counted from x.pdb)`), so a
  wrong tier is visible before submission rather than after an OOM. Remaining
  gap: a contig with a segment this parser can't read falls through to the
  structure count, which is the *input* length, not the designed length.
- **First end-to-end cluster run done 2026-08-03** (Snellius, jobs 25172015 /
  25172019: 2 specs x 8 designs, 440 res, partial diffusion + LigandMPNN, ~6.5
  min each). Three problems found; all fixed. Remaining calibration issues and
  the full report: **`HANDOFF_FIRST_RUN.md`** -- read it before changing tiers,
  the walltime model or the QC summary.
- Unverified upstream: whether MPNN reads `.cif.gz` directly (the job stages
  decompressed copies via `--gunzip-to` to sidestep it), and whether MPNN's
  batch-JSON path names outputs like the single-`--structure_path` path.

## Building another tool like this

The pattern this repo and `boltz-setup` share — what `hpcjob` owns, when a
dedicated generator is worth it, the config rule, and the trap catalogue — is
written up in `hpcjob/docs/NEW-CLUSTER-TOOL.md`. Read it before starting a
third toolbox, and add to it when something new bites.
