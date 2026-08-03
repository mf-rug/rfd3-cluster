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
- Transport is **not** this repo's job: `hpcjob` handles submit/pull. A
  `--jobs-dir` flag was added there so RFD3 jobs land in `rfd3_jobs/` without
  duplicate cluster entries — **but that change was never pushed**, and the
  published `hpcjob` (0.1.0) rejects the flag, so the submit line this tool
  prints fails for anyone starting from a clean install. Works without it: the
  registry entry's own `jobs_dir` routes correctly. See HANDOFF_FIRST_RUN.md 3.4.

## State of calibration

- **Snellius: measured.** `gpu_tiers` and the `timing` constants come from the
  install report of 2026-08-03 (length × batch grid on A100 / H100 / A100-MIG).
  Caveats stated there: the MIG `max_length: 200` and the H100 `max_length: 700`
  are *inferred*, not measured at the boundary; the A100 `400` is measured on
  both sides. Bisect before trusting the inferred ones.
- **Hábrók: provisional.** Nothing measured. Tiers are extrapolated from the
  Snellius VRAM curve against each card's VRAM, and `python_module` is a
  deliberate `PLACEHOLDER-Python-3.12` — `rc-foundry` needs `>=3.12,<3.13`, so
  the 3.11.5 Boltz module cannot be reused. Fix both after that install.
- The walltime model is fitted to A100 at 200 timesteps and has a `safety_factor`
  of 2. H100 is ~1.4–1.6× faster; no per-tier speed factor is applied yet
  (`estimate_seconds` accepts `gpu_speedup` but the CLI never passes it).

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
