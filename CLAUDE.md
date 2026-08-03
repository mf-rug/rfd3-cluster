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
- **MPNN CLI** takes either one `--structure_path` or `--config_json`, the
  latter being `{globals…, "inputs": [ {structure_path, name, …}, … ]}`. The
  batch form is what the job uses: one process, model loaded once.
  `is_legacy_weights: true` is required for every checkpoint in the default
  registry.
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
- Transport is **not** this repo's job: `hpcjob` handles submit/pull. The only
  change made there was a `--jobs-dir` flag on `submit`, so RFD3 jobs land in
  `rfd3_jobs/` without needing duplicate cluster entries.

## Known provisional bits

- `gpu_tiers` (`max_length` thresholds) and `timing.sec_per_unit` are guesses
  until the cluster install reports measured timings (HANDOFF.md §4.3/§4.4).
  Recalibrate in config, not code.
- `spec.max_design_length()` assumes 300 residues when length is implied by a
  contig rather than stated, since resolving a contig needs the structure.
- Untested end-to-end: no job has run on a cluster yet (Foundry install pending).
  The generator, validator, MPNN-config builder and log parser have local unit
  coverage only.
