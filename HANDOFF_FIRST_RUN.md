# Handoff — first end-to-end use of rfd3-cluster on a real cluster

Date: **2026-08-03**. Cluster: **Snellius** (SURF), the only one with a working
Foundry install. Workload: partial-diffusion pocket redesign of a carboxylic acid
reductase reductase-domain, ~440 residues, ligand-conditioned (NAD), two
independent scaffolds, 8 designs each + LigandMPNN.

Before this, `CLAUDE.md` said: *"Untested end-to-end: no job has run on a cluster
yet."* That is no longer true. Six submissions, three distinct failure modes, two
of them fixed here. This is the report back.

---

## 1. Bottom line

The generator works. Two real bugs and one gap were found; all three are fixed
and pushed (`d0fa6d9`, `73fea67`). Everything below §3 is for you to decide on —
none of it is broken, but the calibration is provably off and a few defaults
cost queue time or mislead.

**Successful run for reference:** jobs 25172015 / 25172019, `gpu_a100`, 2 specs ×
8 designs at 200 timesteps + LigandMPNN 8 seq/backbone, **6:28** and **6:59**
wall — against a 1-hour request. 32 backbones, 256 sequences, no errors.

---

## 2. Fixed here (pushed — review, don't redo)

### 2.1 Input paths were resolved against the wrong directory — `73fea67`

`designs.json` is written to `<jobdir>/input/`, structures to
`<jobdir>/input/structures/`. The generator recorded `./input/structures/x.pdb`,
assuming resolution relative to the job dir (the job's cwd). **RFD3 resolves a
relative `input` against the directory of the inputs JSON**, so it looked for
`<jobdir>/input/input/structures/x.pdb`.

Every design died in the `load_input` validator at **49 s**, after the queue
wait, before any GPU work. Both write sites now share
`spec.STAGED_INPUT_PREFIX = "./structures/"`.

*Worth knowing:* `prevalidate_inputs=True` is what made this cheap — it failed
in under a minute instead of part-way through a batch. Keep that default.

### 2.2 MPNN redesigned the entire protein — `73fea67`

MPNN's own default is *"Design scope — if all None, design all residues"*. The
generator never set a scope, so on a scaffold redesign it returned sequences
**168/440 residues divergent** from the parent — **161 of those outside the
redesign windows**, including a conserved pocket aromatic mutated to Ile in
every one of 128 sequences. Sequence recovery ~0.59–0.61 reported as if healthy.

This is the dangerous class: the job succeeds, the log is clean, the QC summary
looks fine, and the output is unusable. Nothing surfaces it except reading the
sequences.

Fix: the scope is derived from each spec's `select_unfixed_sequence`, written to
`input/mpnn_designed.json`, and applied per backbone by matching the spec name
in its filename. Deriving rather than adding a parallel flag means the two
cannot drift. `--mpnn-designed` overrides (needed for `--mpnn-only`, which has
no specs); `--mpnn-design-all` restores the old behaviour for de-novo work,
where redesigning everything *is* correct.

Note for anyone extending this: MPNN's `fixed_residues` and `designed_residues`
are **mutually exclusive** — it raises on both. Format is `["A801","A802",…]`.

### 2.3 Range syntax was not expanded on the `--mpnn-designed` path

Introduced by 2.2 and caught by the first `--mpnn-only` run. The derived map
expanded `A799-804` into individual IDs; the explicit flag split on commas and
passed the range straight through. MPNN's `_parse_id` accepts only
`<chain>`, `<chain><num>` or `<chain><num><icode>` and raises:

    ValueError: ID 'A799-804' must look like ...

Two code paths to the same destination, one tested. Both now go through
`spec.expand_residue_selection`. If you add a third entry point for design
scope, route it through the same function.

### 2.4 Design length was guessed at 300 — `d0fa6d9`

`max_design_length()` only understood an explicit `length` field and assumed 300
otherwise. Every partial-diffusion spec falls in that gap, since length is
implicit in the input structure. A 441-residue design was tiered as 300 → A100
(tier cap 400). Replaced with `design_length()`: explicit `length` → contig
arithmetic → CA count of the input → 300, and the CLI now prints which route it
took.

---

## 3. Open — calibration and defaults

### 3.1 The A100 tier cap is too conservative (measured)

Config says `{max_length: 400, gpu_sbatch: a100:1}` and pushes anything longer
to H100. **Measured: 440 residues, `diffusion_batch_size=8`, ran fine on an
A100-40GB** — no OOM, 6.5 min. Predicted VRAM from the documented curve
(L400=26 GB, L600=57 GB, ~L²) is ~31–32 GB, ~20 % headroom.

This matters more than it sounds. On the day: `gpu_h100` had **0 idle nodes of
88** (80 allocated) while `gpu_a100` had **5 idle**. The two H100 jobs sat in
`PENDING Priority` indefinitely; resubmitted to A100 they started within a
minute. The cap was costing an unbounded queue wait for a job the smaller card
handles comfortably.

Suggest raising the A100 tier to ~500 and bisecting the real boundary (known:
400 fits, 600 OOMs). Worth measuring at batch 1 vs 8 too — the "batch is nearly
free" claim held here but was not measured at the boundary.

### 3.2 The walltime model badly over-predicts for partial diffusion

The model scales on `num_timesteps`. With `partial_t` set, the sampler starts
from a low noise level and the run costs a small fraction of what the timestep
count implies. Measured: 2 specs × 8 designs × 440 res at `timesteps=200`,
`partial_t=10` → **6.5 min including MPNN**, against a model estimate that
floored to `min_minutes: 20`.

Over-requesting walltime is not free — it worsens scheduling priority. Suggest
scaling the estimate by `partial_t / <reference noise>` when `partial_t` is
present, or at minimum documenting that partial-diffusion jobs need far less
than the model says.

### 3.3 QC "clean" counts inherited crystallographic gaps as chainbreaks

The summary reported `0/16 clean` for one job and `7/16` for the other. Every
flag was `n_chainbreaks=1, interresidue_clashes_w_backbone=0,
interresidue_clashes_w_sidechain=0` — i.e. **no clashes at all**. The chainbreak
was already in the input crystal (unmodelled loops at 741→744 and 962→968).
A template with no gaps scored 7/8 clean; the same designs from a gapped
template of the *same protein* scored 0/8.

So the headline QC number currently reports a property of the input, not of the
design. Suggest computing chainbreaks in the input structure once and reporting
`introduced = design − input`, or at least splitting the count. As it stands the
metric will make anyone working from a normal crystal think every design failed.

### 3.4 `--jobs-dir` does not exist in the published `hpcjob`

`README.md`, the `rfd3-design` skill, and the submit line printed by
`rfd3-setup-job` all instruct:

```
hpcjob submit <job>/job.sh --cluster <name> --jobs-dir <path>
```

Published `hpcjob` (0.1.0, GitHub HEAD `8b9f2ad`, the only commit) **rejects
`--jobs-dir`**: `error: unrecognized arguments`. `CLAUDE.md` says the flag was
added there, but that change was never pushed — it exists only on the machine it
was written on.

Either push the `hpcjob` change or drop the flag from the printed line and the
docs. It works without it: the registry entry's `jobs_dir` already routes RFD3
jobs correctly, which is how the successful runs were submitted.

### 3.5 MPNN bias / omit is not exposed, and scoping alone is not enough

With the scope fix in place, LigandMPNN was run over 32 backbones from three
independent scaffolds, restricted to the 16 pocket positions, with the NAD
ligand present (no 2'-phosphate). Across **256 sequences**:

- the 2'-phosphate-gripping arginine came back in **256/256** — 100 %;
- an acidic residue at the anchor position appeared **0 times**;
- the second basic position stayed basic (Arg or Lys) in ~97 %.

RFD3 had removed that arginine in most backbones (Leu/Ile modal). MPNN puts it
back. For a cofactor-specificity switch the sequence stage must be *constrained*,
not merely scoped — `bias_per_residue` / `omit_per_residue` exist in MPNN's
per-input config but are not exposed by this generator, so the constrain-and-
redesign step currently has to be run by hand or off-cluster.

This is the highest-value missing feature for redesign work, as opposed to the
de-novo case: without it the cluster pipeline cannot produce a switched
sequence at all.

### 3.6 Smaller things

- **`hpcjob pull` writes to `./<name>/`**, not into the job directory the results
  belong to. `--dest` exists. Since `rfd3-setup-job` already prints a submit
  line, consider printing the matching pull line with `--dest <out-dir>`.
- **Resubmitting to an existing remote dir needs `--overwrite`** or it prompts;
  fine, but the printed submit line doesn't mention it and reruns are common
  after a fast failure.
- **A successful-but-empty run would report `COMPLETED`.** The exit trap
  correctly wrote a `_FAILED_` log on non-zero exit, so this did not bite — but
  nothing asserts that any structures were produced. A count check in the trap
  would close the gap between "Slurm says fine" and "there is output".

---

## 4. Verified working — don't re-litigate these

Everything here was exercised on real jobs and behaved as documented:

- **`.cif.gz` handling**: RFD3 writes gzipped mmCIF; the gunzip staging into
  `output/staged_cif/` and the MPNN hand-off both work.
- **Checkpoint resolution by glob** against `$FOUNDRY_CHECKPOINT_DIRS` — found
  `ligandmpnn_v_32_010_25.pt` with no explicit path.
- **Batched MPNN** via `--config_json`: one process, 16 backbones, model loaded
  once.
- **Exit trap and summary log**: `_COMPLETED_<id>.log` and `_FAILED_<id>.log`
  both written correctly, with per-design QC metrics parsed out.
- **`rfd3_tools/` shipped into the job dir** — no cluster-side install needed,
  as designed.
- **Spec validation before submission** caught malformed specs locally.
- **Waters are dropped by RFD3 at parse time.** A template carrying 475 waters
  and 4 dangling CONECT records produced `num_atoms=3436` = protein + ligand
  only, `num_chains=2`. Inputs need not be stripped of solvent; the per-design
  JSON's `num_atoms` is the way to confirm what was actually ingested.
- **The `+specification` Hydra rule** and the absolute-venv-path invocation both
  behaved as `CLAUDE.md` describes.

---

## 5. Suggested order of work

1. §3.5 — MPNN bias/omit, without which the pipeline cannot do redesign work.
2. §3.4 — the `--jobs-dir` mismatch, since it makes the documented command fail
   for anyone starting from the README. Cheapest, most visible.
3. §3.3 — the QC chainbreak metric, since it actively misleads.
4. §3.1 — raise the A100 cap after a bisect; the queue cost is real.
5. §3.2 — walltime model for partial diffusion.
6. §3.6 — the ergonomics.
