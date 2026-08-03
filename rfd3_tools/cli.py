"""`rfd3-setup-job` — generate an RFD3 / MPNN job directory for a Slurm cluster.

Produces, under `--out-dir`:

    input/designs.json      the RFD3 inputs file (one entry per design spec)
    input/structures/       any local PDB/CIF a spec refers to
    job.sh                  cluster-correct Slurm script
    rfd3_tools/             this package, so the job's exit trap works with no
                            cluster-side install

Submit it with the existing transport tool:

    hpcjob submit <out-dir>/job.sh --cluster <same cluster> --jobs-dir <jobs_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__, cluster as cluster_mod, generate, spec as spec_mod


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rfd3-setup-job",
        description="Generate an RFD3 (+ MPNN) Slurm job directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # 20 unconditional backbones, 100 residues
  rfd3-setup-job --length 100 --num-designs 20 --name denovo --out-dir ./denovo/

  # length screen: three independent specs in one job
  rfd3-setup-job --length "80|120|160" --num-designs 8 --name lenscan --out-dir ./lenscan/

  # binder design against a target, with hotspots, then sequences
  rfd3-setup-job --input target.pdb --contig "A1-100,50-150" \\
      --hotspots "A45-50,A75" --num-designs 32 --mpnn \\
      --name binder --out-dir ./binder/

  # anything exotic (symmetry, h-bond conditioning): hand-write the JSON
  rfd3-setup-job --spec-json my_specs.json --num-designs 8 --name custom --out-dir ./custom/

  # MPNN on its own, over backbones already sitting on the cluster
  rfd3-setup-job --mpnn-only --structures remote:/scratch/me/rfd3_jobs/old/output/rfd3 \\
      --mpnn-seqs 8 --name reseq --out-dir ./reseq/
""",
    )

    ap.add_argument("--version", action="version", version=f"rfd3-setup-job {__version__}")

    g = ap.add_argument_group("design specification")
    g.add_argument("--length", help="Design length: '100', '80-120', or '80|120|160' "
                                    "for one spec per value.")
    g.add_argument("--input", dest="input_path", help="Input PDB/CIF (shipped with the job).")
    g.add_argument("--contig", help="Indexed motif contig, e.g. 'A1-100,50-150'.")
    g.add_argument("--unindex", help="Unindexed motif components, e.g. 'A431,A572-573'.")
    g.add_argument("--ligand", help="Ligand name(s) to include, e.g. 'HAX,OAA'.")
    g.add_argument("--hotspots", help="Hotspot selection for binder design, e.g. 'A45-50'.")
    g.add_argument("--partial-t", type=float, default=None,
                   help="Partial-diffusion noise in Angstrom (<=15 recommended).")
    g.add_argument("--set", dest="extra_sets", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Any other RFD3 spec field (repeatable). Values are parsed "
                        "as JSON when possible: --set plddt_enhanced=false")
    g.add_argument("--spec-json", type=Path, default=None,
                   help="Use a hand-written RFD3 inputs JSON instead of the flags above.")

    s = ap.add_argument_group("sampling")
    s.add_argument("--num-designs", type=int, default=8,
                   help="Designs per spec (default 8).")
    s.add_argument("--batch-size", type=int, default=8,
                   help="diffusion_batch_size (default 8); n_batches is derived.")
    s.add_argument("--timesteps", type=int, default=200,
                   help="inference_sampler.num_timesteps (default 200).")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--ckpt", default=None, help="ckpt_path override (default: registry 'rfd3').")
    s.add_argument("--dump-trajectories", action="store_true")
    s.add_argument("--low-memory", action="store_true",
                   help="low_memory_mode=True (memory-efficient tokenization).")
    s.add_argument("--rfd3-set", dest="rfd3_overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="Extra raw Hydra override (repeatable).")

    m = ap.add_argument_group("MPNN sequence design")
    m.add_argument("--mpnn", action="store_true", help="Run MPNN after RFD3, same job.")
    m.add_argument("--mpnn-only", action="store_true",
                   help="Skip RFD3; run MPNN over --structures.")
    m.add_argument("--structures", default=None,
                   help="For --mpnn-only: a local directory (shipped with the job) or "
                        "'remote:/abs/path' already on the cluster.")
    m.add_argument("--mpnn-model", default="protein_mpnn",
                   choices=["protein_mpnn", "ligand_mpnn"])
    m.add_argument("--mpnn-seqs", type=int, default=8,
                   help="Sequences per backbone (default 8).")
    m.add_argument("--mpnn-batch-size", type=int, default=1)
    m.add_argument("--mpnn-temperature", type=float, default=None)
    m.add_argument("--mpnn-checkpoint", default=None, help="Explicit checkpoint path.")
    m.add_argument("--mpnn-designed", default=None, metavar="IDS",
                   help="Positions MPNN may design, e.g. 'A799-804,A826-830'; "
                        "everything else is held fixed. Default: derived from each "
                        "spec's select_unfixed_sequence, so MPNN designs exactly "
                        "what RFD3 was allowed to change.")
    m.add_argument("--mpnn-force", default=None, metavar="SPEC",
                   help="Force amino acids at positions, 'A801:DE' = only Asp or "
                        "Glu at 801 (repeatable via commas). Implemented as "
                        "omit-everything-else, MPNN's only per-residue lever. "
                        "Needed for a specificity switch: unconstrained, MPNN "
                        "restores the wild-type pocket residue nearly every time.")
    m.add_argument("--mpnn-forbid", default=None, metavar="SPEC",
                   help="Forbid amino acids at positions, e.g. 'A828:RK' = no Arg "
                        "or Lys at 828.")
    m.add_argument("--mpnn-design-all", action="store_true",
                   help="Let MPNN redesign every residue (its own default). Only "
                        "sensible for de-novo backbones -- on a scaffold redesign "
                        "it returns sequences tens of percent divergent from the "
                        "parent, mutating conserved positions outside the pocket.")
    m.add_argument("--no-mpnn-structures", action="store_true",
                   help="Write sequences only, skip designed CIFs.")

    c = ap.add_argument_group("cluster / Slurm")
    c.add_argument("-c", "--cluster", default=None, help="Cluster from config (default: default_cluster).")
    c.add_argument("--partition", default=None, help="Override the recommended partition.")
    c.add_argument("--time", dest="walltime", default=None, help="Override walltime, HH:MM:SS.")
    c.add_argument("--gpu", default=None, help="Override GPU spec, e.g. a100:1.")
    c.add_argument("--mem", default=None, help="Override memory, e.g. 48GB.")
    c.add_argument("--auto-route", action="store_true",
                   help="Check live GPU availability (via `hpcjob preflight`) and "
                        "move UP a tier if the size-chosen GPU is fully busy. Never "
                        "moves down: the tier cap is a hard OOM cliff.")
    c.add_argument("--cpus", type=int, default=None,
                   help="Cores to request (default: the partition's full QOS "
                        "per-GPU allowance, since nodes are charged per GPU).")

    o = ap.add_argument_group("output")
    o.add_argument("--name", default=None, help="Job name (default: out-dir basename).")
    o.add_argument("--out-dir", type=Path, default=None, help="Job directory to create.")
    o.add_argument("--stdout", action="store_true",
                   help="Print the design JSON and exit; write nothing.")
    o.add_argument("--list-clusters", action="store_true")
    o.add_argument("--print-fields", action="store_true",
                   help="Print the accepted RFD3 spec field names and exit.")

    return ap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_package(dest: Path) -> None:
    """Ship rfd3_tools/ into the job dir (mirrors boltz_tools/)."""
    src = Path(__file__).resolve().parent
    target = dest / src.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _preflight(cluster_name: str) -> Dict[str, Any]:
    """Ask `hpcjob preflight` what is free right now.

    hpcjob owns the cluster registry and the ssh layer, so querying it is one
    subprocess rather than a second implementation here. Any failure returns {}
    and the caller falls back to the size-based choice.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["hpcjob", "preflight", "--cluster", cluster_name, "--json"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {}


def _split_batches(num_designs: int, batch_size: int) -> tuple[int, int]:
    """Map a design count onto (diffusion_batch_size, n_batches)."""
    batch_size = max(1, min(batch_size, num_designs))
    n_batches = max(1, math.ceil(num_designs / batch_size))
    return batch_size, n_batches


def _resolve_specs(args: argparse.Namespace, name: str) -> tuple[Dict[str, Any], List[str]]:
    """Return (specs, local structure files to ship)."""
    if args.spec_json:
        specs = spec_mod.load_spec_file(args.spec_json)
        specs, sources = spec_mod.rewrite_input_paths(specs)
        return specs, sources

    specs = spec_mod.build_specs(
        name=name,
        length=args.length,
        input_path=args.input_path,
        contig=args.contig,
        unindex=args.unindex,
        ligand=args.ligand,
        hotspots=args.hotspots,
        partial_t=args.partial_t,
        extra_sets=args.extra_sets,
    )
    sources = [args.input_path] if args.input_path else []
    return specs, sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_clusters:
        default = cluster_mod.default_cluster_name()
        for entry in cluster_mod.list_clusters():
            print(f"{'*' if entry == default else ' '} {entry}")
        return 0

    if args.print_fields:
        for field_name in sorted(spec_mod.SPEC_FIELDS):
            print(field_name)
        return 0

    run_rfd3 = not args.mpnn_only
    run_mpnn = args.mpnn or args.mpnn_only

    if args.mpnn_only and not args.structures:
        ap.error("--mpnn-only requires --structures (a local dir or 'remote:/abs/path')")

    name = args.name or (args.out_dir.name if args.out_dir else None)
    if not name:
        ap.error("--name or --out-dir is required")

    # ---- design specs -----------------------------------------------------
    specs: Dict[str, Any] = {}
    sources: List[str] = []
    if run_rfd3:
        if not (args.spec_json or args.length or args.input_path or args.contig):
            ap.error(
                "nothing to design: pass --length (and/or --input/--contig), "
                "or --spec-json, or use --mpnn-only"
            )
        try:
            specs, sources = _resolve_specs(args, name)
        except spec_mod.SpecError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.stdout:
        print(json.dumps(specs, indent=2))
        return 0

    if not args.out_dir:
        ap.error("--out-dir is required (or use --stdout)")

    # ---- cluster ----------------------------------------------------------
    try:
        clu, timing = cluster_mod.load(args.cluster)
    except cluster_mod.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # ---- resources --------------------------------------------------------
    batch_size, n_batches = _split_batches(args.num_designs, args.batch_size)
    total_designs = batch_size * n_batches * max(1, len(specs) or 1)
    if specs:
        max_length, length_origin = spec_mod.design_length(specs, sources)
    else:
        max_length, length_origin = 200, "default"

    if args.gpu and args.mem:
        gpu_sbatch, mem, warn = args.gpu, args.mem, False
    else:
        gpu_sbatch, mem, warn = generate.recommend_gpu(clu, max_length)
        gpu_sbatch = args.gpu or gpu_sbatch
        mem = args.mem or mem

    over_tier = run_rfd3 and generate.exceeds_largest_tier(clu, max_length)

    # Live queue state, if asked for. Never fatal: a job that cannot be routed
    # optimally is still a job worth generating.
    preflight_report: Dict[str, Any] = {}
    route_note: Optional[str] = None
    if args.auto_route:
        preflight_report = _preflight(clu.name)
        if preflight_report.get("reachable"):
            gpu_sbatch, new_mem, route_note = generate.reroute_for_availability(
                clu, gpu_sbatch, preflight_report)
            if new_mem and not args.mem:
                mem = new_mem
        elif preflight_report:
            route_note = (f"{clu.name} is not reachable -- "
                          f"{preflight_report.get('ssh_error', 'unknown error')}")
        else:
            route_note = "could not run `hpcjob preflight` (is hpcjob installed?)"

    # Smallest partial_t across specs -- the least discount, so the estimate
    # stays conservative when a job mixes partial and full diffusion.
    partial_ts = [
        float(s["partial_t"]) for s in specs.values() if s.get("partial_t") is not None
    ]
    job_partial_t = min(partial_ts) if partial_ts and len(partial_ts) == len(specs) else None

    if args.walltime:
        walltime = args.walltime
        seconds = 0
    else:
        seconds = generate.estimate_seconds(
            length=max_length,
            batch_size=batch_size if run_rfd3 else 1,
            n_batches=n_batches if run_rfd3 else 1,
            n_specs=len(specs) if run_rfd3 else 1,
            timesteps=args.timesteps if run_rfd3 else 0,
            timing=timing,
            partial_t=job_partial_t,
            # backbones x sequences each. For --mpnn-only the backbone count
            # lives on the cluster and is unknown here, so this under-counts;
            # pass --time for large re-sequencing runs.
            n_sequences=(
                batch_size * n_batches * max(len(specs), 1) * args.mpnn_seqs
                if run_mpnn and run_rfd3
                else 0
            ),
        )
        walltime = generate.format_walltime(seconds)

    try:
        if args.partition:
            partition_name = args.partition
            partition = generate.partition_by_name(clu, partition_name)
        else:
            partition = generate.find_partition(clu, gpu_sbatch, seconds)
            partition_name = str(partition["name"])
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cpus = generate.cap_cpus(partition, args.cpus)

    # ---- MPNN input location ---------------------------------------------
    if args.mpnn_only:
        if args.structures.startswith("remote:"):
            mpnn_structures = args.structures.split("remote:", 1)[1]
            local_structure_dir = None
        else:
            mpnn_structures = "./input/structures"
            local_structure_dir = Path(args.structures)
    else:
        mpnn_structures = "./output/rfd3"
        local_structure_dir = None

    # MPNN design scope. Its own default is to redesign every residue, which
    # silently ruins a scaffold redesign, so derive the scope from what RFD3 was
    # allowed to change unless the user asks otherwise.
    designed_map_path = None
    designed_summary = "whole protein (MPNN default)"
    if run_mpnn and not args.mpnn_design_all and not args.mpnn_designed:
        designed_by_spec = spec_mod.designed_residues_by_spec(specs)
        if designed_by_spec:
            designed_map_path = "./input/mpnn_designed.json"
            n = len(next(iter(designed_by_spec.values())))
            designed_summary = (f"{n} position(s) per spec, from "
                                f"select_unfixed_sequence")
    elif args.mpnn_designed:
        designed_summary = args.mpnn_designed

    omit_map_path = None
    omit_constraints = {}
    if run_mpnn and (args.mpnn_force or args.mpnn_forbid):
        try:
            if args.mpnn_force:
                omit_constraints.update(
                    spec_mod.parse_residue_constraints(args.mpnn_force, allow=True))
            if args.mpnn_forbid:
                for res_id, omitted in spec_mod.parse_residue_constraints(
                        args.mpnn_forbid, allow=False).items():
                    omit_constraints[res_id] = sorted(
                        set(omit_constraints.get(res_id, [])) | set(omitted))
        except spec_mod.SpecError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        omit_map_path = "./input/mpnn_omit.json"

    # ---- write the job dir ------------------------------------------------
    out_dir: Path = args.out_dir
    (out_dir / "input").mkdir(parents=True, exist_ok=True)

    if run_rfd3:
        (out_dir / "input" / "designs.json").write_text(json.dumps(specs, indent=2))

    if designed_map_path:
        (out_dir / "input" / "mpnn_designed.json").write_text(
            json.dumps(spec_mod.designed_residues_by_spec(specs), indent=2))

    if omit_map_path:
        (out_dir / "input" / "mpnn_omit.json").write_text(
            json.dumps(omit_constraints, indent=2))

    if sources or local_structure_dir:
        struct_dir = out_dir / "input" / "structures"
        struct_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            src = Path(source).expanduser()
            if not src.is_file():
                print(f"error: input structure not found: {src}", file=sys.stderr)
                return 1
            shutil.copy2(src, struct_dir / src.name)
        if local_structure_dir:
            if not local_structure_dir.is_dir():
                print(f"error: --structures is not a directory: {local_structure_dir}",
                      file=sys.stderr)
                return 1
            for src in sorted(local_structure_dir.iterdir()):
                if src.suffix.lower() in (".pdb", ".cif"):
                    shutil.copy2(src, struct_dir / src.name)

    rfd3_params = (
        generate.Rfd3Params(
            diffusion_batch_size=batch_size,
            n_batches=n_batches,
            num_timesteps=args.timesteps,
            seed=args.seed,
            ckpt_path=args.ckpt,
            dump_trajectories=args.dump_trajectories,
            low_memory_mode=args.low_memory,
            extra_overrides=args.rfd3_overrides,
        )
        if run_rfd3
        else None
    )

    mpnn_batch, mpnn_nbatches = _split_batches(args.mpnn_seqs, args.mpnn_batch_size)
    mpnn_params = generate.MpnnParams(
        enabled=run_mpnn,
        model_type=args.mpnn_model,
        structures=mpnn_structures,
        batch_size=mpnn_batch,
        number_of_batches=mpnn_nbatches,
        checkpoint=args.mpnn_checkpoint,
        temperature=args.mpnn_temperature,
        write_structures=not args.no_mpnn_structures,
        designed_residues=args.mpnn_designed,
        designed_map=designed_map_path,
        omit_map=omit_map_path,
    )

    slurm = generate.SlurmParams(
        job_name=name,
        time=walltime,
        partition=partition_name,
        gpus_per_node=gpu_sbatch,
        mem=mem,
        cpus_per_task=cpus,
    )

    try:
        script = generate.build_job_script(
            cluster=clu, slurm=slurm, rfd3=rfd3_params, mpnn=mpnn_params
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    job_path = out_dir / "job.sh"
    job_path.write_text(script)
    job_path.chmod(0o755)
    _copy_package(out_dir)

    # ---- report -----------------------------------------------------------
    print(f"cluster   : {clu.name}")
    if run_rfd3:
        print(f"specs     : {len(specs)} ({', '.join(specs)})")
        print(f"designs   : {batch_size} x {n_batches} = {batch_size * n_batches} per spec"
              f"  ({total_designs} total)")
        print(f"timesteps : {args.timesteps}")
        print(f"length    : {max_length} residues ({length_origin})")
    if run_mpnn:
        print(f"mpnn scope: {designed_summary}")
        if omit_constraints:
            forced = ", ".join(f"{k}({20 - len(v) + 1} allowed)"
                               for k in sorted(omit_constraints)[:6]
                               for v in [omit_constraints[k]])
            print(f"mpnn constr: {len(omit_constraints)} position(s) -- {forced}")
        print(f"mpnn      : {args.mpnn_model}, {mpnn_batch * mpnn_nbatches} seq/backbone "
              f"from {mpnn_structures}")
    print(f"resources : {partition_name}, {gpu_sbatch}, {mem}, {cpus} cpus, "
          f"walltime {walltime}")
    if route_note:
        print(f"routing   : {route_note}")
    if warn:
        print("            (largest GPU tier selected -- consider splitting the job)")
    if over_tier:
        print(f"WARNING   : design length {max_length} exceeds every configured GPU "
              f"tier on {clu.name}.")
        print("            RFD3 memory grows ~L^2 and OOMs abruptly (no graceful "
              "degradation),")
        print("            so this will likely die in seconds. Shorten the design or "
              "add a bigger tier.")
    print(f"written   : {out_dir}/")
    print()
    submit = f"hpcjob submit {job_path} --cluster {clu.name}"
    if clu.jobs_dir:
        submit += f" --jobs-dir {clu.jobs_dir}"
    print("submit with:")
    print(f"  {submit}")
    # Resubmits are common after a fast validation failure, and hpcjob prompts
    # when the remote dir exists.
    print(f"  {submit} --overwrite      # if resubmitting into the same dir")
    print()
    # `pull` defaults to ./<name>/, which is not the job dir the results belong
    # to; --dest puts output/ next to the inputs that produced it.
    print("fetch results with:")
    print(f"  hpcjob pull {name} --cluster {clu.name} --dest {out_dir} --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
