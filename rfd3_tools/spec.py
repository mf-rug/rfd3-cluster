"""Build and validate RFD3 design-input JSON.

An RFD3 inputs file is a mapping of ``{spec_name: {spec fields}}`` where each
key is one independent design configuration::

    {
      "monomer": {"length": "80-120"},
      "binder":  {"input": "target.pdb", "contig": "A1-100,50-150",
                  "select_hotspots": "A45-50,A75"}
    }

The field list and the required-field rules below mirror
``rfd3.inference.input_parsing.DesignInputSpecification`` (a pydantic model
declared with ``extra="forbid"``, so an unknown key is a hard error at runtime
on the cluster).  Validating locally turns a wasted queue slot into an
immediate error message.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Mirrors DesignInputSpecification's fields (minus `atom_array_input`, which is
# an in-memory object and cannot appear in JSON).  Keep in sync with the
# installed rfd3 version; `rfd3-setup-job --print-fields` dumps this list.
SPEC_FIELDS = {
    # data inputs, motif generation & selection
    "input",
    "contig",
    "unindex",
    "length",
    "ligand",
    "cif_parser_args",
    "extra",
    "dialect",
    # conditioning
    "select_fixed_atoms",
    "select_unfixed_sequence",
    "select_buried",
    "select_partially_buried",
    "select_exposed",
    "select_hbond_acceptor",
    "select_hbond_donor",
    "select_hotspots",
    "redesign_motif_sidechains",
    # global conditioning & symmetry
    "symmetry",
    "ori_token",
    "infer_ori_strategy",
    "plddt_enhanced",
    "is_non_loopy",
    "partial_t",
}

# Fields whose value names a file that must travel to the cluster with the job.
PATH_FIELDS = ("input",)


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------


def parse_value(raw: str) -> Any:
    """Parse a `--set key=value` value.

    JSON first (so `15.0`, `false`, `{"A431":"TIP"}`, `["a","b"]` all work),
    falling back to the bare string (so `A1-10,B1-5` needs no quoting).
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def split_variants(value: Optional[str]) -> List[Optional[str]]:
    """`"80|120|160"` -> three variants; anything else -> a single value."""
    if value is None:
        return [None]
    parts = [p.strip() for p in str(value).split("|")]
    return [p for p in parts if p] or [None]


def _slug(value: str) -> str:
    """Filesystem/JSON-key-safe suffix for a variant."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "x"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(name: str, spec: Dict[str, Any]) -> None:
    """Re-implement RFD3's own input checks, locally and cheaply.

    Mirrors ``DesignInputSpecification.validate_input_schema``.
    """
    unknown = sorted(set(spec) - SPEC_FIELDS)
    if unknown:
        raise SpecError(
            f"spec {name!r}: unknown field(s) {', '.join(unknown)}.\n"
            f"RFD3 rejects unknown keys outright (extra='forbid'). "
            f"Known fields: {', '.join(sorted(SPEC_FIELDS))}"
        )

    has = lambda k: spec.get(k) is not None  # noqa: E731

    if not (has("input") or has("contig") or has("length")):
        raise SpecError(
            f"spec {name!r}: needs at least one of 'input', 'contig' or 'length'."
        )

    if has("input") and not (
        has("contig") or has("unindex") or has("ligand") or has("partial_t")
    ):
        raise SpecError(
            f"spec {name!r}: 'input' is set but unused -- an input structure "
            f"must be consumed by 'contig', 'unindex', 'ligand' or 'partial_t'."
        )

    if not has("partial_t") and has("unindex") and not (has("contig") or has("length")):
        raise SpecError(
            f"spec {name!r}: 'unindex' needs 'contig' or 'length' alongside it "
            f"(outside partial diffusion)."
        )

    if has("length"):
        length = spec["length"]
        if not isinstance(length, (str, int)):
            raise SpecError(f"spec {name!r}: 'length' must be an int or 'min-max'.")
        if isinstance(length, str) and not re.fullmatch(r"\d+(-\d+)?", length.strip()):
            raise SpecError(
                f"spec {name!r}: 'length' must be an integer or a 'min-max' "
                f"range, got {length!r}."
            )

    if has("partial_t"):
        try:
            pt = float(spec["partial_t"])
        except (TypeError, ValueError):
            raise SpecError(f"spec {name!r}: 'partial_t' must be a number.") from None
        if pt < 0:
            raise SpecError(f"spec {name!r}: 'partial_t' must be >= 0.")


ASSUMED_LENGTH = 300  # last resort when nothing else resolves


def _contig_length(contig: str) -> Optional[int]:
    """Residue count a contig string resolves to, or None if unparseable.

    Segments are summed, taking the longest option of a variable-length one:
    ``A1-100`` (motif from the input, 100 res) + ``50-150`` (de novo, up to
    150) = 250.  A chain-qualified segment is a residue *range*, so its length
    is ``end - start + 1``; a bare numeric one is a length in its own right.
    """
    total = 0
    for token in re.split(r"[,/]", contig):
        token = token.strip()
        if not token:
            continue
        chain_range = re.fullmatch(r"[A-Za-z]+(-?\d+)-(-?\d+)", token)
        if chain_range:
            start, end = int(chain_range.group(1)), int(chain_range.group(2))
            total += abs(end - start) + 1
            continue
        if re.fullmatch(r"[A-Za-z]+-?\d+", token):
            total += 1                       # single named residue
            continue
        numeric_range = re.fullmatch(r"(\d+)-(\d+)", token)
        if numeric_range:
            total += int(numeric_range.group(2))
            continue
        if token.isdigit():
            total += int(token)
            continue
        return None                          # unrecognised -- don't guess
    return total or None


def _pdb_residue_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            count += 1
    return count


def _cif_residue_count(text: str) -> int:
    """Count CA atoms of the first model in an mmCIF `_atom_site` loop."""
    columns: List[str] = []
    count = 0
    in_loop = False
    first_model: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            columns.append(stripped.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if not in_loop:
            continue
        if not stripped or stripped.startswith(("#", "loop_", "_", "data_")):
            if columns:
                break                        # loop ended
            continue
        fields = stripped.split()
        if len(fields) != len(columns):
            continue
        row = dict(zip(columns, fields))
        if row.get("group_PDB", "ATOM") != "ATOM":
            continue
        atom = row.get("label_atom_id") or row.get("auth_atom_id") or ""
        if atom.strip('"\'') != "CA":
            continue
        model = row.get("pdbx_PDB_model_num")
        if model is not None:
            if first_model is None:
                first_model = model
            elif model != first_model:
                continue
        count += 1
    return count


def structure_residue_count(path: Path) -> Optional[int]:
    """Polymer residue count of a local PDB/mmCIF, or None if unreadable.

    Counts CA atoms, so it reports the *modelled* length -- which is what RFD3
    allocates for, and is what a partial-diffusion design ends up being.
    """
    try:
        name = path.name.lower()
        if name.endswith(".gz"):
            import gzip
            with gzip.open(path, "rt", errors="replace") as handle:
                text = handle.read()
            name = name[: -len(".gz")]
        else:
            text = path.read_text(errors="replace")
        count = (_cif_residue_count(text) if name.endswith(".cif")
                 else _pdb_residue_count(text))
        return count or None
    except (OSError, ValueError):
        return None


def design_length(
    specs: Dict[str, Dict[str, Any]],
    sources: Iterable[str] = (),
) -> Tuple[int, str]:
    """Largest design length across specs, plus how it was determined.

    Length drives the GPU tier, and `max_length` is a hard OOM cliff rather
    than a soft limit, so guessing low is expensive.  Resolved in order of
    reliability: an explicit `length`, else the contig's arithmetic, else the
    residue count of the input structure (the partial-diffusion case, where
    length is implicit in the file and used to be assumed at 300).

    `sources` are the pre-rewrite local paths from `rewrite_input_paths`; by
    the time the CLI computes resources, each spec's `input` already points at
    the not-yet-populated `./input/structures/` copy, so the originals are
    matched back by basename.
    """
    by_name = {Path(source).name: Path(source).expanduser() for source in sources}
    best, origin = 0, ""

    def better(length: int, why: str) -> None:
        nonlocal best, origin
        if length > best:
            best, origin = length, why

    for spec in specs.values():
        length = spec.get("length")
        if length is not None:
            nums = [int(n) for n in re.findall(r"\d+", str(length))]
            if nums:
                better(max(nums), "from length")
                continue

        contig = spec.get("contig")
        if isinstance(contig, str) and contig:
            resolved = _contig_length(contig)
            if resolved:
                better(resolved, "from contig")
                continue

        source = spec.get("input")
        if isinstance(source, str) and source:
            path = by_name.get(Path(source).name, Path(source).expanduser())
            counted = structure_residue_count(path)
            if counted:
                better(counted, f"counted from {path.name}")
                continue

        better(ASSUMED_LENGTH, "assumed -- not stated and no readable input")

    if not best:
        return 100, "default"
    return best, origin


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_specs(
    *,
    name: str,
    length: Optional[str] = None,
    input_path: Optional[str] = None,
    contig: Optional[str] = None,
    unindex: Optional[str] = None,
    ligand: Optional[str] = None,
    hotspots: Optional[str] = None,
    partial_t: Optional[float] = None,
    extra_sets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the spec mapping from CLI flags.

    `length` may hold `|`-separated variants, producing one spec per value
    (mirroring boltz-setup-yaml's screening syntax).
    """
    base: Dict[str, Any] = {}
    if input_path is not None:
        # The job runs with the job dir as cwd and the structure shipped into
        # input/structures/, so store the path it will have *there*.
        base["input"] = f"./input/structures/{Path(input_path).name}"
    if contig is not None:
        base["contig"] = contig
    if unindex is not None:
        base["unindex"] = unindex
    if ligand is not None:
        base["ligand"] = ligand
    if hotspots is not None:
        base["select_hotspots"] = hotspots
    if partial_t is not None:
        base["partial_t"] = float(partial_t)

    for assignment in extra_sets or []:
        if "=" not in assignment:
            raise SpecError(
                f"--set expects key=value, got {assignment!r} "
                f"(e.g. --set plddt_enhanced=false)"
            )
        key, raw = assignment.split("=", 1)
        base[key.strip()] = parse_value(raw)

    variants = split_variants(length)
    specs: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        spec = dict(base)
        if variant is not None:
            spec["length"] = variant
        key = name if len(variants) == 1 or variant is None else f"{name}_{_slug(variant)}"
        specs[key] = spec

    for key, spec in specs.items():
        validate_spec(key, spec)
    return specs


def load_spec_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a hand-written RFD3 inputs JSON (the escape hatch for exotic specs).

    Validates the same way, so a typo still fails locally rather than in the
    queue.  Symmetry, h-bond conditioning, per-atom selections etc. are all
    reachable this way without the CLI needing a flag for each.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a JSON object of {{spec_name: spec}}")
    for key, spec in data.items():
        if not isinstance(spec, dict):
            raise SpecError(f"{path}: spec {key!r} must be an object")
        validate_spec(key, spec)
    return data


def referenced_paths(specs: Dict[str, Dict[str, Any]]) -> List[str]:
    """Local files a spec refers to, which must be shipped with the job."""
    out: List[str] = []
    for spec in specs.values():
        for field_name in PATH_FIELDS:
            value = spec.get(field_name)
            if isinstance(value, str) and value:
                out.append(value)
    return out


def rewrite_input_paths(
    specs: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Point `input` at the shipped copy; return (specs, local source paths).

    A hand-written spec file may name a structure anywhere on the local disk;
    the cluster only sees what travels with the job dir.
    """
    sources: List[str] = []
    for spec in specs.values():
        value = spec.get("input")
        if not isinstance(value, str) or not value:
            continue
        if value.startswith("./input/structures/"):
            continue  # already rewritten by build_specs
        sources.append(value)
        spec["input"] = f"./input/structures/{Path(value).name}"
    return specs, sources
