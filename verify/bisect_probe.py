#!/usr/bin/env python3
"""One RFD3 memory probe: run a single design length, report peak VRAM.

Runs RFD3 in-process so `torch.cuda.max_memory_allocated()` sees the real
peak. `nvidia-smi --query-gpu` is not usable for this: on a MIG slice it
reports the parent device and includes other tenants' memory.

Only a couple of timesteps are run. Peak VRAM comes from the pairwise L x L x D
activation inside a single forward pass and is identical at every timestep, so
a 2-step run measures the same peak as a 200-step one for a fraction of the
GPU time. (Wall time is what scales with timesteps -- memory does not.)

Prints one machine-readable line:
    RESULT<TAB>length<TAB>batch<TAB>ok|oom|error<TAB>peak_bytes<TAB>seconds
"""

import os
import sys
import time

LENGTH = int(sys.argv[1])
BATCH = int(sys.argv[2])
OUT_DIR = sys.argv[3]
TIMESTEPS = int(os.environ.get("PROBE_TIMESTEPS", "2"))


def emit(status: str, peak: int, seconds: float) -> None:
    print(f"RESULT\t{LENGTH}\t{BATCH}\t{status}\t{peak}\t{seconds:.1f}", flush=True)


def main() -> int:
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    status = "ok"

    # `specification` is an empty mapping in the shipped config and OmegaConf
    # runs in struct mode, so adding a key needs the leading `+`. Every other
    # key already exists and must NOT have one.
    sys.argv = [
        "rfd3", "design",
        "inputs=null",
        f"out_dir={OUT_DIR}/L{LENGTH}_B{BATCH}",
        f"+specification.length={LENGTH}",
        f"diffusion_batch_size={BATCH}",
        "n_batches=1",
        f"inference_sampler.num_timesteps={TIMESTEPS}",
        "skip_existing=False",
        "dump_trajectories=False",
    ]

    try:
        from rfd3.cli import app
        app()
    except SystemExit as exc:                     # typer exits normally
        if exc.code not in (0, None):
            status = "error"
    except torch.cuda.OutOfMemoryError:
        status = "oom"
    except RuntimeError as exc:                   # older torch raises plain RuntimeError
        status = "oom" if "out of memory" in str(exc).lower() else "error"
    except Exception as exc:                       # noqa: BLE001 - report, never crash the sweep
        print(f"# unexpected: {type(exc).__name__}: {exc}", file=sys.stderr)
        status = "error"

    emit(status, torch.cuda.max_memory_allocated(), time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
