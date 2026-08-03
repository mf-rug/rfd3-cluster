#!/bin/bash
#SBATCH --job-name=rfd3_bisect
#SBATCH --time=00:40:00
#SBATCH --partition=CHANGEME_gpu_partition
#SBATCH --gpus-per-node=CHANGEME_gputype:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#SBATCH --output=slurm-%j.out
#
# Find the largest RFD3 design length that fits on this GPU.
#
# Known bracket on A100-40GB: 442 tokens ran fine in production, 600 OOMs.
# Ascending sweep with early stop rather than a true binary search: probes are
# seconds each (2 timesteps -- peak VRAM is per forward pass, so it does not
# depend on timestep count), an OOM aborts quickly, and a full curve lets the
# other tiers be set by extrapolation instead of guessed.
#
# Cluster-agnostic apart from the four values below and the SBATCH header.

VENV=${RFD3_VENV:?set RFD3_VENV to the cluster Foundry venv}
CKPT_DIR=${RFD3_CKPT_DIR:?set RFD3_CKPT_DIR to the checkpoint directory}
MODULE=${RFD3_MODULE:-}
LENGTHS="460 480 500 520 540 560 580 600"

_job_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
cd "$_job_dir"

module purge
[ -n "$MODULE" ] && module load $MODULE
export FOUNDRY_CHECKPOINT_DIRS="$CKPT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# $TMPDIR is deleted when the job ends, so results are written to the job dir.
WORK="${TMPDIR:-/tmp}/bisect_$$"
RESULTS="$_job_dir/bisect_results.tsv"
mkdir -p "$WORK"
printf "length\tbatch\tstatus\tpeak_bytes\tpeak_GiB\tseconds\n" > "$RESULTS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_probe() {          # run_probe <length> <batch> -> echoes status
    local len=$1 batch=$2
    # Each probe is a separate process: an OOM leaves the CUDA context in a
    # state the next probe should not have to trust.
    local line
    line=$("$VENV/bin/python" "$_job_dir/bisect_probe.py" "$len" "$batch" "$WORK" \
           2>>"$_job_dir/probe_errors.log" | grep '^RESULT' | tail -1)
    if [ -z "$line" ]; then
        echo "died"
        printf "%s\t%s\tdied\t0\t0.00\t0\n" "$len" "$batch" >> "$RESULTS"
        return
    fi
    local L B status peak secs
    IFS=$'\t' read -r _ L B status peak secs <<< "$line"
    printf "%s\t%s\t%s\t%s\t%.2f\t%s\n" "$L" "$B" "$status" "$peak" \
        "$(awk -v p="$peak" 'BEGIN{print p/1073741824}')" "$secs" >> "$RESULTS"
    echo "$status"
}

echo "=== ascending sweep at diffusion_batch_size=8 (production setting) ==="
LAST_OK=""
for L in $LENGTHS; do
    printf "  L=%-4s " "$L"
    st=$(run_probe "$L" 8)
    echo "$st"
    case "$st" in
        ok)  LAST_OK=$L ;;
        oom) echo "  -> OOM at L=$L; cliff is between ${LAST_OK:-?} and $L"; break ;;
        *)   echo "  -> probe failed (not an OOM); see probe_errors.log"; break ;;
    esac
done

# Batch dependence has only ever been measured well below the limit (+1% VRAM
# at L=200/400). If it widens at the boundary, the tier table needs a second
# dimension -- so check it exactly there.
if [ -n "$LAST_OK" ]; then
    echo "=== batch dependence at the boundary (L=$LAST_OK) ==="
    for B in 1 4; do
        printf "  B=%-2s " "$B"
        echo "$(run_probe "$LAST_OK" "$B")"
    done
fi

echo
echo "=== results ==="
column -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
echo
echo "largest length that fit at batch 8: ${LAST_OK:-none}"
echo "suggested tier cap (10-15% headroom): $(awk -v l="${LAST_OK:-0}" 'BEGIN{printf "%d", l*0.875}')"
rm -rf "$WORK"
