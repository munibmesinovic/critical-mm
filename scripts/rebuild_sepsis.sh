#!/usr/bin/env bash
# rebuild_sepsis.sh — rebuild the SEP-3 cascade outputs for one dataset
# under a cgroup memory cap. Always use this instead of bare
# `python -c "Sepsis.build(...)"` from a shell — uncapped rebuilds
# have OOM-killed the user cgroup and broken the SSH login session
# (see "Phase 1 interim builds" follow-up +
# postmortem).
#
# Usage:
# scripts/rebuild_sepsis.sh <dataset> [--skip-abx_duration] [--uncapped]
# where <dataset> ∈ {eicu, miiv, hirid, nwicu}
#
# Pass --uncapped to skip the mem_run.sh self-wrap. Use this for datasets
# whose natural peak exceeds the cgroup accounting limits (hirid sepsis
# naturally peaks at ~102 GiB; cgroup accounting adds overhead that
# pushes it past any reasonable cap). The safety tradeoff: an uncapped
# rebuild that runs away will pressure user.slice and may break SSH —
# but for known-good workloads on an otherwise idle host, it's safe.
#
# Env overrides:
# MEM_MAX hard cap for the scope. If unset, the script picks a per-dataset
# default that's empirically safe under cold cache:
# eicu → 110G (events_long pivot is the heaviest)
# miiv → 80G
# hirid → 100G (full SEP-3 cascade on 32k stays + SOFA pivot)
# nwicu → 30G (small dataset)
#
# What it does (in order, per the ricu-faithful path):
# 1. read_abx_duration → data/interim/<ds>/abx_duration.parquet
# (cached; rebuilt only on cache miss)
# 2. build_base_cohort → data/processed/base_cohort/<ds>/stays.parquet
# 3. Sepsis.build → data/processed/sepsis/<ds>/{sta,dyn,outc}.parquet (CM-native)
# 4. export_yaib(sepsis) → same paths, overwritten to YAIB-shape
#
# Idempotent. Safe to re-run after a failure.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve per-dataset MEM_MAX before the self-wrap (so the cap is set
# correctly when the script re-execs inside mem_run.sh).
_resolve_mem_max {
    if [ -n "${MEM_MAX:-}" ]; then echo "$MEM_MAX"; return; fi
    # Caps empirically set above each dataset's natural memory peak
    # measured via uncapped trace on 2026-05-20 (host has 123 GiB RAM,
    # we leave 3-8 GiB headroom for OS / other users):
    # hirid natural peak 102.3 GiB → cap 120G
    # eicu natural peak (unknown, similar magnitude expected) → 120G
    # miiv natural peak ~50 GiB → 80G
    # nwicu natural peak ~5 GiB → 30G
    # The cap exists for ISOLATION (kill the scope cleanly, not the
    # SSH session) — it's not meant to make the workload fit in less
    # memory than it needs.
    case "${1:-}" in
        eicu) echo "120G";;
        miiv) echo "80G";;
        hirid) echo "120G";;
        nwicu) echo "30G";;
        *) echo "120G";; # safe-ish default for unknown
    esac
}

# Self-wrap in a cgroup memory scope unless already inside one.
#
# Why we set MEM_SWAP_MAX to the host's swap instead of the mem_run.sh
# default 0: polars' parquet reads mmap the input file, and intermediate
# pivot/join buffers are anonymous heap. With MEM_SWAP_MAX=0 inside the
# cgroup, anonymous pages can't be swapped — even when natural use is
# fine, brief allocation spikes can OOM-kill the scope. Allowing the
# host's ~1.9 GB swap to be visible to the cgroup eliminates the spurious
# kills (verified on hirid: 55 s uncapped vs OOM under MEM_MAX=115G with
# MEM_SWAP_MAX=0). MEM_MAX still bounds the scope so a runaway can't take
# down the host's SSH session.
# Quick scan for --uncapped before the self-wrap decision (real arg
# parsing happens after the wrap re-exec).
_uncapped_flag=""
for _a in "$@"; do
    if [ "$_a" = "--uncapped" ]; then _uncapped_flag=1; fi
done
if [ -z "${CMM_MEM_RUN_ACTIVE:-}" ] && [ -z "$_uncapped_flag" ] && [ -x "$REPO/scripts/mem_run.sh" ]; then
    _mm="$(_resolve_mem_max "${1:-}")"
    # MEM_HIGH at 60% of MEM_MAX triggers cgroup reclaim before the hard
    # kill. polars mmap of parquet files puts file-backed pages into the
    # cgroup's accounting; without an early reclaim signal, the kernel
    # only frees them at memory.max and by then it's too late and the
    # scope gets OOM-killed even when natural use is fine. With MEM_HIGH
    # set, the kernel starts throttling allocations + reclaiming cache
    # pages once usage crosses it, giving the process room to breathe.
    _mh="${MEM_HIGH:-}"
    if [ -z "$_mh" ]; then
        # Derive MEM_HIGH = 60% of MEM_MAX. Strip the trailing letter,
        # multiply, append the same suffix.
        _mm_num="${_mm%[GMK]}"
        _mm_unit="${_mm: -1}"
        _mh="$(awk -v n="$_mm_num" 'BEGIN{printf "%d", n*0.6}')$_mm_unit"
    fi
    exec env MEM_MAX="$_mm" MEM_HIGH="$_mh" MEM_SWAP_MAX="${MEM_SWAP_MAX:-1500M}" \
        "$REPO/scripts/mem_run.sh" "self-rebuild-sepsis-${1:-unknown}" \
        bash "$0" "$@"
fi

# Use the project's conda env if available; fall back to PATH python.
PY="$HOME/miniforge3/envs/critical-mm/bin/python"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PY" ]]; then
    echo "no python interpreter found" >&2
    exit 3
fi

DATASET=""
SKIP_ABX=""
UNCAPPED=""
for arg in "$@"; do
    case "$arg" in
        --skip-abx_duration) SKIP_ABX="--skip-abx_duration";;
        --uncapped) UNCAPPED="1";;
        --*) echo "unknown flag: $arg" >&2; exit 2;;
        *) DATASET="$arg";;
    esac
done
if [ -z "$DATASET" ]; then
    echo "usage: $0 <eicu|miiv|hirid|nwicu> [--skip-abx_duration] [--uncapped]" >&2
    exit 2
fi
case "$DATASET" in
    eicu|miiv|hirid|nwicu);;
    *) echo "unknown dataset: $DATASET" >&2; exit 2;;
esac

LOG_DIR="$REPO/logs/rebuilds"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rebuild_sepsis_${DATASET}_$(date +%Y%m%dT%H%M%S).log"
echo "rebuild_sepsis $DATASET log=$LOG_FILE MEM_MAX=${MEM_MAX:-(scope default)}"

cd "$REPO"

"$PY" - <<PY 2>&1 | tee "$LOG_FILE"
import time
from pathlib import Path

DATASET = "${DATASET}"
SKIP_ABX = "${SKIP_ABX}" == "--skip-abx_duration"

def ts -> str:
    return time.strftime("%H:%M:%S")


def reader_for(ds):
    # Local import so a syntax-error in any reader doesn't block the others.
    if ds == "eicu":
        from critical_mm.datasets.eicu import EICUReader
        return EICUReader(raw_root=Path("data/raw/eicu-crd-2.0"),
                          interim_root=Path("data/interim"),
                          repo_root=Path("."))
    if ds == "miiv":
        from critical_mm.datasets.mimic_iv import MIMICIVReader
        return MIMICIVReader(raw_root=Path("data/raw/mimic-iv-3.1"),
                             interim_root=Path("data/interim"),
                             repo_root=Path("."))
    if ds == "hirid":
        from critical_mm.datasets.hirid import HiRIDReader
        return HiRIDReader(raw_root=Path("data/raw/hirid-1.1.1"),
                           interim_root=Path("data/interim"),
                           repo_root=Path("."))
    if ds == "nwicu":
        from critical_mm.datasets.nwicu import NWICUReader
        return NWICUReader(raw_root=Path("data/raw/nwicu-0.1.0"),
                           interim_root=Path("data/interim"),
                           repo_root=Path("."))
    raise ValueError(f"unknown dataset: {ds}")


# Step 1: abx_duration (cached; cache hit on second run is ~1s).
if not SKIP_ABX:
    t0 = time.time
    print(f"[{ts}] step 1/4: read_abx_duration({DATASET})...", flush=True)
    reader_for(DATASET)._harmonise_one("abx_duration", concepts=[], force=False)
    print(f"[{ts}] step 1/4: ok ({time.time-t0:.1f}s)", flush=True)
else:
    print(f"[{ts}] step 1/4: SKIPPED (--skip-abx_duration)", flush=True)

# Step 2: base cohort.
from critical_mm.cohorts.base import build_base_cohort
t0 = time.time
print(f"[{ts}] step 2/4: build_base_cohort({DATASET})...", flush=True)
build_base_cohort(dataset=DATASET, interim_root=Path("data/interim"),
                  processed_root=Path("data/processed"), repo_root=Path("."))
print(f"[{ts}] step 2/4: ok ({time.time-t0:.1f}s)", flush=True)

# Step 3: Sepsis.build.
from critical_mm.tasks import Sepsis
t0 = time.time
print(f"[{ts}] step 3/4: Sepsis.build({DATASET})...", flush=True)
Sepsis.build(dataset=DATASET, interim_root=Path("data/interim"),
               processed_root=Path("data/processed"), repo_root=Path("."))
print(f"[{ts}] step 3/4: ok ({time.time-t0:.1f}s)", flush=True)

# Step 4: YAIB-shape export (overwrites the CM-native outc/dyn/sta).
from critical_mm.exports.yaib import export_yaib
t0 = time.time
print(f"[{ts}] step 4/4: export_yaib(sepsis, {DATASET})...", flush=True)
export_yaib(task="sepsis", dataset=DATASET,
            base_cohort_root=Path("data/processed/base_cohort"),
            task_output_root=Path("data/processed"),
            yaib_output_root=Path("data/processed"))
print(f"[{ts}] step 4/4: ok ({time.time-t0:.1f}s)", flush=True)

print(f"[{ts}] DONE: data/processed/sepsis/{DATASET}/{{sta,dyn,outc}}.parquet in YAIB shape", flush=True)
PY
