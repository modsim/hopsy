#!/usr/bin/env bash
# Run the two-model FluxML validation with one eight-worker replicate at a time.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: run_nrpt_benchmarks.sh [--smoke | --reference-8x] [OUTPUT_DIR]

Default: 250,000 retained cold draws per model and seed, about 9 hours total
         on the measured Ryzen 5950X. Four seeds and 2,000 warm-up scans.
--smoke: 1,000 retained draws, 20 warm-up scans and two seeds per model.
--reference-8x: 960,000 retained draws per model and seed, above 10 hours total.

Pass the same OUTPUT_DIR to resume at the first incomplete replicate.
HOPSY_PYTHON and HOPSY_BINDINGS override the venv interpreter and build directory.
NRPT_NATPROT_DRAWS and NRPT_GLUC_DRAWS override the default profile's draw counts.
USAGE
}

profile=overnight
natprot_draws=${NRPT_NATPROT_DRAWS:-250000}
gluc_draws=${NRPT_GLUC_DRAWS:-250000}
warmup=2000
seeds=(42 136 271 314)
weights=(0 0.33 0.67 0.995)
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --smoke)
        profile=smoke
        natprot_draws=1000
        gluc_draws=1000
        warmup=20
        seeds=(42 136)
        weights=(0 0.67)
        shift ;;
    --reference-8x)
        profile=reference-8x
        natprot_draws=960000
        gluc_draws=960000
        shift ;;
    -*) usage >&2; exit 2 ;;
esac
if (( $# > 1 )) || [[ ${1:-} == -* ]]; then usage >&2; exit 2; fi
for count in "$natprot_draws" "$gluc_draws"; do
    if ! [[ $count =~ ^[1-9][0-9]{3,}$ ]]; then
        echo "Expected at least 1000 retained draws." >&2
        exit 2
    fi
done

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd -- "$repo"
bench="$repo/cpp/benchmarks"
runner="$bench/benchmark_flux_convergence.py"
python=$(command -v "${HOPSY_PYTHON:-$repo/cpp/.benchmark-venv/bin/python}") || {
    echo "Build the Python bindings and set HOPSY_PYTHON if using another environment." >&2
    exit 1
}
bindings=${HOPSY_BINDINGS:-$repo/cpp/build/python/bindings}
if [[ ! -d $bindings ]]; then
    echo "Binding build directory missing: $bindings. See cpp/benchmarks/README.md." >&2
    exit 1
fi

mkdir -p cpp/build
out=${1:-}
if [[ -z $out ]]; then out=$(mktemp -d "$repo/cpp/build/nrpt-benchmarks.XXXXXX"); fi
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd -P)
exec 9>"$out/.lock"
flock -n 9 || { echo "Another launcher is using $out" >&2; exit 1; }
exec > >(tee -a "$out/run.log") 2>&1

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
# Linux uptime provides a monotonic wall timer in centiseconds.
wall_ticks() {
    local uptime idle
    read -r uptime idle < /proc/uptime
    printf '%s\n' "${uptime/./}"
}
trap 'status=$?; if (( status )); then log "Run stopped (status $status). Completed replicates are in $out."; fi' EXIT
log "Profile $profile. Results in $out"
if [[ $profile == reference-8x ]]; then
    log "The eight-times-draws profile exceeds the ten-hour budget."
fi
invocation_start=$(wall_ticks)

export PYTHONPATH="$bindings"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$out/matplotlib-cache}"

workers=8
cpus=$("$python" - "$workers" <<'PY'
import os
import subprocess
import sys

workers = int(sys.argv[1])
allowed = os.sched_getaffinity(0)
physical = {}
for line in subprocess.check_output(
    ["lscpu", "--parse=CPU,CORE,SOCKET,ONLINE"], text=True
).splitlines():
    if line.startswith("#"):
        continue
    cpu, core, socket, online = line.split(",")
    if online == "Y" and int(cpu) in allowed:
        physical.setdefault((socket, core), int(cpu))
if len(physical) < workers:
    sys.exit(f"Need {workers} available physical cores, found {len(physical)}.")
print(",".join(map(str, sorted(physical.values())[:workers])))
PY
)
log "One replicate at a time on physical CPUs $cpus"
binding=$("$python" -c 'import _hopsy_core; print(_hopsy_core.__file__)')

mode=nrpt_parallel_rounded_reference
reference_max_beta=0.5
hit_and_run_steps=20
step_size=0.15
nat_betas=(0 0.005 0.02 0.06 0.16 0.36 0.65 1)
gluc_betas=(0 0.000001 0.00001 0.0001 0.001 0.01 0.1 1)
gluc_caps=(100 10000)
nat_name=Ecoli_NatProt_GOX.fml
gluc_name='e_coli_+EDP+GOX+MGOX-TPI.fml'
nat_dir="$out/natprot-gox"
gluc_dir="$out/edp-gox-mgox-tpi-12gluc"

# Resume only with the same inputs, settings and compiled extension.
config=$(mktemp "$out/.configuration.XXXXXX")
{
    printf 'git_commit\t%s\nprofile\t%s\nphysical_cpus\t%s\npython\t%s\n' \
        "$(git rev-parse HEAD)" "$profile" "$cpus" "$python"
    declare -p workers mode reference_max_beta hit_and_run_steps step_size
    declare -p seeds weights warmup natprot_draws gluc_draws nat_betas gluc_betas gluc_caps
    printf 'natprot_configuration\tdefault\ngluc_configuration\t[12]Gluc\n'
    "$python" - <<'PY'
from importlib.metadata import version
import platform

print(f"python_version\t{platform.python_version()}")
for package in ("x3cflux", "hopsy", "numpy", "scipy", "arviz", "matplotlib"):
    print(f"{package}\t{version(package)}")
PY
    sha256sum -- "$bench/run_nrpt_benchmarks.sh" "$runner" "$bench/flux_benchmark_common.py" \
        "$binding" "$bench/$nat_name" "$bench/$gluc_name"
} > "$config"
if [[ -e "$out/configuration.txt" ]]; then
    if ! cmp -s -- "$config" "$out/configuration.txt"; then
        log "Configuration differs from $out/configuration.txt. Use the original settings or a new directory."
        exit 1
    fi
    rm -- "$config"
else
    mv -- "$config" "$out/configuration.txt"
fi

mkdir -p -- "$nat_dir" "$gluc_dir"
"$python" - "$bench/$nat_name" "$nat_dir/$nat_name" "$bench/$gluc_name" "$gluc_dir/$gluc_name" <<'PY'
from pathlib import Path
import re
import sys

for i in (1, 3):
    source, output = map(Path, sys.argv[i:i + 2])
    data = source.read_bytes()
    if i == 3:
        data = re.sub(rb'\s+xsi:schemaLocation="[^"]*"', b'', data, count=1)
    if output.exists():
        if output.read_bytes() != data:
            sys.exit(f"Existing model copy differs from the planned input: {output}")
    else:
        pending = output.with_suffix(".fml.tmp")
        pending.write_bytes(data)
        pending.replace(output)
PY

elapsed() {
    local ticks=$(( 10#$(wall_ticks) - 10#$1 ))
    printf '%s.%02d\n' "$((ticks / 100))" "$((ticks % 100))"
}

run_model() {
    local directory=$1 fml=$2 configuration=$3 draws=$4
    shift 4
    local tag=${directory##*/} i seed replicate attempt started file
    local -a replicate_dirs=()
    for i in "${!seeds[@]}"; do
        seed=${seeds[i]}
        replicate="$directory/seed-$seed"
        replicate_dirs+=("$replicate")
        if [[ -d $replicate ]]; then
            for file in report.json wall-seconds.txt "$mode.npy" "${mode}_replicas.npz"; do
                if [[ ! -s $replicate/$file ]]; then
                    log "Incomplete published result $replicate/$file. Move this seed directory aside to rerun it."
                    return 1
                fi
            done
            log "Keeping completed $tag seed $seed"
            continue
        fi
        attempt=$(mktemp -d "$directory/.seed-$seed.XXXXXX")
        log "Starting $tag seed $seed with $draws retained cold draws"
        started=$(wall_ticks)
        taskset -c "$cpus" "$python" "$runner" run \
            --fml "$fml" --configuration "$configuration" "$@" \
            --method "$mode" --workers "$workers" --reference-max-beta "$reference_max_beta" \
            --hit-and-run-steps "$hit_and_run_steps" --step-size "$step_size" \
            --num-samples "$draws" --warmup "$warmup" --seed "$seed" --fit-weight "${weights[i]}" \
            --output "$attempt" 2>&1 | tee "$attempt/run.log"
        elapsed "$started" > "$attempt/wall-seconds.txt"
        mv -- "$attempt" "$replicate"
        log "Saved $tag seed $seed"
    done
    if [[ ! -s $directory/summary.json || ! -s $directory/summary.md ]]; then
        log "Computing $tag diagnostics and posterior plots"
        "$python" "$runner" summarize --sequential-model "$tag" \
            --output "$directory" "${replicate_dirs[@]}" 2>&1 | tee "$directory/summary.log"
    fi
    log "$tag results in $directory/summary.md"
}

run_model "$nat_dir" "$nat_dir/$nat_name" default "$natprot_draws" --betas "${nat_betas[@]}"
run_model "$gluc_dir" "$gluc_dir/$gluc_name" '[12]Gluc' "$gluc_draws" \
    --betas "${gluc_betas[@]}" --unbounded-flux-caps "${gluc_caps[@]}"
elapsed "$invocation_start" > "$out/latest-invocation-wall-seconds.txt"
log "Both models complete in $out"
