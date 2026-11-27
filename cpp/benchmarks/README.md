# Benchmarks

## Non-reversible parallel tempering

### Native C++

Install the [C++ build prerequisites](../README.md#build-install-and-test).
From the repository root, build and run the two standalone benchmark programs.

```sh
cmake -S cpp -B cpp/build/native-benchmarks -DCMAKE_BUILD_TYPE=Release \
  -DHOPSY_BUILD_CPP_TESTS=OFF -DHOPSY_BUILD_CPP_BENCHMARKS=ON
cmake --build cpp/build/native-benchmarks --parallel 4
cpp/build/native-benchmarks/hopsy_parallel_tempering_benchmark
cpp/build/native-benchmarks/hopsy_parallel_tempering_comparison \
  > cpp/build/native-benchmarks/parallel_tempering_comparison.csv
```

The first program measures exchange-only and full-scan timing on a truncated
Gaussian and reports ESS, exchange counts and round trips across the kernel
families. The comparison measures elapsed sampling time and ESS/s for eight
independent cold chains and eight-slot NRPT on two truncated Gaussian targets.
Run the programs sequentially, without competing builds. Use the same machine,
compiler and flags when comparing runs.

### FluxML benchmarks

#### Install

Build the [Python bindings](../README.md#python-bindings), then add the benchmark
dependencies to the same environment. Use an interpreter that matches the
13CFLUX wheel. Run these commands from the repository root.

```sh
hopsy_python="$PWD/cpp/.benchmark-venv/bin/python"
"$hopsy_python" -m pip install scipy matplotlib arviz
export PYTHONPATH="$PWD/cpp/build/python/bindings"
```

The scripts import `_hopsy_core` from the build directory and use the separate
legacy `hopsy` package for rounding. Use the build directory on `PYTHONPATH`
so the installed 2.0 package does not shadow legacy hopsy.

##### 13CFLUX native target wheel

Use a wheel from the [`201-implement-hopsy-native-target` branch](https://jugit.fz-juelich.de/IBG-1/ModSim/Fluxomics/13CFLUX/-/tree/201-implement-hopsy-native-target?ref_type=heads) that matches your platform and Python ABI. Install the actual wheel file with `"$hopsy_python" -m pip install /path/to/downloaded.whl`. The package imports as `x3cflux`. The wheel requires the native libraries it was linked against, and installing `cyipopt` from source requires Ipopt development files and `pkg-config`.

To build from the branch, use an environment with the 13CFLUX build dependencies. Its CMake project requires Boost compiled components (Log, DateTime and Test), SUNDIALS 6 or later, and FluxML. It can fetch Eigen if unavailable. The branch's `pyproject.toml` supplies the Python build dependencies and the CMake wheel target. Set `CMAKE_PREFIX_PATH` for prerequisites installed in nonstandard prefixes.

```sh
mkdir -p cpp/build
git clone --single-branch --branch 201-implement-hopsy-native-target \
  https://jugit.fz-juelich.de/IBG-1/ModSim/Fluxomics/13CFLUX.git cpp/build/13cflux-native-target
wheel_dir="$(mktemp -d cpp/build/x3cflux-wheels.XXXXXX)"
"$hopsy_python" -m pip wheel --no-deps ./cpp/build/13cflux-native-target --wheel-dir "$wheel_dir"
"$hopsy_python" -m pip install "$wheel_dir"/x3cflux-*.whl
git -C cpp/build/13cflux-native-target rev-parse HEAD
```

#### Run

The two model files are in `cpp/benchmarks/`. Use `--fml` to select a model. Both scripts accept `--configuration`, `--betas`, `--num-samples`, `--warmup` and `--seed`. A sample is one retained cold draw. β lists include 0 and 1 and determine the number of temperature slots. An independent replicate is a separate invocation with its own seed, not a temperature slot.

Both scripts require at least 1,000 retained draws for batch-means ESS. The single-seed script defaults to 100,000 draws and 1,000 warm-up iterations. The replicated runner defaults to 20,000 draws and 2,000 warm-up iterations. Warm-up uses the same local step counts as sampling and its draws are discarded.

**Single-seed comparison.** `benchmark_ecoli_nrpt.py` compares a standalone chain to serial or parallel NRPT via `--nrpt-mode {parallel,serial,both}`. Select `--kernel {hit-and-run,gaussian-hit-and-run,csmmala,both}`, where `both` runs dense uniform Hit-and-Run and CSmMALA. `--step-size` sets the Gaussian line scale or CSmMALA proposal scale. `--round` rounds the whole problem. For CSmMALA in model coordinates use `--reference-kernel {same,rounded-coordinate,both}` with `--reference-max-beta` to select the hottest slots for rounded Hit-and-Run. `--reference-steps` sets the step count at β = 0, `--local-steps` sets it at every positive β, and `--standalone-thinning` controls standalone steps per draw. With no explicit β list, `--num-replicas` sets the length of a linear ladder. JSON goes to stdout. Use `--replica-trace` for an NPZ and `--posterior-plots` for SVGs.

**Replicated comparison.** `benchmark_flux_convergence.py run` uses rounded uniform coordinate Hit-and-Run for the standalone and all-Hit-and-Run controls. Use `--method {standalone,nrpt_parallel,nrpt_parallel_rounded_reference}` to select one strategy. Omitting it runs all three. `--mixed-only` is shorthand for selecting the mixed method. In mixed runs, `--reference-max-beta` selects the inclusive Hit-and-Run cutoff and `--hit-and-run-steps` gives the number of steps at each of those slots. Every higher slot takes one CSmMALA step at `--step-size` (default 0.15). `--fit-weight` sets the start between the interior point (0) and the FluxML initial point (near 1).

`--workers` controls native workers per NRPT replicate. A worker explores slots at strides of the worker count before joining the exchange barrier. Each physical replica has a separate native capsule context, so local exploration can run without the GIL. When running processes concurrently, budget the sum of their workers and cap numerical library threads.

#### Short NatProt smoke test

This runs two independent chains of the mixed method, checks the installed capsule against the actual NatProt model, and creates convergence summaries and figures. From the repository root, with the venv and bindings set up as above, run in Bash.

```bash
hopsy_python="$PWD/cpp/.benchmark-venv/bin/python"
export PYTHONPATH="$PWD/cpp/build/python/bindings"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
runner=cpp/benchmarks/benchmark_flux_convergence.py
(
  set -e
  out="$(mktemp -d cpp/build/natprot-smoke.XXXXXX)"
  for seed in 42 136; do
    weight=0.33
    if [ "$seed" -eq 136 ]; then weight=0.67; fi
    "$hopsy_python" -u "$runner" run \
      --fml 'cpp/benchmarks/Ecoli_NatProt_GOX.fml' --configuration default \
      --betas 0 0.005 0.02 0.06 0.16 0.36 0.65 1 \
      --method nrpt_parallel_rounded_reference --reference-max-beta 0.5 \
      --hit-and-run-steps 2 --workers 4 --num-samples 1000 --warmup 20 \
      --seed "$seed" --fit-weight "$weight" --output "$out/seed-$seed"
  done
  "$hopsy_python" "$runner" summarize --output "$out" "$out/seed-42" "$out/seed-136"
  printf '%s\n' "$out"
)
```

Use four distinct seeds for the established convergence comparison. The smoke test checks the execution path and file formats.

#### Harder model smoke test

The supplied harder file's schema hint points to a URL that the tested FluxML parser could not load. Use a local copy without that hint. The `[12]Gluc` configuration also needs a bounded reference. The following caps reproduce the earlier benchmark's **capped** model and change its posterior support. The report lists the added inequalities.

```bash
hopsy_python="$PWD/cpp/.benchmark-venv/bin/python"
export PYTHONPATH="$PWD/cpp/build/python/bindings"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
runner=cpp/benchmarks/benchmark_flux_convergence.py
hard_out="$(mktemp -d cpp/build/gluc-smoke.XXXXXX)"
hard_name='e_coli_+EDP+GOX+MGOX-TPI.fml'
"$hopsy_python" - "cpp/benchmarks/$hard_name" "$hard_out/$hard_name" <<'PY'
from pathlib import Path
import re
import sys

data = Path(sys.argv[1]).read_bytes()
data = re.sub(rb'\s+xsi:schemaLocation="[^"]*"', b'', data, count=1)
Path(sys.argv[2]).write_bytes(data)
PY
"$hopsy_python" -u "$runner" run \
  --fml "$hard_out/$hard_name" --configuration '[12]Gluc' --unbounded-flux-caps 100 10000 \
  --betas 0 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 \
  --method nrpt_parallel_rounded_reference --reference-max-beta 0.5 \
  --hit-and-run-steps 2 --workers 4 --num-samples 1000 --warmup 20 \
  --seed 42 --fit-weight 0.33 --output "$hard_out/seed-42"
```

#### Reading the output

- `run` writes `report.json`, one cold sample matrix per method as NPY (original free parameters × retained draws), and NRPT trace NPZs (scan × slot). Traces start with the post-warm-up ensemble and record replica IDs and untempered densities after swaps. The report gives exchange counts, round trips per physical replica and native draw time. `sampling_seconds` includes warm-up plus retained draws. `elapsed_seconds` also includes setup and most output work.
- `summarize` requires at least two matching, distinctly seeded replicates. It writes `summary.json` with ArviZ rank-normalized and folded split R̂ and bulk ESS, keeping the independent cold chains distinct for diagnostics. It checks ESS thresholds of 100 per chain and 50 per half-chain for every free flux. Recorded ESS rates use summed retained-draw timer values. Measure elapsed wall time around a concurrent batch when comparing throughput.
- The SVGs overlay normalized histograms by method for varying net and exchange fluxes reconstructed in original model coordinates. Fixed flux values are listed in the report. Round trips measure movement along the ladder. Check R̂ and ESS alongside them.

#### Overnight validation

After building the Python bindings and installing the native-target 13CFLUX
wheel, run this from the repository root on Linux with `taskset`, `lscpu`, and
`flock` available.

```bash
cpp/benchmarks/run_nrpt_benchmarks.sh cpp/build/nrpt-overnight
```

For a short end-to-end check, use a separate directory.

```bash
cpp/benchmarks/run_nrpt_benchmarks.sh --smoke cpp/build/nrpt-overnight-smoke
```

The overnight profile runs the two models in sequence, with one independent
replicate at a time and eight native workers pinned to eight distinct physical
cores. Numerical libraries are limited to one thread. Each worker advances one
temperature slot, then the workers synchronize for the usual alternating swaps.
The launcher uses seeds `42, 136, 271, 314`, initialization weights
`0, 0.33, 0.67, 0.995`, 2,000 warm-up scans, and the ladders in the smoke
examples above. Slots with β ≤ 0.5 take 20 rounded coordinate Hit-and-Run
steps. Slots above the cutoff take one CSmMALA step at scale 0.15. The harder
model uses `[12]Gluc` with the same explicit caps shown above. The launcher
copies both inputs into the result directories and removes the external
schema hint from the harder copy.

The reference NatProt run kept 120,000 cold draws per seed and took about one
hour with four replicates running concurrently and ten Hit-and-Run steps per
scan. Eight times that draw count is 960,000 per seed. For the new sequential
20-to-1 configuration, four 2,000-draw pilots per model on the Ryzen 5950X
measured mean retained scan times of 18.124 ms for NatProt and 13.991 ms for
`[12]Gluc`. Multiplying each rate by four seeds and by the draws plus warm-up
gives these sampling estimates.

| Retained draws per seed per model | NatProt GOX | Capped `[12]Gluc` | Combined sampling |
|---:|---:|---:|---:|
| 250,000 (default) | 5 h 04 min | 3 h 55 min | **9 h 00 min** |
| 960,000 (`--reference-8x`) | 19 h 22 min | 14 h 57 min | **34 h 20 min** |

The default targets the 8–10 h window. Setup, diagnostics and plots add time,
and elapsed time depends on CPU load and the trajectories. A fixed draw count
gives each seed the same sample size. On a different machine, use
`NRPT_NATPROT_DRAWS` and `NRPT_GLUC_DRAWS` to set the default profile's counts
from a local pilot. The optional `HOPSY_PYTHON` and `HOPSY_BINDINGS` variables
select another interpreter or binding build.

Monitor both models from another terminal.

```bash
tail -F cpp/build/nrpt-overnight/run.log
```

The output includes `configuration.txt` with the Git commit, seeds, settings,
package versions and input and extension hashes. Each `seed-*` directory is
published after its draws, replica trace, report, and timing have been written.
Run the same command with the same directory to resume. Completed seeds are
reused, and an interrupted seed restarts from its original RNG stream in a new
attempt directory. The previous attempt's log stays in its hidden
`.seed-*` directory.

```bash
cat cpp/build/nrpt-overnight/{natprot-gox,edp-gox-mgox-tpi-12gluc}/summary.md
```

Each model's `summary.json` has the machine-readable diagnostics in
`sequential_run`. The equivalent `summary.md` reports generated and retained
cold draws, per-seed wall times, round trips per seed and per physical replica,
and round-trip rates. The sampling wall covers warm-up and retained scans
across the sequential replicates. Total wall adds replicate setup, output,
and diagnostics. `latest-invocation-wall-seconds.txt` at the run root measures
the launcher invocation, including setup and orchestration.

ESS and R̂ use the independent retained cold chains. ESS/s divides by sampling
wall, and the minimum, mean and sum use defined marginal ArviZ bulk ESS values
across varying fluxes. The sum is descriptive because fluxes can be correlated.
R̂ is the maximum rank-normalized and folded split R̂ across varying fluxes.
The summarizer requires at least two distinct seeds. The numbered,
model-specific SVGs overlay the cold-chain histograms by seed with shared bins
in original model units. Fixed fluxes appear as vertical lines at their model
values.

#### NatProt GOX on the Ryzen 9 5950X

Each strategy ran separately with four seeds (42, 136, 271, and 314), starting
at weights 0, 0.33, 0.67, and 0.995 between the interior and FluxML points.
Each cold chain kept 120,000 draws after 2,000 warm-up steps or scans. NRPT used
the ladder `[0, 0.005, 0.02, 0.06, 0.16, 0.36, 0.65, 1]` and CSmMALA
used step size 0.15. Every mixed slot above the listed cutoff took one
CSmMALA step per scan.

| Strategy | Sampling wall (s) | Total trips | Max. R̂ | Min. / mean bulk ESS | Min. / mean ESS rate (s⁻¹) |
|---|---:|---:|---:|---:|---:|
| Standalone rounded Hit-and-Run | 42.0 | — | 1.7908 | 5.9 / 3,331.7 | 0.141 / 79.243 |
| Rounded Hit-and-Run NRPT, 1 step per slot | 119.5 | 77 | 1.0503 | 53.6 / 7,382.8 | 0.448 / 61.778 |
| Mixed, 1 Hit-and-Run step at β ≤ 0.5 | 2,297.4 | 85 | 1.0155 | 271.6 / 2,098.6 | 0.118 / 0.913 |
| Mixed, 10 Hit-and-Run steps at β ≤ 0.65 | 2,332.6 | 714 | 1.0017 | 951.9 / 19,633.2 | 0.408 / 8.417 |
| Mixed, 10 Hit-and-Run steps at β ≤ 0.5 | 2,767.0 | 726 | 1.0016 | 1,049.9 / 6,851.6 | 0.379 / 2.476 |

The standalone replicates used one core each. For NRPT, four concurrent
replicates with four workers each beat two waves of eight-worker replicates
in every short pilot. Each replicate process was pinned to a disjoint group
of physical cores, with numerical libraries capped at one thread. Sampling
wall time spans the first launch through completion of all four replicates,
including setup, warm-up and writing draws. ESS rates use this wall time.
The full session took 8,118.5 s including pilots and diagnostics. Both
ten-step schedules met the ESS targets in every replicate for all 27 free
fluxes.

## Kernel and geometry benchmarks

The [native build](#native-c) also produces
`cpp/build/native-benchmarks/hopsy_hit_and_run_benchmark`. It reports CSV
timings for the components of coordinate and dense Hit-and-Run.

With the [Python bindings built](../README.md#python-bindings), run a short
linear and bilinear throughput check from the repository root.

```sh
PYTHONPATH="$PWD/cpp/build/python/bindings" cpp/.benchmark-venv/bin/python \
  cpp/benchmarks/benchmark_bilinear.py --steps 20 --repeats 1 \
  --output cpp/build/bilinear-smoke.json
```
