# hopsy C++ core

hopsy 2.0 is a header-only C++20 library for constrained Markov chain Monte Carlo. The optional nanobind module exposes the same samplers and a native target capsule interface to Python.

## How non-reversible parallel tempering works

For a bounded, positive-volume feasible region (P), slot β targets a density proportional to `1_P(x) exp(β ℓ(x))`, where ℓ is the entire supplied target log density. The hottest slot at β = 0 is uniform on (P), even where the supplied target gives zero mass. The cold slot at β = 1 targets the requested posterior. Intermediate slots use strictly increasing β. A constant-volume affine coordinate map preserves these semantics.

Each scan explores every slot, saves one cold draw, then offers disjoint adjacent swaps. With zero-based slots the first scan offers edges 1, 3, … and the second offers edges 0, 2, …. The log acceptance ratio for slots (i, j) is `(β[j] - β[i]) (ℓ(x[i]) - ℓ(x[j]))`, capped at zero. Each local transition and each matching of Metropolis swaps preserves the product of the slot target distributions. The deterministic composition is generally non-reversible.

Positive β always scales the log density, even when β is close to zero. The supplied target may return negative infinity to exclude a point from slots with β > 0. Initial states at positive β must have finite log densities, and target evaluations of NaN or positive infinity are errors.

Slots hold temperatures and point to physical replicas. A replica retains its state, RNG stream, kernel workspaces, native target context and untempered density cache when it changes slots. Temperature-dependent caches are invalidated on a β change. A round trip is a completed hot → cold → hot journey by one physical replica within the current diagnostics window. Warm-up diagnostics can inform a new ladder through `TemperatureLadder::adapted`. Fix the ladder for retained sampling.

Hit-and-Run combines random dense, random coordinate or cycling coordinate directions with uniform, Gaussian or overrelaxed line proposals. The Gaussian kernel family includes Gaussian random walk, MALA, Dikin walk and CSmMALA. The mixed FluxML sampler uses rounded random-coordinate Hit-and-Run at or below a chosen β and CSmMALA in the model coordinates above it. Columns of the rounding matrix give equivalent chord directions in model coordinates, so swaps need no coordinate conversion. Switching walks rebuilds the incoming geometry workspace from the current state. Positive-beta Hit-and-Run still uses tempered Metropolis acceptance.

| Source | Responsibility |
|---|---|
| [non_reversible_parallel_tempering.h](include/hopsy/sampling/non_reversible_parallel_tempering.h) | Replica ownership, local exploration, swaps and trip counters |
| [temperature_ladder.h](include/hopsy/sampling/temperature_ladder.h) | Ladder validation and warm-up adaptation |
| [basis_reference.h](include/hopsy/sampling/basis_reference.h), [log_density_cache.h](include/hopsy/target/log_density_cache.h) | Rounded chord directions and cached accepted and proposed densities |
| [tempering.cpp](bindings/tempering.cpp), [tempering.h](bindings/tempering.h) | Python API, native worker assignment and the per-scan exchange barrier |
| [benchmark_ecoli_nrpt.py](benchmarks/benchmark_ecoli_nrpt.py), [benchmark_flux_convergence.py](benchmarks/benchmark_flux_convergence.py), [flux_benchmark_common.py](benchmarks/flux_benchmark_common.py) | Single-seed comparisons, replicated convergence diagnostics, and shared model and trace utilities |

See the [complete C++ example](examples/non_reversible_parallel_tempering.cpp) and the [Python, C++ and capsule API guide](bindings/README.md).

Each chain needs its own mutable kernel, workspace and RNG stream. Geometry and targets can be shared when their const operations support concurrent use. Borrowed C++ objects must outlive the chain. A pending proposal reserves its workspace until acceptance or rejection. The bound-chain API manages this lifecycle and exposes the last evaluated accepted and proposed log densities.

The bilinear geometry supports uniform and Gaussian-displacement Hit-and-Run over unions of intervals. The Gaussian family and overrelaxed line kernel require linear geometry. Bilinear slices use numerical filters with an exact fallback for ambiguous boundaries. Use binary round-to-nearest arithmetic without fast-math.

## Build, install and test

Use CMake 3.26 or newer, Git, a C++20 compiler and standard library with thread support, and a CMake build tool such as Make or Ninja. The build looks for Boost 1.89 and Eigen 5.0 or newer and fetches them if needed. It fetches PCG and uses Catch2 3.8.1 or newer for native tests. A first build without these dependencies needs network access.

From the repository root, build and install the C++ headers and CMake package into a local prefix.

```sh
cmake -S cpp -B cpp/build/native -DCMAKE_BUILD_TYPE=Release \
  -DHOPSY_BUILD_CPP_TESTS=ON -DHOPSY_BUILD_CPP_BENCHMARKS=ON
cmake --build cpp/build/native --parallel 4
ctest --test-dir cpp/build/native --output-on-failure
cmake --install cpp/build/native --prefix "$PWD/cpp/build/native-install"
```

A downstream CMake project uses `find_package(hopsy CONFIG REQUIRED)` and links `hopsy::core`. Pass the installation prefix in `CMAKE_PREFIX_PATH` along with any nonstandard Boost and Eigen installation prefixes. The installed package needs those dependencies' CMake packages. Run just the Dirichlet stationary-distribution checks with `ctest --test-dir cpp/build/native -L '^dirichlet$' --output-on-failure`.

### Python bindings

Use Python 3.10 or newer with its development headers. Choose an interpreter compatible with your 13CFLUX wheel and use the same interpreter for CMake, pip and the benchmarks. The handover was tested with Python 3.14 and nanobind 3.1.0.

```sh
python3 -m venv cpp/.benchmark-venv
hopsy_python="$PWD/cpp/.benchmark-venv/bin/python"
"$hopsy_python" -m pip install nanobind pytest numpy scipy matplotlib arviz
cmake -S cpp -B cpp/build/python -DCMAKE_BUILD_TYPE=Release \
  -DHOPSY_BUILD_CPP_TESTS=OFF -DHOPSY_INSTALL_CPP=OFF \
  -DHOPSY_BUILD_PYTHON=ON -DHOPSY_BUILD_PYTHON_TESTS=ON \
  -DPython_EXECUTABLE="$hopsy_python" \
  -Dnanobind_DIR="$("$hopsy_python" -m nanobind --cmake_dir)"
cmake --build cpp/build/python --parallel 4
ctest --test-dir cpp/build/python --output-on-failure
cmake --install cpp/build/python --prefix "$PWD/cpp/build/python-install"
PYTHONPATH="$PWD/cpp/build/python-install" "$hopsy_python" -c 'import hopsy; print(hopsy.TemperatureLadder)'
```

For FluxML benchmarks, put `cpp/build/python/bindings` on `PYTHONPATH`. The scripts import the new extension as `_hopsy_core` and use the separate legacy `hopsy` package for rounding. Keep the `python-install` prefix out of the benchmark's `PYTHONPATH` so it does not shadow legacy hopsy.

### 13CFLUX native target wheel

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

## FluxML benchmarks

Put the two model files in `cpp/benchmark-python/` or pass their paths with `--fml`. The models are supplied separately from the source tree. Both scripts accept `--configuration`, `--betas`, `--num-samples`, `--warmup` and `--seed`. A sample is one retained cold draw. β lists include 0 and 1 and determine the number of temperature slots. An independent replicate is a separate invocation with its own seed, not a temperature slot.

Both scripts require at least 1,000 retained draws for batch-means ESS. The single-seed script defaults to 100,000 draws and 1,000 warm-up iterations. The replicated runner defaults to 20,000 draws and 2,000 warm-up iterations. Warm-up uses the same local step counts as sampling and its draws are discarded.

**Single-seed comparison.** `benchmark_ecoli_nrpt.py` compares a standalone chain to serial or parallel NRPT via `--nrpt-mode {parallel,serial,both}`. Select `--kernel {hit-and-run,gaussian-hit-and-run,csmmala,both}`, where `both` runs dense uniform Hit-and-Run and CSmMALA. `--step-size` sets the Gaussian line scale or CSmMALA proposal scale. `--round` rounds the whole problem. For CSmMALA in model coordinates use `--reference-kernel {same,rounded-coordinate,both}` with `--reference-max-beta` to select the hottest slots for rounded Hit-and-Run. `--reference-steps` sets the step count at β = 0, `--local-steps` sets it at every positive β, and `--standalone-thinning` controls standalone steps per draw. With no explicit β list, `--num-replicas` sets the length of a linear ladder. JSON goes to stdout. Use `--replica-trace` for an NPZ and `--posterior-plots` for SVGs.

**Replicated comparison.** `benchmark_flux_convergence.py run` uses rounded uniform coordinate Hit-and-Run for the standalone and all-Hit-and-Run controls. Use `--method {standalone,nrpt_parallel,nrpt_parallel_rounded_reference}` to select one strategy. Omitting it runs all three. `--mixed-only` is shorthand for selecting the mixed method. In mixed runs, `--reference-max-beta` selects the inclusive Hit-and-Run cutoff and `--hit-and-run-steps` gives the number of steps at each of those slots. Every higher slot takes one CSmMALA step at `--step-size` (default 0.15). `--fit-weight` sets the start between the interior point (0) and the FluxML initial point (near 1).

`--workers` controls native workers per NRPT replicate. A worker explores slots at strides of the worker count before joining the exchange barrier. Each physical replica has a separate native capsule context, so local exploration can run without the GIL. When running processes concurrently, budget the sum of their workers and cap numerical library threads.

### Short NatProt smoke test

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
      --fml 'cpp/benchmark-python/Ecoli_NatProt_GOX.fml' --configuration default \
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

### Harder model smoke test

The supplied harder file's schema hint points to a URL that the tested FluxML parser could not load. Use a local copy without that hint. The `[12]Gluc` configuration also needs a bounded reference. The following caps reproduce the earlier benchmark's **capped** model and change its posterior support. The report lists the added inequalities.

```bash
hard_out="$(mktemp -d cpp/build/gluc-smoke.XXXXXX)"
hard_name='e_coli_+EDP+GOX+MGOX-TPI.fml'
"$hopsy_python" - "cpp/benchmark-python/$hard_name" "$hard_out/$hard_name" <<'PY'
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

### Reading the output

- `run` writes `report.json`, one cold sample matrix per method as NPY (original free parameters × retained draws), and NRPT trace NPZs (scan × slot). Traces start with the post-warm-up ensemble and record replica IDs and untempered densities after swaps. The report gives exchange counts, round trips per physical replica and native draw time. `sampling_seconds` includes warm-up plus retained draws. `elapsed_seconds` also includes setup and most output work.
- `summarize` requires at least two matching, distinctly seeded replicates. It writes `summary.json` with ArviZ rank-normalized and folded split R̂ and bulk ESS, keeping the independent cold chains distinct for diagnostics. It checks ESS thresholds of 100 per chain and 50 per half-chain for every free flux. Recorded ESS rates use summed retained-draw timer values. Measure elapsed wall time around a concurrent batch when comparing throughput.
- The SVGs overlay normalized histograms by method for varying net and exchange fluxes reconstructed in original model coordinates. Fixed flux values are listed in the report. Round trips measure movement along the ladder. Check R̂ and ESS alongside them.

See the [benchmark notes](benchmarks/README.md) for the native benchmark programs and measured NatProt results.
