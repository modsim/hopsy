# Sampling performance comparison

This benchmark compares Release builds from two or more source snapshots. It covers native C++ draws, Python single-chain and independent parallel draws, and serial and parallel NRPT. The inexpensive coordinate Hit-and-Run cases expose overhead that a large target evaluation can hide. Single-sample draws expose call and allocation costs. Thinning 16 increases the number of local transitions per retained sample. The Gaussian CSmMALA and optional FluxML cases cover more expensive targets.

## Build two snapshots

Run from the repository root. The first command archives the pre-refactor commit into a separate directory. The second snapshot copies the current C++ sources, including uncommitted changes.

```bash
baseline_rev=2ff0023573af4c8786691ad8d2d65a7d8e765097
results=$(mktemp -d "$PWD/cpp/build/sampling-performance.XXXXXX")
bench="$PWD/cpp/benchmarks/sampling_performance"
python="$PWD/cpp/.benchmark-venv/bin/python"

mkdir -p "$results/baseline/cpp" "$results/refactored/cpp"
git archive "$baseline_rev" cpp | tar -x -C "$results/baseline"
cp -r cpp/{CMakeLists.txt,bindings,cmake,dependencies,examples,include,tests,benchmarks} \
  "$results/refactored/cpp/"

"$bench/build.sh" "$results/baseline/cpp" "$results/baseline"
"$bench/build.sh" "$results/refactored/cpp" "$results/refactored"
```

The build uses GCC, Ninja, the venv's Python and nanobind, and the existing Eigen and PCG source caches in `cpp/build/_deps`. Set `CXX`, `HOPSY_PERF_PYTHON`, `HOPSY_PERF_DEPS`, or `HOPSY_PERF_BUILD_JOBS` to override those defaults. Use the same values for every snapshot. Each output directory contains the build log and CMake's `compile_commands.json`. Use `git rev-parse` for committed snapshots and retain the source directory or a manifest for uncommitted snapshots.

## Run and summarize

Choose eight distinct physical cores on one shared L3 cache if available. CPUs 8–15 share an L3 cache on the Ryzen 5950X used for this investigation. Keep other computational workloads off those CPUs and run one comparison at a time.

```sh
"$python" "$bench/compare.py" record \
  --variant "baseline=$results/baseline" \
  --variant "refactored=$results/refactored" \
  --cpus 8-15 --reps 10 --seconds 2.5 --output "$results/paired.jsonl"

"$python" "$bench/compare.py" summarize "$results/paired.jsonl" > "$results/summary.csv"
```

Use `--cases` for a subset. To add an optimized build, pass a third `--variant optimized=PATH`. The summary compares each revision with `baseline` by default. Use `summarize --reference refactored` to compare with the pre-fix build. The optional `--cases flux-nrpt` uses the same FluxML model loader as the [FluxML benchmarks](../README.md). It needs the native 13CFLUX wheel and `cpp/benchmarks/Ecoli_NatProt_GOX.fml`. It runs eight dense Hit-and-Run replicas with the ladder `0 .005 .02 .06 .16 .36 .65 1` and records replica and density traces.

Each case calibrates a fixed batch count, then uses the same count in every revision. Each repetition starts a fresh process, constructs fresh chains and RNGs, warms them up, and alternates version order. The code hashes the final batch outside the timed interval and checks equal seeded outputs across versions. BLAS and OpenMP worker counts are set to one. Wall time includes output allocation and conversion, worker launches for NRPT draw calls, and synchronization. Setup and warm-up are outside that interval. The setup metric excludes imports in ordinary cases and includes FluxML model loading in the FluxML case.

The JSONL file is written incrementally and includes cases, seeds, batch counts, wall time, CPU time, setup time, and output hashes. The summary reports the geometric mean throughput across repetitions, the coefficient of variation, and a percentile 95% bootstrap interval using paired log throughput ratios. Throughput counts all local transitions across chains and includes thinned transitions. A cold NRPT sample can represent several local transitions. The original Python API had no ordinary trace method, so those trace cases report only the refactored version. NRPT traces are comparable across both versions.

To profile the native Gaussian NRPT case with the same Release build, run a separate counter measurement after the timing run.

```sh
taskset -c 8 perf stat -r 3 -e cycles:u,instructions:u,branches:u,branch-misses:u \
  "$results/refactored/native" nrpt-gaussian 8 4096 1 270 1150
```
