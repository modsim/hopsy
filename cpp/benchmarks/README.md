# Representative benchmarks

Build with `-DCMAKE_BUILD_TYPE=Release -DHOPSY_BUILD_CPP_BENCHMARKS=ON`.
Run benchmarks sequentially, without competing builds, using the same compiler,
flags and machine for before/after comparisons.

- `build/hopsy_sampling_benchmark`: coordinate/dense Hit-and-Run at dimensions
  10 and 100, target-dependent coordinate MH, Gaussian random walk, Dikin and
  CSmMALA. Five seeded batches of 100,000 transitions after warm-up; reports
  median ns/transition, throughput and final-state checksums.
- `build/hopsy_bound_sampling_benchmark`: the same workload through `BoundChain`.
- `build/hopsy_hit_and_run_benchmark`: direction, chord, materialization and MH
  component costs at dimensions 8, 64 and 256. Includes a fused in-place baseline,
  which has different rejection capabilities. Component timings are not additive.
- `build/hopsy_parallel_tempering_benchmark` measures communication with cached
  densities and complete serial scans for 8 and 32 replicas. See the NRPT
  benchmark details below.
- `build/hopsy_parallel_tempering_comparison` compares ESS per wall-clock
  second for eight independent cold chains and eight-replica NRPT.

The sampling and Hit-and-Run timing benchmarks exclude output storage and
count attempted transitions, including rejections.

## NRPT benchmark

```sh
cpp/build/hopsy_parallel_tempering_benchmark
```

The cold target is a standard Gaussian truncated to an 8-dimensional box.
Each case uses a linear temperature ladder and takes one local step per slot
per scan. The benchmark covers uniform, Gaussian-displacement, and overrelaxed
Hit-and-Run with random-coordinate, cycling-coordinate, and dense directions.
It also covers Gaussian random walk, identity-metric MALA, Dikin walk, and
CSmMALA. Step sizes are fixed at 1.0 for Gaussian-displacement Hit-and-Run, 0.7
for the identity-metric Gaussian kernels, and 0.3 for Dikin walk and CSmMALA.

Each case retains 30,000 cold samples in each of five independently seeded runs.
The median scan time includes writing samples and updating exchange diagnostics.
Warm-up and ESS calculation are outside the timed interval. The report gives
per-edge exchange accepts and attempts, mean conditional rejection by edge,
their sum (the global communication barrier), and per-replica and total
hot–cold–hot round trips. Edges run from hot to cold. Physical replica IDs are
in initial slot order.

ESS estimates the efficiency of each cold-coordinate mean. The benchmark
uses 240-draw batch means within each run to estimate the long-run variance.
It divides the average marginal variance by the average long-run variance,
then multiplies by the total retained sample count. Minimum ESS per second
uses the sum of the five measured draw times. The communication-only timing
uses fixed configurations and reports timing and round trips.
See [Flegal and Jones on batch means](https://arxiv.org/abs/0811.1729).

### Eight-chain ESS per second

```sh
cpp/build/hopsy_parallel_tempering_comparison > comparison.csv
```

The comparison uses three seeded repeats and the same number of local updates
per chain or replica within each workload. Independent chains run concurrently
and keep every cold draw. Parallel NRPT runs eight persistent slot workers,
with a synchronization point for each exchange and one retained cold draw
per exchange. Serial NRPT uses the same schedule on one worker and checks that
it produces the same samples and communication statistics as parallel NRPT.
The NRPT cases exchange after 1, 2, 4, 8, 16, or 32 local steps per slot.

- The original standard Gaussian on `[-4, 4]^8` uses uniform
  cycling-coordinate Hit-and-Run, 8,192 warm-up steps and 131,072 measured
  local steps per chain or replica.
- The correlated Gaussian on `[-1, 1]^8` has unit marginal variances and
  pairwise correlation 0.975 before truncation. It uses CSmMALA with
  `step_size = 0.25` and uniform Hit-and-Run with random dense directions.
  Each case uses 32,768 warm-up steps and 262,144 measured local steps per
  chain or replica.

The report includes actual retained cold draws, minimum coordinate ESS and
ESS for the mean of the eight coordinates. This mean measures movement along
the common direction of the correlated target. The batch-means windows span
512 local steps for the original target and 8,192 for the correlated target.
Batches stay within each independent trace and account for the retained draw
spacing. ESS per second divides pooled ESS by the sum of the measured wall
times across repeats. Timings include writing samples and NRPT synchronization
and exclude startup, warm-up, and the ESS calculation. For a CPU-controlled
comparison, restrict the process to eight distinct physical cores.

## Python benchmarks

For Python, select the built extension with `PYTHONPATH` and use an interpreter
with numpy installed:

```sh
PYTHONPATH=cpp/build-python/core/bindings python cpp/benchmarks/benchmark_bound_python.py --output /tmp/chains.json
PYTHONPATH=cpp/build-python/core/bindings python cpp/benchmarks/benchmark_bilinear.py --output /tmp/bilinear.json
```

These scripts exercise one and four threads with independent kernels and RNGs.
They include output storage and thread-pool setup. The bilinear benchmark includes
linear, empty-bilinear, disconnected and dense-cross constraints. Coordinate-only
sampling of disconnected sign components is excluded because it is reducible.
Use `--worker-local` to compare worker-local construction. CPU affinity is left
to the caller; no machine-specific CPU numbering is assumed.

Historical comparison drivers, frozen-API branches, legacy patches, profiler
experiments and accumulated result files have been removed. Git retains their
history. The optional `python_build` CMake entry point builds current bindings
using an installed nanobind package.
