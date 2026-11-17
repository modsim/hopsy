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

Native benchmarks exclude output storage; they measure attempted transitions,
including rejections. Statistical correctness belongs to the retained tests;
throughput alone does not measure mixing or effective sample size.

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
