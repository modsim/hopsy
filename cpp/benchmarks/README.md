# Representative benchmarks

Build using the [native commands](../README.md#build-install-and-test).
Run benchmarks sequentially, without competing builds, using the same compiler,
flags and machine for before/after comparisons.

- `cpp/build/native/hopsy_sampling_benchmark`: coordinate/dense Hit-and-Run at dimensions
  10 and 100, target-dependent coordinate MH, Gaussian random walk, Dikin and
  CSmMALA. Five seeded batches of 100,000 transitions after warm-up; reports
  median ns/transition, throughput and final-state checksums.
- `cpp/build/native/hopsy_bound_sampling_benchmark`: the same workload through `BoundChain`.
- `cpp/build/native/hopsy_hit_and_run_benchmark`: direction, chord, materialization and MH
  component costs at dimensions 8, 64 and 256. Includes a fused in-place baseline,
  which has different rejection capabilities. Component timings are not additive.

- `cpp/build/native/hopsy_parallel_tempering_benchmark` measures communication
  and full scans on a truncated Gaussian. It reports ESS, exchange statistics
  and round trips for the native kernel families.
- `cpp/build/native/hopsy_parallel_tempering_comparison` compares eight
  independent cold chains with eight-slot NRPT on two truncated Gaussian
  targets. It emits CSV, including elapsed sampling time and ESS/s.

The transition microbenchmarks count attempted transitions, including
rejections. The NRPT timings include recording samples.

For Python, select the built extension with `PYTHONPATH` and use an interpreter
with numpy installed:

```sh
PYTHONPATH=cpp/build/python/bindings python cpp/benchmarks/benchmark_bound_python.py --output cpp/build/chains.json
PYTHONPATH=cpp/build/python/bindings python cpp/benchmarks/benchmark_bilinear.py --output cpp/build/bilinear.json
```

These scripts exercise one and four threads with independent kernels and RNGs.
They include output storage and thread-pool setup. The bilinear benchmark includes
linear, empty-bilinear, disconnected and dense-cross constraints. Coordinate-only
sampling of disconnected sign components is excluded because it is reducible.
Use `--worker-local` to compare worker-local construction. CPU affinity is left
to the caller; no machine-specific CPU numbering is assumed.

## FluxML benchmarks

The [C++ handover guide](../README.md#fluxml-benchmarks) covers setup, options,
model selection, smoke tests and result files. The single-seed script reports
approximate minimum ESS from 500-draw batch means. Use the replicated runner for
cross-chain R̂ and pooled bulk ESS.

### NatProt GOX on the Ryzen 9 5950X

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
