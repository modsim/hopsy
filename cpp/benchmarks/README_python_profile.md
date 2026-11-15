> Cache/lifecycle follow-up: [README_csmala_cache.md](README_csmala_cache.md)
> records the accepted-state caching implementation and the matched single-chain
> benchmark. Measurements and implementation descriptions below are historical.

# Python multiprocessing/thread comparison and native profiling

This follow-up patches legacy CSmMALA's `norm != 0` normalization to `norm > 1`
clipping. The independent analytic proposal regression is
`cpp/tests/legacy_csmala_test.cpp`. The old seeded trajectory golden was updated;
all four original legacy CSmMALA tests pass. No other legacy sampler semantics
were changed. In particular the comparisons use coldness=1 and fisher_weight=0.

The core modernization commit leaves the dirty `extern/hops` submodule unchanged.
For reproduction from an unpatched legacy checkout, the two-file reference fix is
preserved in `legacy_csmala_clipping.patch`. Check and apply it once before building
the legacy comparison or enabling `HOPSY_TEST_LEGACY_REFERENCE`:

```sh
git -C extern/hops apply --check ../../cpp/benchmarks/legacy_csmala_clipping.patch
git -C extern/hops apply ../../cpp/benchmarks/legacy_csmala_clipping.patch
```

Do not apply it again to an already-patched working tree. The new core and its
normal tests do not depend on this reference patch.

## Python results

The actual legacy `hopsy.sample(..., n_procs=4)` API is compared with four
`ThreadPoolExecutor` tasks calling nanobind `chain.draw(...)`. Both use native,
dense Gaussian models on `[-3,3]^10`. The native Gaussian implementation is not
an identity-covariance shortcut. The legacy Python facade is copied, unmodified,
from `src/hopsy` into `cpp/benchmark-python/hopsy`, beside a locally rebuilt
patched pybind11 extension. The original source-tree extension is untouched.

Every case has four chains, 100,000 retained samples per chain, 20,000 untimed
warm-up transitions per chain, and three repeats. Seeds are 42..45, 52..55, and
62..65. Scales are the same as the native comparison: sigma=.7 for GRW and .3
for Dikin/CSmMALA, converted to the legacy step conventions. Thinning 1 and 20
are reported separately: 400,000 and 8,000,000 attempted transitions per case.

Medians of actual multiprocessing/thread cases:

| Thinning | Sampler | Old steps/s | New steps/s | Wall-time speedup | Old ESS/sample | New ESS/sample | Old ESS/s | New ESS/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GRW | 315,667 | 11,997,425 | 38.01x | .02835 | .02876 | 8,948 | 345,030 |
| 1 | Dikin | 282,811 | 2,341,609 | 8.28x | .00841 | .00964 | 2,362 | 22,446 |
| 1 | CSmMALA | 285,045 | 1,987,493 | 6.97x | .01220 | .01161 | 3,463 | 23,074 |
| 20 | GRW | 4,406,957 | 12,685,900 | 2.88x | .54002 | .53979 | 119,029 | 342,484 |
| 20 | Dikin | 1,887,406 | 2,661,936 | 1.41x | .21585 | .21459 | 20,259 | 28,522 |
| 20 | CSmMALA | 1,615,272 | 1,979,168 | 1.23x | .28083 | .27269 | 22,386 | 27,083 |

ESS is the sum of each chain's minimum-coordinate Geyer initial-positive,
monotone FFT ESS estimate, divided by total retained samples or sampling wall
time. It is not a multichain convergence diagnostic. Statistics, validation and
ESS calculation are outside the sampling timer. Individual moment diagnostics
and all repeat measurements are in `results/python_gaussian.csv` and `.json`.

Timings include legacy process spawn/imports, IPC/pickling, list/array assembly,
per-sample Python loops, acceptance metadata and shutdown. The new API returns
four native sample matrices without concatenating them and does not collect
per-sample acceptance metadata. Its timings include allocation/storage and
thread-pool startup/shutdown. These are end-to-end sampling pipeline comparisons,
not an isolated pybind11-versus-nanobind or process-versus-thread microbenchmark.

Controls run the same four chains serially through each API. New four-thread
speedups over new serial execution are 3.58--4.02x at thinning 1 and 3.99--4.08x
at thinning 20. Legacy multiprocessing is slower than legacy serial for thinning
1 on these jobs (0.38--0.76x), but faster at thinning 20 (1.38--2.89x).
Therefore large unthinned-batch gains include elimination of process overhead
and per-sample Python work; they should not be attributed solely to threading.
No claim of an optimal step size or universal speedup is made.

The existing uniform coordinate Hit-and-Run Python benchmark was also run:
`[-1,1]^100`, 20,000 stored samples/chain, thinning=100, three repeats. Old four
processes: 4,713,429 steps/s, 1.697 s; new four threads: 19,285,633 steps/s,
0.415 s, a 4.09x throughput gain. Mean/second-moment checks pass. Measurements
are in `results/python_hit_and_run.*`. No core Hit-and-Run source was changed.

## Native profiling: what costs more?

Native profiles use the controlled isotropic target arithmetic from
`gaussian_family_benchmark.cpp`, with clipping fixed on both sides. This cheap
fixture intentionally differs from the dense Gaussian model in Python tests,
and avoids attributing target-implementation differences to sampler abstractions.

1. **Repeated target work.** Instrumented counts over 100,000 transitions at
   seed=42, sigma=.3, excluding warm-up:

   | Sampler | Old density calls | New density calls | Old gradient calls | New gradient calls |
   |---|---:|---:|---:|---:|
   | GRW | 98,112 | 196,224 | 0 | 0 |
   | Dikin | 99,924 | 199,848 | 0 | 0 |
   | CSmMALA | 99,947 | 199,894 | 99,947 | 199,947 |

   Legacy retains current density and drifted means. The new core recomputes
   current density, and CSmMALA current gradient/preconditioning, each transition.
   The latter adds two triangular solves even after rejection. Native target
   identity/change is not currently part of the core cache contract.

2. **Sampled CPU work.** `perf record -F 999 -g --call-graph dwarf` on optimized
   binaries, 3,000,000 transitions, shows triangular solve routines consuming
   about 20.6% of new CSmMALA cycles versus 14.7% of legacy cycles. Metric assembly,
   matrix multiplication and LLT are major costs in both. Percentages refer to
   each binary's own sampled cycles and are not an additive attribution of the
   entire runtime difference. New GRW is heavily inlined, so symbol-level self
   attribution is coarse; counts and small probes are more informative there.

3. **Local arithmetic choices.** Five-repeat median stage probes at dimension 10:

   | Stage | ns |
   |---|---:|
   | Row-by-row metric scaling with division | 132.78 |
   | Reciprocal slacks followed by diagonal multiplication | 50.53 |
   | Allocation-free row-wise feasibility | 71.96 |
   | Legacy-style Eigen product feasibility expression | 38.05 |
   | Gradient norm | 2.56 |
   | Stable gradient norm | 7.75 |
   | Complete clipped, preconditioned drift | 184.12 |

   These isolate operations on fixed inputs; they are not end-to-end speedup
   predictions. The allocation-free row traversal is slower here than Eigen's
   column-friendly product despite avoiding a temporary. The metric scaling
   implementation likewise does more divisions and less contiguous work.

4. **Zero-drift specialization probe.** A temporary header under `cpp/build`
   omitted zero-drift buffer operations for the supported symmetric random walk.
   Five paired million-step runs gave 4.326M steps/s for production versus 4.309M
   for the probe, with identical sample hashes. It showed no repeatable benefit
   and was not applied. The production Gaussian correction lifecycle is unchanged.

All six perf reports, target call counts and stage/probe measurements are under
`results/profile/`; raw perf files and binaries remain under ignored `cpp/build`.

There is no evidence here for a virtual-dispatch penalty in the new core:
policies are template instantiations, with compile-time correction selection.
There **is** additional work from cache policy and local Eigen expressions.
The recommendation is to retain composition and review target-dependent cache
ownership/invalidation separately. Cache an unscaled preconditioned gradient
rather than a scaled mean, so step-size updates remain correct. A target-change
contract is required before reusing such a cache across calls. Faster feasibility
or candidate-slack reuse must also preserve accepted-workspace semantics.
No production sampler cache redesign or scalar-correction removal was made.

After fixing legacy clipping, a fresh five-repeat native comparison gives:

| Sampler | Old steps/s | New steps/s | Old ESS/sample | New ESS/sample |
|---|---:|---:|---:|---:|
| GRW | 4,579,925 | 4,136,759 | .02952 | .02952 |
| Dikin | 769,451 | 746,842 | .01061 | .01061 |
| CSmMALA | 630,243 | 567,164 | .01300 | .01300 |

CSmMALA's acceptance and ESS now match to numerical precision; the remaining
roughly 10% native throughput gap is not a gradient-clipping confound.
Exact repeats and ESS/s are in `results/gaussian_family_clipped.*`. Earlier
`gaussian_family_final.*` intentionally preserves the pre-fix historical data.

## Reproduction

Environment: AMD Ryzen 9 5950X, Linux x86_64, GCC 16.2.1, Python 3.14.8 with GIL
enabled; installed nanobind 3.1.0 and pybind11 3.1.0; NumPy 2.5.4; existing Eigen
5.0.1 sources for both builds. Default Release options are used: legacy C++17,
`-O3 -DNDEBUG` with pybind11's LTO; new core C++20, `-O3 -DNDEBUG`, nanobind
`NOMINSIZE`, no `-march=native` in either. No MKL is linked. CPU frequency scaling
remains enabled. This compares local rebuilds, not a previously shipped wheel.
The installed nanobind package is reused through a local CMake driver rather
than fetched; no project dependency version is changed. Exact Python package
versions and installed environment requirements are recorded alongside results.

```sh
python3 -m venv --system-site-packages cpp/.benchmark-venv
cpp/.benchmark-venv/bin/python -m pip install -r cpp/benchmarks/results/benchmark_requirements.txt
cpp/.benchmark-venv/bin/python cpp/benchmarks/prepare_python_benchmark.py
PYTHONPATH=cpp/benchmark-python:cpp/build-python/core/bindings \
  cpp/.benchmark-venv/bin/python cpp/benchmarks/benchmark_python_gaussian.py
PYTHONPATH=cpp/benchmark-python:cpp/build-python/core/bindings \
  cpp/.benchmark-venv/bin/python cpp/bindings/benchmark_old_vs_new.py \
  --samples 20000 --thinning 100 --warmup 1000 --repeats 3 \
  --output cpp/benchmarks/results/python_hit_and_run.csv
```

The environment inherits system NumPy/nanobind/pybind11/pytest; match
`results/python_versions.json` when reproducing on another machine. Legacy
runtime dependencies were installed only inside `cpp/.benchmark-venv`.

Native profiles (repeat for old/new GRW, Dikin, CSmMALA):

```sh
c++ -std=c++20 -O3 -g -fno-omit-frame-pointer -DNDEBUG \
  -Icpp/include -Icpp/build/_deps/eigen3-src \
  -Icpp/build/_deps/pcg_cpp-src/include -Iextern/hops/src \
  cpp/benchmarks/gaussian_family_benchmark.cpp \
  extern/hops/src/hops/Utility/DefaultDimensionNames.cpp \
  -o cpp/build/gaussian_family_profile
perf record -F 999 -g --call-graph dwarf -o cpp/build/new_csmala.perf -- \
  cpp/build/gaussian_family_profile new_csmala 42 3000000 0.3 cpp/build/profile_chain.bin
perf report --stdio --no-children --call-graph none --percent-limit 0.5 \
  -i cpp/build/new_csmala.perf
```

Use `-DHOPSY_COUNT_TARGET_CALLS` for the separate call-count binary; do not use
its instrumented throughput for comparisons. `gaussian_profile_stages.cpp`
builds with the same core include paths and `-O3 -DNDEBUG`.

Validation performed:

```sh
cmake -S cpp -B cpp/build -DHOPSY_TEST_LEGACY_REFERENCE=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build cpp/build --target hopsy_core_tests hopsy_legacy_csmala_tests -j2
cpp/build/tests/hopsy_core_tests
cpp/build/tests/hopsy_legacy_csmala_tests
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python -m pytest -q cpp/bindings/tests
c++ -std=c++20 -O2 -DNDEBUG -Icpp/build/_deps/eigen3-src -Iextern/hops/src \
  extern/hops/tests/MarkovChain/Proposal/CSmMALATestSuite.cpp \
  cpp/build-legacy/extern/hops/libhops.a -lboost_unit_test_framework \
  -o cpp/build/legacy_csmala_suite
cpp/build/legacy_csmala_suite --log_level=message
```

Observed: 91 core cases / 419,221 assertions; 8 legacy analytic assertions;
53 Python binding tests; all four original legacy CSmMALA cases pass. Binding
coverage includes shared-native-target serial/thread reproducibility, scale
updates/reset/reseed, state ownership, writable output, Python gradient targets,
exception recovery, and initial metric/target validation.
