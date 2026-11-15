# CSmMALA caching and Gaussian proposal lifecycle

This checkpoint removes Gaussian's stored `log_proposal_ratio`, caches accepted
state evaluations, and keeps parameter kernels independent of any outer sampling
algorithm. Hit-and-Run retains its line correction and its specialized state and
workspace updates. No nested-sampling state or kernel is implemented here.

## Cache ownership and validity

`GaussianKernel` owns current/candidate log densities and geometry metric factors.
For `ClippedGradientDrift`, it also owns the current/candidate **coefficient of
sigma squared**, `0.5 G(x)^-1 clip_1(gradient log pi(x))`. The scaled displacement
is constructed using the proposal's scale. Consequently, changing standard
deviation does not invalidate density, gradient/preconditioning or metric caches.
Rejection retains accepted-state data; acceptance promotes candidate data.

The drift policy's compile-time `scales_quadratically` opt-in makes this reuse
explicit. Unknown/nonquadratic policies recompute the current displacement at
each proposal's actual scale. Scale-dependent clipping must not opt in.

Targets must represent a fixed mathematical density while their cached data
remain valid. A change of target type or address invalidates automatically.
In-place mutation or replacement at the same address requires the caller to
invoke `kernel.invalidate_target_cache()` between transitions. Every kernel
using that changed target must be invalidated. This hook clears target density
and drift data; it retains the geometry-only metric factors used here. Changing
state or geometry externally still requires kernel reinitialization.

`draw`/`draw_into` start every nonempty batch with fresh target caches. The hook
is selected with `if constexpr` and imposes no requirement or runtime work on
Hit-and-Run. Python single-step calls likewise invalidate at entry, so callers
can change Python targets between calls. A target must remain fixed during a
batch; diagnostic counters are fine, changing the mathematical density inside
callbacks is not. Per-batch invalidation also avoids depending on the lifetime
or stack address of temporary Python target adapters.

Candidate log density is checked after geometric feasibility and **before**
reverse drift/factor preparation. A target may therefore express additional hard
support using `-inf` without defining its gradient outside that support. This
is a general target capability, not knowledge of a particular outer algorithm.

## Pending proposals

The public `Proposal` contains only candidate state and scale. Symmetry is a
compile-time property, independent of always-acceptance. Asymmetric corrections
are local values calculated during MH evaluation, after support checks and
reverse preparation. There is no pending correction or synthetic NaN sentinel.
Repeated MH evaluations reuse density/drift/factor data, but calculate the scalar
correction from those data again.

The private lifecycle is idle -> proposed -> usable evaluation -> idle:

- `propose` requires idle; `has_pending_proposal()` then becomes true.
- `log_acceptance_ratio` requires a pending proposal and the same target.
- `accept` requires a usable evaluation; zero-mass/NaN candidates cannot commit.
- `reject` cancels either pending phase and is idempotent and noexcept.
- Scale changes and explicit cache invalidation require idle.
- A forward evaluation exception leaves idle; an exception during MH evaluation
  leaves a cancellable proposal. Low-level C++ callers can reject in a catch
  block. Python wrappers rebuild from the last accepted state on exceptions;
  RNG consumption is not rolled back.

State/workspace ownership is unchanged. The removed public C++ correction field
is an intentional source-interface change; tests now verify full MH ratios
against independent density formulas instead of reading that diagnostic field.
The Python API did not expose that field.

Future extended-state targets can provide joint density terms and support via
ordinary target views. Outer algorithms invalidate target-dependent caches when
those views change. Proposal-law corrections remain the responsibility of the
corresponding kernels; no parameter kernel needs level or nested-sampling logic.

## Performance and validation

The profiled row-by-row metric division was replaced by reciprocal-slack
multiplication in contiguous columns. The target density and drift cache changes
remove repeated accepted-state evaluations, without changing clipping at norm 1.

Five single-chain runs were pinned to CPU 0. Each run used 20,000 warm-up steps,
500,000 retained transitions without thinning, standard N(0,I_10) on [-3,3]^10,
CSmMALA sigma=.3, and corrected legacy clipping with fisher_weight=0. Timings
include sample storage, exclude setup/file I/O/ESS, and alternate old/new order.
Both sides use the same cheap native target arithmetic. This is a native-kernel
comparison, not a repeat of the earlier four-chain Python comparison.

| CSmMALA | Median steps/s | Median ESS/sample | Median ESS/s | Acceptance |
|---|---:|---:|---:|---:|
| Corrected legacy | 643,407 | .01300395 | 8,366.83 | .352428 |
| Cached core | 683,417 | .01300395 | 8,895.08 | .352428 |

Median throughput improves by **6.2%** over legacy on this workload. Each of the
five paired new/old throughput ratios exceeds one. One legacy run was a timing
outlier; all measurements are retained in `results/gaussian_family_cached.csv`.
No claim of globally optimal performance or the same gain on other workloads
is made. ESS is the minimum coordinate Geyer initial-positive/monotone estimate.

An instrumented 100,000-transition run after warm-up now makes **99,947 density
and 99,947 gradient calls on both sides**. Previously the new core made 199,894
and 199,947 calls, respectively. Instrumented throughput is not used in the
performance table. Results are in `results/cached_counts.log`.

Reproduce using the existing dependencies:

```sh
cmake -S cpp -B cpp/build -DHOPSY_TEST_LEGACY_REFERENCE=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build cpp/build --target hopsy_core_tests hopsy_legacy_csmala_tests -j2
cpp/build/tests/hopsy_core_tests
cpp/build/tests/hopsy_legacy_csmala_tests
cmake --build cpp/build-python --target _hopsy_core -j2
PYTHONPATH=cpp/build-python/core/bindings \
  cpp/.benchmark-venv/bin/python -m pytest -q cpp/bindings/tests
c++ -std=c++20 -O3 -DNDEBUG \
  -Icpp/include -Icpp/build/_deps/eigen3-src \
  -Icpp/build/_deps/pcg_cpp-src/include -Iextern/hops/src \
  cpp/benchmarks/gaussian_family_benchmark.cpp \
  extern/hops/src/hops/Utility/DefaultDimensionNames.cpp \
  -o cpp/build/gaussian_family_cached
taskset -c 0 python3 cpp/benchmarks/run_gaussian_family.py \
  --executable cpp/build/gaussian_family_cached --samples 500000 --repeats 5 \
  --output cpp/benchmarks/results/gaussian_family_cached
```

Use `-DHOPSY_COUNT_TARGET_CALLS` to build the separate instrumented diagnostic.
The six-kernel runner also records GRW/Dikin comparisons, but this checkpoint's
performance goal and quoted comparison concern CSmMALA.

Observed validation: **101 core cases / 419,286 assertions**, **59 Python binding
tests**, and **8 legacy clipping regression assertions**, all passing. New cases
cover cache reuse on accept/reject and scale updates, target replacement and
in-place invalidation, batch refresh, cancellation/exception recovery, invalid
commit prevention, copied pending buffers, nonquadratic drift fallback, and
uniform-target asymmetric corrections. Existing independent-density, stationary
moment, clipping-continuity, fixed-size/float and allocation checks remain active.
