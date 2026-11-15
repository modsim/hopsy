# Persistent direct C++ bindings

This iteration replaces the Python-owned compatibility chains in commit
`31870a9` with direct bindings to C++ problems, kernels, RNGs, and an owning
accepted-state frontend. The public API is documented in
[bindings/README.md](../bindings/README.md).

## Design and ownership

`OwnedStateChain<P,K>` lives in `include/hopsy/sampling/owned_chain.h`. It owns
accepted-state and workspace storage, borrows K, and holds a persistent typed
`BoundChain`. It optionally shares ownership of an immutable P. Python supplies
an rvalue Eigen vector through the caster and a shared problem; the original
C++ lvalue-state overload still returns the original borrowed view. Owner objects
cannot move because the internal binding references their members. Temporary
borrowed problems are rejected at compile time.

Nanobind registers actual kernel, RNG, and problem template instantiations.
There is no Python state-management or sampling layer. A problem factory selects
one of uniform/native Gaussian/Python callback targets and identity/affine maps
once. Each concrete chain calls its corresponding C++ method directly. Native
calls release the GIL; callbacks retain it and receive copied vectors. Overrides
on native target subclasses use the Python adapter. Kernels are kept alive while
borrowed; the current problem is retained by C++ shared ownership. Rebinding drops
the old shared owner, with tests for actual callback-target destruction.

The core CMake target exports the platform Threads dependency for these public
mutex-based guards, including installed-package consumers.

Both C++ and Python owning frontends exclude duplicate kernel bindings and
concurrent/reentrant chain/RNG use at call boundaries. These checks are not inside
transition loops. Raw C++ views retain their caller-managed exclusivity contract.
The guards do perform ordinary bookkeeping allocation at call boundaries; the
claim is no per-transition dispatch/check/allocation overhead from the frontend.
A native target may still allocate internally (the existing Gaussian target does).

Ordinary batches preserve accepted-state density/drift caches and mapped evaluation
buffers. `rebind_problem` invalidates target-dependent data once and retains the
geometry metric; it requires the same concrete problem type and identical geometry
and map objects. C++ borrowed geometry should use `std::cref` when sharing it among
problems. Cross-type rebinding and tempering families remain separate features.
A fresh owning binding clears state/geometry Gaussian caches, covered by a test
that reuses a kernel with a different box and compares to a fresh kernel.

The core `make_sampling_resource<T>` helper provides isolated, 64-byte-aligned
storage for independently mutated kernels and RNGs. A standard aliasing shared
pointer preserves the actual T type and correctly destroys the enclosing storage.
The Python constructors use this allocation helper. Kernel layout, mathematical
operations, and Hit-and-Run source remain unchanged. An initial direct-inline
allocation experiment showed parallel slowdowns; allocation isolation was measured
before retaining it. This padding targets the measured x86 cache-line size and is
not a promise about every machine's performance.

The trade-off for concrete dispatch is more template instantiations and a larger
extension (approximately 1.9 MiB versus 0.9 MiB before, before final stripping).
Registration is split across kernel-family translation units to reduce compiler
memory pressure. C++ view ownership, Python object lifetime retention, and
read-only state snapshots remain explicit rather than hidden behind a virtual
kernel hierarchy.

## Benchmark protocol

`benchmark_bound_python.py` runs the old or new API selected by `--api`. The frozen
baseline extension was copied before source changes to ignored
`cpp/build/bindings-before`. It implements the committed `31870a9` interface.
Both extensions use the same existing Release build environment, Eigen, PCG,
seeds, initial vectors, step-size conventions, and target arithmetic.

Targets are uniform or dense `N(0,I)` on `[-3,3]^10`. GRW sigma is 0.7; Dikin and
CSmMALA sigma is 0.3. Each trial warms up for 1000 transitions, then times Python
calls, output allocation, sampling, and (for four workers) thread-pool startup and
shutdown. Setup and warm-up are outside the timer. Each timed worker includes one
final-state sum; there are no per-transition diagnostics or ESS computations.

The short matrix attempts 20k transitions/worker, using batches of 1, 16, or 20k
and thinning 1. It exposes call overhead but very short parallel tasks are noisy.
The sustained parallel comparison uses four workers, 500k transitions/worker,
5000 stored samples/worker, and thinning 100. Workers are restricted to CPUs 0..3
on the Ryzen 9 5950X (distinct physical cores). The runner alternates executable
order over three rounds and pools all nine durations per case. It requires exact
final-state checksums across every old/new trial. These are throughput checks;
ESS is not re-estimated because the seeded trajectory is preserved.

```sh
# Before editing, freeze the extension built from the baseline source:
mkdir -p cpp/build/bindings-before
cp cpp/build-python/core/bindings/_hopsy_core*.so cpp/build/bindings-before/
# Then rebuild the modified extension and run tests before timing.
cmake --build cpp/build-python -j 2
cpp/.benchmark-venv/bin/python cpp/benchmarks/run_bound_python.py \
  --parallel --output cpp/benchmarks/results/bound_python_parallel_batch
cpp/.benchmark-venv/bin/python cpp/benchmarks/run_bound_python.py \
  --output cpp/benchmarks/results/bound_python_calls_batch
```

The frozen extension is a local artifact, not committed. Reproduction elsewhere
requires building `31870a9` separately. Final runner JSON includes module hashes,
platform, CPU affinity, parameters, and summary rates. Individual JSON/log files
retain all durations/checksums. `bound_python_summary.json` and the first
`bound_python_parallel_summary.json` are preliminary measurements of unisolated
allocation; they are retained to show why allocation was investigated, and must
not be confused with the final results. The `*_final` checkpoint measures aligned
allocation before the batch-loop simplification. `*_batch` is the final source:
the loop resolves target/geometry/workspace references once and places exception
cancellation around a batch, retaining the compile-time always-accepted path.
A regression checks exceptions after completed transitions within a batch.

## Validation commands

```sh
cmake --build cpp/build -j 2
cpp/build/tests/hopsy_core_tests
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  -m pytest -q cpp/bindings/tests
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  cpp/examples/gaussian.py
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  cpp/bindings/example.py
```

The updated legacy/new benchmark drivers use the persistent API for their new
backend. Historical legacy measurements are retained; this iteration does not
modify or rebuild the legacy implementation.

## Final measurements

The final `*_batch.json` results pool nine trials per case. Throughput below is
aggregate attempted millions of transitions/second for four chains, with 500k
transitions/chain and thinning 100; setup/warm-up are excluded.

| Target | Kernel | Frozen bindings | Direct bindings | Change |
|---|---|---:|---:|---:|
| gaussian | UniformCoordinateHitAndRunKernel | 27.062 | 30.641 | +13.2% |
| gaussian | UniformHitAndRunKernel | 10.337 | 10.558 | +2.1% |
| gaussian | GaussianRandomWalkKernel | 14.762 | 15.720 | +6.5% |
| gaussian | DikinWalkKernel | 2.793 | 2.883 | +3.2% |
| gaussian | CSmMALAKernel | 2.295 | 2.346 | +2.2% |
| uniform | UniformCoordinateHitAndRunKernel | 137.353 | 136.490 | -0.6% |
| uniform | UniformHitAndRunKernel | 14.079 | 14.736 | +4.7% |

Uniform coordinate HR is within 1% of the baseline; other sustained cases improve
about 2–13%. These measurements detect no significant regression in the tested
workloads and do not prove zero cost on other machines or arbitrary target types.
CSmMALA repeated single-step Python calls improve about 9.2%; 16-step batches
improve about 3.5%, and 20k-output single-chain batches improve about 3.0%.
Exact final-state checksums match across all compared trials. The Gaussian example
also retains its original seeded means/covariance output.

Observed validation: 116 core cases / 523795 assertions; 148 Python tests,
including actual PolyRound equality reduction, direct construction of every kernel
family, callback ownership and exceptions, partial-batch interruption, external
RNG exclusion, persistent mapped/unmapped cache counts, parallel reproducibility,
and old-target release on rebinding. Both Python examples run. The updated
three-family Python benchmark driver also passed a 2000-sample smoke run for
serial/four-thread cases, with its output kept outside the results directory so
those short smoke timings are not presented as performance evidence.
