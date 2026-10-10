# Hit-and-run implementation

Completed against the supplied workspace on 2026-10-10.

## API and architecture

- Retained `HitAndRunKernel<State, DirectionGenerator, LineKernel>` and the existing `propose`, `log_acceptance_ratio`, `accept`, and `reject` sampling interface.
- Construction takes `(initial_state, direction_generator, line_kernel)`. It preallocates an owning candidate state. Class template argument deduction is supported.
- Kept and completed the existing nested `Proposal`, with `state`, `step`, and `log_proposal_ratio`. A separate generic `Proposal<State, Extra>` abstraction was unnecessary here.
- Added `step_direction(direction, state, distance, output)` overloads for Eigen vectors and `CoordinateDirection`. This preserves existing Eigen direction APIs without a wrapper. Input and output must not alias.
- Added a const `direction()` accessor to each generator. The generator retains the direction until its next `generate()` call; the kernel therefore needs no second dense direction buffer or direction copy. Custom generators must implement the same contract and use a state-independent direction law.
- Proposal generation ignores the target, obtains the geometry's line domain, samples a displacement, caches the line correction, and materializes the candidate. Accepted state and cached slacks remain unchanged; geometric scratch such as `direction_product` can change.
- Every line kernel implements `log_proposal_ratio(domain, step)`. Uniform and overrelaxed kernels return zero. Gaussian uses `log Z(x) - log Z(y)`, where the reverse chord is translated by minus the sampled displacement, and now returns `double` for both float and double domains.
- `log_acceptance_ratio` returns the target log-density difference plus the cached correction in double precision. No line-kernel symmetry branch is used to obtain the correction.
- The existing `always_accept<Target>` filter hook is supported for uniform targets and the two symmetric line kernels. Other cases use the existing MH decision path, now explicitly in double precision. Nonnegative ratios accept without an RNG draw; NaN and negative infinity reject without a draw.
- Acceptance advances the workspace and swaps owning state buffers; rejection leaves the accepted state and slacks unchanged. No adaptation, statistics framework, or rounding-specific branch was introduced.

A typical setup is:

```cpp
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/target/uniform.h>

hopsy::Matrix<double> A(4, 2);
A << 1, 0, -1, 0, 0, 1, 0, -1;
hopsy::LinearPolytope<double> geometry(A, hopsy::Vector<double>::Ones(4));
hopsy::Vector<double> state = hopsy::Vector<double>::Zero(2);
auto workspace = geometry.create_workspace(state);
hopsy::HitAndRunKernel kernel(
    state, hopsy::RandomDirectionGenerator<double>(2), hopsy::UniformLineKernel{});
hopsy::RandomNumberGenerator rng(42);
hopsy::UniformTarget target;
auto samples = hopsy::draw(kernel, state, rng, target, geometry, workspace, 1000);
```

For coordinate sampling, substitute `CyclingCoordinateDirectionGenerator(2)` or `RandomCoordinateDirectionGenerator(2)`. For a Gaussian displacement use `GaussianLineKernel(sigma)` and include its header.

## Lifecycle contracts

The state must be an owning vector with dimensions matching the generator and geometry. Its initial value must be feasible, and its workspace must describe that state. Between a proposal and its accept/reject decision, do not modify the accepted state or overwrite workspace scratch with another geometry query. Each concurrently sampled chain needs its own kernel and workspace.

`proposal()` returns a const reference to reusable storage. A subsequent proposal overwrites it; after acceptance its state buffer contains the previous accepted state. External Eigen views and data pointers must not be retained across acceptance because buffers are swapped. Copying a kernel copies its owned state and direction, without dangling references.

Cached slacks retain the existing incremental arithmetic. The tests bound accumulated floating-point error over 10,000 mixed decisions; they do not claim exact arithmetic or indefinite absence of drift. `geometry.reset_workspace(state, workspace)` remains available to callers.

## Build repairs

The supplied baseline had a missing template declaration in `line_kernel_requires_mh`, missing symmetric line corrections (already called by an existing test), and an include-order dependency in the overrelaxed header. These are repaired.

The archive also lacked `cmake/hopsyConfig.cmake.in`, and the parent installation rules could not see `pcg_cpp_SOURCE_DIR`. Added the package-config template and exported the needed variables from the dependency subdirectory. Installed consumers still need Eigen and Boost available through CMake packages; these are not bundled into the output archive.

## Verification

Environment: GCC 13.3.0, CMake 4.4.4, Linux x86-64, AMD EPYC 9V74. Dependencies: Eigen 5.0.1, Boost 1.89.0, Catch2 3.8.1, and the workspace's pinned PCG commit.

- Repaired baseline: all 43 existing tests pass.
- Final optimized build: all 64 CTest cases pass, approximately one second total here.
- Tests cover both materialization paths, all three direction generators, float and double states, candidate reuse, accepted-state isolation, accept/reject workspace consistency, cached and empty workspaces, and independent copying of pending kernels.
- Analytic Gaussian normalization tests check correction magnitude and reverse sign. Fixed-seed chains check uniform and truncated exponential moments with all three line kernels and both dense and cycling coordinate directions.
- RNG tests verify deterministic decisions consume no acceptance draw, including NaN, and uniform hit-and-run consumes only the proposal RNG sequence.
- Eigen's runtime allocation guard passes after initialization for 2,000 MH steps per combination of the three line kernels and two direction families. Assertions are deliberately enabled in optimized test builds. This checks Eigen heap allocations, not every possible allocation inside arbitrary user components.
- All 20 public headers compile independently.
- Default installation configuration and `cmake --install` succeed. A downstream `find_package` consumer was not tested against separately installed Eigen/Boost packages.

Reproduce:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOPSY_BUILD_CPP_BENCHMARKS=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
./build/hopsy_hit_and_run_benchmark
cmake --install build --prefix /your/install/prefix
```

## Benchmark interpretation

`benchmarks/results.csv` records nanoseconds per operation, using medians of five batches at dimensions 8, 64, and 256. The benchmark is optional and off by default. Its full-chain measurements use cached geometry, a uniform target, and uniform line proposals.

The fused baseline uses the same components but commits directly in place; it does not support candidate materialization followed by rejection. It is not old HOPS. No old-HOPS performance claim is made, because no runnable old-HOPS comparison was set up for this task.

Direction generation and geometry dominate dense hit-and-run at larger dimensions. Coordinate proposal materialization adds a state copy, as required by the proposed interface, while acceptance swaps buffers. Separate timings for direction generation, geometry, line sampling, materialization, a quadratic target-density ratio, and a finite-negative MH decision help distinguish these costs. The target and MH decision columns are illustrative costs and are not part of the uniform full-chain timing. Isolated columns should not be summed to predict total time: compiler optimization and cache behavior differ. Measurements were made on a shared host without CPU pinning and are indicative rather than stable performance guarantees.

| Direction | Dimension | Full step (ns) | Fused baseline (ns) | Materialization alone (ns) |
|---|---:|---:|---:|---:|
| coordinate | 8 | 32.0 | 28.5 | 3.2 |
| general | 8 | 272.9 | 258.5 | 3.5 |
| coordinate | 64 | 179.4 | 156.1 | 10.9 |
| general | 64 | 2839.6 | 2888.8 | 21.1 |
| coordinate | 256 | 645.8 | 550.4 | 36.8 |
| general | 256 | 33719.0 | 40106.8 | 76.4 |
