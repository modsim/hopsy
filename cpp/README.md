# hopsy C++ core

A header-only C++20 library for constrained Markov chain Monte Carlo sampling.
Link the CMake target `hopsy::core`; installed consumers use `find_package(hopsy)`.
Dependencies are Eigen, Boost headers and PCG; tests additionally use Catch2.

## Sampling

```cpp
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/chain.h>
#include <hopsy/target/uniform.h>

hopsy::Matrix<double> A(4, 2);
A << 1, 0, -1, 0, 0, 1, 0, -1;
hopsy::LinearPolytope<double> geometry(A, hopsy::Vector<double>::Ones(4));
hopsy::Vector<double> state = hopsy::Vector<double>::Zero(2);
hopsy::HitAndRunKernel kernel(
    state, hopsy::RandomCoordinateDirectionGenerator(2), hopsy::UniformLineKernel{});
auto problem = hopsy::Problem(std::cref(geometry), hopsy::UniformTarget{});
auto chain = hopsy::bind_chain(problem, kernel, state);
hopsy::RandomNumberGenerator rng(42);
auto samples = chain.draw(rng, 1000); // Each column is one sample.
```

`Problem` combines immutable geometry, target and optional coordinates. It owns
values unless explicitly given `std::cref`. Geometry is already in sampling
coordinates: affine maps transform target evaluations and output, not constraints.

`bind_chain` with an lvalue state borrows the problem, kernel and state. With an
rvalue state it returns an owning chain, which owns state and workspace, guards
concurrent access and exclusively leases its kernel. Borrowed objects must outlive
the chain. Shared problem ownership is also available. See the
[binding guide](bindings/README.md) for owning C++ and Python examples.

## Non-reversible parallel tempering

`<hopsy/sampling/non_reversible_parallel_tempering.h>` provides
`NonReversibleParallelTempering` for a single constrained problem. See the
[complete C++ example](examples/non_reversible_parallel_tempering.cpp), which
can be built as an executable linked to `hopsy::core`.

Pass a `TemperatureLadder`, a `TemperingReplica<Kernel>` for each slot, and an
exchange RNG. A ladder contains strictly increasing inverse temperatures from
exactly 0 to exactly 1. Use `TemperatureLadder::linear(n)` or supply the values.
Each replica starts with a feasible state and a fresh kernel initialized at
that state. Seed each PCG engine explicitly and give it a unique stream
selector, including the exchange engine.

- `scan(local_steps = 1, reference_steps = 10)` explores every slot and then
  communicates. `draw(count, local_steps, reference_steps)` returns one cold
  sample per scan as a column in target coordinates, taken after exploration
  and before exchange, as in `annealing.py`.
- To use an existing executor, call `explore(slot, steps)` for distinct slots,
  join the workers, and call `communicate()` once. Read states or record samples
  on the coordinating thread. The shared problem, including the target and
  coordinate map, must support concurrent const calls. Each explorer returns
  its accepted local step count. Calling `communicate()` alone uses the current
  states, corresponding to identity local exploration.
- `sampling_state(slot)` and `target_state(slot)` give the current
  configuration in sampling and target coordinates, respectively.
  `replica_indices()` maps slots to physical replica IDs. `round_trips()` counts
  completed hot→cold→hot journeys per ID. `exchange_statistics()` reports
  attempts and accepts per edge. `mean_exchange_rejections()` reports the mean
  conditional rejection probability per edge, including scans that did not
  attempt the edge. Their sum estimates the global communication barrier.
- During warm-up, call `set_ladder(nrpt.ladder().adapted(nrpt.mean_exchange_rejections()))`
  between scans. For the reference's doubling rounds, collect 1, 2, 4, … scans
  on successive ladders. Changing the ladder retains replica states and scan
  parity and resets the diagnostics. Hold the ladder fixed for production
  samples. `reset_statistics()` starts a new observation window on the same
  ladder and tracks trips starting at its current hot configuration.

The ensemble owns the problem, physical replicas, local kernels, workspaces,
and RNGs. `Problem` can still explicitly borrow immutable resources through
`std::cref`. Exchanges swap physical IDs in the slot table. Each geometry
workspace stays with its configuration. A kernel that caches target-dependent
data must provide `invalidate_target_cache()`, as the Gaussian family does.
The ensemble calls it before that replica explores a different beta. Use local
kernels that preserve their given tempered distribution with fixed settings.
The exchange phase reads cached untempered densities. A local batch that
accepts a move refreshes its exchange density once at the end.

On a feasible polytope (P), the slot distribution is proportional to
`1_P(x) exp(beta * ell(x))`, where `ell` is the entire model log density. At beta
zero the reference is uniform on the geometry, including regions where the
model gives zero mass. At each positive beta, a model value of negative
infinity gives zero mass. Initial states at positive beta must have finite log
density, and model evaluations of NaN or positive infinity raise an error.
A non-uniform model prior is powered along with the likelihood.
The reference must have finite positive volume. Identity and affine
coordinates are supported, since their volume factor is constant.

The exchange log ratio for slots `i,j` is
`(beta[j] - beta[i]) * (ell(x[i]) - ell(x[j]))`, clipped at zero.
Each local kernel preserves its slot's marginal, so independent exploration
preserves their product. Each disjoint set of pairwise Metropolis exchanges
also preserves that product. With fixed replica-specific kernel settings, the
invariant law on states and labels is the slot product times a uniform label
permutation. The Metropolis ratio is unchanged when both labels are swapped.
Alternating the matchings deterministically
preserves the joint target at every phase, while the ordered composition
generally breaks reversibility. With zero-based slots, the first scan attempts
edges 1, 3, …, then 0, 2, … on the second scan, matching the Python scan numbering.

The C++ implementation treats exactly beta zero as uniform. In Python the
uniform branch runs for beta ≤ 1e-9, which can disagree with a positive beta
used in exchange. All slots use each physical replica's supplied kernel, with
a uniform target at beta zero. A coordinate Hit-and-Run kernel with
`UniformLineKernel` reproduces the reference's uniform explorer. Supply
geometry in a common sampling frame. Rounding is a separate operation.

Adaptation follows the reference's inverse PCHIP interpolation, with evenly
spaced barrier levels divided by the number of gaps (slots minus one), as in
[Syed et al., Algorithm 2](https://arxiv.org/abs/1905.02939). A rejection floor
of 1e-10 is applied to the mean estimates for interpolation. Diagnostics retain
the actual estimates, and an all-zero barrier leaves the ladder unchanged.

## Algorithms and invariants

- Hit-and-Run composes a direction generator and a line kernel. Coordinate,
  cycling-coordinate and dense directions share the same transition logic.
  Uniform and overrelaxed line kernels preserve uniform targets directly;
  Gaussian displacement proposals require the translated-chord MH correction.
- Gaussian random walk, Dikin walk and CSmMALA share `GaussianKernel` with drift
  and metric policies. Proposal symmetry removes reverse-density work only when
  mathematically valid. Dikin factors and accepted target data are reused.
- Linear polytopes cache per-chain slacks. Coordinate chord computation and
  accepted-step updates cost O(number of constraints), without matrix products.
- Bilinear polytopes support disconnected interval unions with uniform or
  Gaussian-displacement Hit-and-Run. Overrelaxation and Gaussian-family kernels
  are intentionally unsupported on this geometry. Long-double filters and exact
  rational fallback protect near-degenerate topology and feasibility decisions.

Every chain needs its own kernel, workspace and RNG stream. Problems may be shared
immutably. A proposal leaves accepted state and slacks unchanged until acceptance;
workspace direction scratch must remain untouched while a proposal is pending.
Rejection preserves accepted caches. Do not retain Eigen views across acceptance,
which may swap buffers. Raw `step` calls require the caller to maintain these
contracts; `BoundChain` fixes target lifetime and cancels failed evaluations.
Changing the mathematical target requires invalidating Gaussian target caches or
rebinding the problem. Changing geometry or state requires a fresh binding/kernel.
No general serialization, pickling or multiprocessing execution layer is provided.

## Bilinear numerical contract

For `z=[x,y]`, constraints are `A*z<=b` and
`x.transpose()*B[k]*y + affine.row(k)*z <= rhs[k]`. Both blocks are nonempty;
every cross matrix has shape `(split, dimension-split)`. Arbitrary quadratic
forms and transformations mixing the blocks can introduce unsupported within-block
terms. Boundaries are closed. Positive feasibility tolerance shrinks support;
negative tolerance relaxes diagnostic checks without changing line domains.

Slices retain every interval component, merge touching intervals and round finite
endpoints inward. A positive-width component that cannot be represented raises
an error. Unresolved roots, overflow and underflow also fail explicitly. No small
quadratic coefficient or negative discriminant is silently rounded to zero.
Long-double filters, compensated discriminants and rare exact rational fallback
protect near-degenerate cases. Exact fallback may allocate. Ordinary queries
reuse aligned workspace buffers; linear slacks refresh every 256 accepted moves.
Arithmetic assumes binary round-to-nearest operation without fast-math.

Returned C++ interval ranges borrow workspace storage until its next query.
Uniform proposals need finite positive total length; Gaussian displacements need
positive Gaussian mass. Start in feasible positive-volume support: ambient
uniform sampling on singular equalities is unsupported. Direction support alone
does not guarantee mixing across disconnected components.

## Build and validation

From the repository root:

```sh
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DHOPSY_BUILD_CPP_BENCHMARKS=ON
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
```

Tests retain exact geometry/proposal checks, numerical regressions, allocation
guards, target/cache lifecycle tests and fixed-seed distribution checks against
known moments. The standalone threaded bilinear smoke test is registered in CTest
and can also be built with compiler sanitizer flags.

The Dirichlet stationary-distribution integration tests run by default and can
also be selected with `ctest --test-dir cpp/build -L '^dirichlet$' --output-on-failure`.
They exercise all 13 kernel configurations as standalone chains and with NRPT
on a two-dimensional simplex embedded in three dimensions. The cold target is
`Dirichlet(8, 2, 2)`. At inverse temperature `beta`, the reference shape of each
component is `1 + beta * (alpha - 1)`.

Each kernel starts from four dispersed interior points. The tests compare means,
variances, covariances, and marginal quartile probabilities to exact values, and
a joint tail probability to independent gamma-normalized draws. Batch means
estimate Monte Carlo error from the correlated samples. The estimated standard
error must be below 2.2% of the independent-draw standard deviation for each
statistic. The observed discrepancy must be below both six standard errors and
8% of that standard deviation.

See [benchmarks](benchmarks/README.md) for representative native and threaded
measurements. Python binding tests are in `bindings/tests`.
