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

Bound chains expose `state_log_density()` and `proposal_log_density()` as
`std::optional<double>`. The first reports the accepted state's cached value.
The second reports the value from the most recent proposal, including a rejected
proposal. An empty result means no value was evaluated for that state or proposal.
These accessors do not evaluate the target. Both values are in sampling
coordinates and include any log-volume adjustment from the coordinate map.
NRPT caches the untempered values with each replica and exposes them in
temperature order through the corresponding slot accessors. The low-level
`draw` and `draw_into` functions reuse values within a batch.

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

See [benchmarks](benchmarks/README.md) for representative native and threaded
measurements. Python binding tests are in `bindings/tests`.
