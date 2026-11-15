# Direct experimental core bindings

`_hopsy_core` exposes concrete C++ problems, kernels, explicit random-number
generators, and persistent chain bindings. The 2.0 API deliberately replaces the
former geometry-owning Python kernel wrappers and per-call `target=` arguments.

## Example

```python
import _hopsy_core as core
import numpy as np

n = 2
geometry = core.LinearPolytope(
    np.vstack((np.eye(n), -np.eye(n))), np.full(2*n, 4.0)
)
target = core.GaussianTarget(np.zeros(n), np.array([[1., .5], [.5, 1.]]))
problem = core.Problem(geometry, target)
initial_state = np.zeros(n)
kernel = core.GaussianRandomWalkKernel(initial_state, step_size=0.7)
chain = core.bind_chain(problem, kernel, initial_state)
rng = core.RandomNumberGenerator(seed=42)

chain.draw(rng, 2_000)  # Discard warm-up.
samples = chain.draw(rng, 20_000, thinning=3)
```

See [the runnable Gaussian example](../examples/gaussian.py). Each column is one
sample. `step_size` is a standard deviation in sampling coordinates, not variance.
`GaussianTarget` takes a covariance matrix. The distribution above is restricted
to the supplied box; its far tails are truncated.

The corresponding C++ construction uses the same objects and operations:

```cpp
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/sampling/owned_chain.h>
// geometry and target are already constructed.
auto problem = hopsy::Problem(std::cref(geometry), std::cref(target));
hopsy::Vector<double> initial_state = hopsy::Vector<double>::Zero(2);
hopsy::GaussianKernel<hopsy::Vector<double>, hopsy::ZeroDrift,
                      hopsy::IdentityMetric> kernel(initial_state, {}, {}, 0.7);
auto chain = hopsy::bind_chain(problem, kernel,
                              hopsy::Vector<double>(initial_state));
hopsy::RandomNumberGenerator rng(42);
auto samples = chain->draw(rng, 20'000, 3);
```

C++ `bind_chain` with an **lvalue** state retains its original borrowed-view
semantics. An **rvalue** state selects `OwnedStateChain`, which owns the accepted
vector and workspace and retains a persistent `BoundChain`. A shared pointer to
an immutable problem optionally transfers shared ownership of that problem to
the C++ owner. Python uses these owning overloads directly: the Eigen caster
copies the NumPy initial vector once and moves it into C++ storage.

The kernel owns its candidate and proposal caches, not the accepted state.
Consequently the initial state appears in both kernel construction and binding.
It must match the kernel candidate when constructing an owning binding; create
a fresh kernel for a different initial state. State snapshots are detached copies.
There is no mutable NumPy view into state, cached metrics, or workspace memory.

## Lifetimes and execution

- A problem constructed through Python borrows immutable geometry, native target,
  and optional map objects. Nanobind retains these borrowed Python objects for the
  lifetime of that concrete C++ problem. Custom Python targets are held by value
  through the callback adapter.
- A Python-held chain shares ownership of its current problem in C++, borrows its
  kernel, and retains that kernel's Python object. Rebinding releases the old
  shared problem. No Python sampling or state-management layer is involved.
  As with C++ shared ownership, avoid strong ownership cycles. A callback target
  that references its chain can use `weakref.ref(chain)`; chains support weakrefs.
- A kernel can belong to one owning binding at a time. A chain and RNG cannot be
  used concurrently or reentrantly. C++ boundary guards enforce these restrictions
  once per public call, outside transition loops. Raw low-level C++ APIs retain
  their existing caller-managed exclusivity contracts.
- `draw` does not rebind, clear accepted-state caches, or allocate coordinate
  evaluation buffers at each batch boundary. It allocates the requested result;
  `draw_into` uses caller-provided output. The resource guards may allocate ordinary
  bookkeeping storage at a call boundary; no such work occurs per transition.
- Native uniform/Gaussian targets release the GIL during sampling. Python callbacks
  retain the GIL and receive owning state copies. Overrides on subclasses of native
  targets use the Python adapter, so overridden densities are respected.
- `rng.seed(value)` reseeds only the RNG. It does not reset a kernel's retained
  direction/distribution state. Construct a fresh kernel and RNG for exact replay.

`Problem(...)` and `bind_chain(...)` select concrete template instantiations at
construction, like C++ template deduction. Returned Python class names beginning
with `_` are implementation details. There is no virtual target/kernel dispatch
inside native transition loops. The bindings are split across translation units
by kernel family to limit compiler memory consumption.

## Operations

```python
accepted = chain.step(rng)
physical = chain.draw(rng, 1000, thinning=2)
sampling = chain.draw_sampling(rng, 1000, thinning=2)

physical_out = np.empty((chain.target_dimension, 1000), order="F")
chain.draw_into(rng, physical_out, thinning=2)
sampling_out = np.empty((chain.sampling_dimension, 1000), order="F")
chain.draw_sampling_into(rng, sampling_out, thinning=2)

state_z = chain.sampling_state
state_x = chain.target_state
chain.step_size = 0.5  # Gaussian-family chains, between completed transitions.
```

Outputs must be writable float64 matrices with compatible dimensions and
nonoverlapping positive strides. C/F contiguous and ordinary sliced matrices
work. Direct kernel step-size access is permitted while unbound; use the chain's
step-size property while bound. Invalid output arguments do not consume the RNG.

All three Gaussian kernels and the nine Hit-and-Run combinations are available:
`GaussianRandomWalkKernel`, `DikinWalkKernel`, `CSmMALAKernel`, and
`{Uniform,Gaussian,OverRelaxed}{,Coordinate,RandomCoordinate}HitAndRunKernel`.
The latter are concrete instantiations of C++ `HitAndRunKernel`; their constructors
also accept the explicit direction-generator and line-kernel objects used in C++.
A uniform problem uses an explicit `UniformTarget()`.

## Coordinates and target changes

Supply an optional `AffineCoordinates(T, shift)` when constructing the problem.
The geometry and initial state are already in sampling coordinates `z`; physical
coordinates are `x = T*z + shift`. Full-column-rank rectangular maps support
PolyRound equality elimination. Targets are evaluated in physical coordinates;
volume corrections and pulled-back gradients are handled by the core problem.
See [coordinate mathematics and historical benchmarks](../benchmarks/README_bound_contract.md).

```python
coordinates = core.AffineCoordinates(T, shift)
problem = core.Problem(rounded_geometry, physical_target, coordinates)
```

For an explicit change of target:

```python
new_problem = core.Problem(geometry, another_gaussian_target)
chain.rebind_problem(new_problem)
```

Rebinding requires the **same concrete problem type**, geometry object, and map
object (for mapped problems). Thus Gaussian-to-Gaussian and custom-Python-to-custom-
Python replacements work; uniform-to-Gaussian requires a fresh binding/kernel.
Target-dependent caches are invalidated once; geometry-only metric factors remain
valid. A fresh owning binding invalidates both state- and geometry-dependent
Gaussian caches. Do not silently mutate a bound target: create/rebind a problem
explicitly, including after changing mathematical parameters on a Python target.
Callbacks are captured at problem construction. Tempering controls and target
families are a separate future feature.

## Build and validation

With the existing configured isolated environment:

```sh
cmake --build cpp/build-python -j 2
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  -m pytest -q cpp/bindings/tests
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  cpp/examples/gaussian.py
```

`bindings/CMakeLists.txt` links the extension to `hopsy::core`. Enable
`HOPSY_BUILD_PYTHON_TESTS` to register the suite with CTest. No legacy package,
facade, or installed extension is changed. API performance measurements and their
reproduction commands are in [README_direct_bindings.md](../benchmarks/README_direct_bindings.md).
