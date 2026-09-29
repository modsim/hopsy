# hopsy C++ Core — Hit-and-Run Design Checkpoint

## Goal

Implement a modern C++20 sampling core for hopsy that is composable, testable, extensible to new geometries and targets, and competitive with the existing hopsy implementation.

The immediate milestone is uniform Coordinate Hit-and-Run and ordinary Hit-and-Run on a linear polytope, tested first on a box.

Performance of Coordinate Hit-and-Run is a first-class requirement. In particular, the implementation must preserve the \(O(m)\) per-transition geometry cost obtained from cached constraint slacks.

## Public namespace and organization

The stable C++ API lives directly in:

```cpp
namespace hopsy
```

Unstable future functionality may live in:

```cpp
namespace hopsy::experimental
```

Implementation details may use `hopsy::detail`.

The filesystem is organized by domain meaning rather than technical category.

```text
hopsy/
    linalg.h
    random.h
    direction.h
    geometry/
        linear_polytope.h
    target/
        ...
    kernel/
        hit_and_run.h
```

`linalg.h` contains the canonical Eigen-backed vocabulary:

```cpp
template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using Index = Eigen::Index;
```

The RNG alias is separate from linear algebra.

## Directions

Directions are shared geometric vocabulary rather than part of a particular geometry or kernel.

The first special direction representation is:

```cpp
struct CoordinateDirection {
    Index coordinate;
};
```

A distinct struct is used instead of an `Index` alias so that coordinate directions are type-safe and can participate in overload resolution.

Ordinary dense directions can initially use `Vector<Scalar>` directly.

A model jump in a trans-dimensional state is not considered a geometric direction.

## LinearPolytope

`LinearPolytope<Scalar>` represents

\[
Ax \le b.
\]

It owns `A_` and `b_` privately and establishes structural invariants in its constructor. Malformed public input results in exceptions such as `std::invalid_argument`; internal implementation invariants are checked with debug assertions where useful.

The associated types include:

```cpp
using scalar_type = Scalar;
using vector_type = Vector<Scalar>;
using matrix_type = Matrix<Scalar>;
using state_type = vector_type;
```

Dimensions and indices use `Eigen::Index` consistently.

The basic public geometry API includes:

```cpp
num_dimensions()
num_constraints()
contains(...)
make_workspace(...)
line_domain(...)
```

### Per-chain workspace

The polytope itself is immutable and can be shared between chains.

Each chain owns a mutable workspace:

```cpp
struct Workspace {
    vector_type slacks;
};
```

with

\[
s = b - Ax.
\]

The workspace is nested in `LinearPolytope` because it is associated with that geometry type, but workspace instances are independent objects and are not stored inside the shared geometry.

### Line domains

The intersection of a convex polytope with a line is one interval, represented by a geometry-specific nested type:

```cpp
struct LineInterval {
    Scalar lower;
    Scalar upper;
};
```

The geometry exposes overloads of:

```cpp
line_domain(state, direction, workspace)
```

For a dense direction \(d\), the geometry computes

\[
Ad
\]

and combines it with cached slacks to determine the feasible interval.

For a coordinate direction \(e_j\), the implementation directly uses

\[
Ae_j = A_{:,j},
\]

avoiding a matrix-vector multiplication.

This gives the intended Coordinate Hit-and-Run hot path.

After a coordinate step \(\lambda\),

\[
x_j \leftarrow x_j + \lambda
\]

and

\[
s \leftarrow s - \lambda A_{:,j}.
\]

Both chord computation and slack update are therefore \(O(m)\), where \(m\) is the number of constraints.

Expensive checks such as comparing cached slacks against

\[
b-Ax
\]

may be performed with debug-only assertions.

`line_domain` is a public geometry capability, but it is not required of every possible future geometry. Algorithms such as Hit-and-Run require geometries that provide this capability.

Future nonconvex or disconnected geometries may return richer line-domain objects containing several valid intervals.

## Hit-and-Run decomposition

Hit-and-Run is decomposed into three independent responsibilities:

```text
DirectionSampler
      ↓
Geometry::line_domain
      ↓
LineKernel
      ↓
HitAndRunKernel
```

### Direction samplers

Initial implementations are:

```cpp
CoordinateDirectionSampler
UniformDirectionSampler
```

`CoordinateDirectionSampler` returns a `CoordinateDirection`.

`UniformDirectionSampler` returns a normalized dense direction.

These algorithm-specific sampler implementations can initially live in `kernel/hit_and_run.h`.

### Line kernels

The previous `LineSampler` terminology is generalized to `LineKernel`.

A line kernel performs the one-dimensional transition along the line selected by Hit-and-Run and returns a scalar step \(\lambda\).

Initial implementation:

```cpp
UniformLineKernel
```

Future examples include:

```cpp
OverrelaxedLineKernel
MetropolisLineKernel
SliceLineKernel
```

A line kernel does not mutate the multidimensional state or geometry workspace.

Conceptually, its richest useful interface is:

```cpp
step = line_kernel(...);
```

with access to:

```cpp
line_domain
state
direction
target
rng
```

A simple `UniformLineKernel` may use only the line domain and RNG.

Passing the remaining objects by reference has effectively no runtime cost for template code when they are unused. Therefore, a complicated system of `needs_state`, `needs_target`, etc. is not required for performance.

C++20 concepts may nevertheless be introduced to express compatibility between line kernels, line-domain types, targets, and directions. More elaborate signature-based capability dispatch should only be introduced when concrete line kernels demonstrate a need for it.

## HitAndRunKernel

The composition is:

```cpp
HitAndRunKernel<DirectionSampler, LineKernel>
```

A transition conceptually performs:

```cpp
direction = direction_sampler(...);

domain = geometry.line_domain(
    state,
    direction,
    workspace
);

step = line_kernel(
    domain,
    state,
    direction,
    target,
    rng
);

geometry applies the accepted step and updates its workspace
```

The geometry rather than the generic Hit-and-Run implementation should own performance-sensitive state/workspace updates, because it knows how the direction is represented and how its cache must change.

## advance versus propose

At the user level, normal sampling should ultimately look like:

```cpp
sample(...)
```

The sampling driver interacts only with complete Markov kernels through an operation such as:

```cpp
advance(...)
```

`advance` performs one complete Markov transition.

Lower-level proposal mechanisms may additionally expose:

```cpp
propose(...)
```

A proposal must not mutate the current chain state before an acceptance decision has been made.

This distinction permits future composition such as:

```cpp
MetropolisHastings<
    HitAndRunKernel<...>
>
```

without changing the top-level `sample()` API.

### Important MH performance requirement

A Hit-and-Run proposal should retain the transition information required to commit the move efficiently.

For Coordinate Hit-and-Run this means at least:

```cpp
direction
step
```

in addition to whatever candidate-state representation is required.

If an MH proposal is accepted, the state and polytope workspace must be updated incrementally using this information.

We must not recompute

\[
b-Ax
\]

after an accepted coordinate proposal, because that would turn the geometry update from \(O(m)\) back into \(O(mn)\).

The exact proposal/commit API remains to be designed, but this performance requirement is fixed.

## Sampling and storage

The sampler owns the current state and repeatedly invokes the outer Markov kernel.

Sample storage is separate from kernel logic.

For fixed-dimensional chains, samples can eventually be stored efficiently in a preallocated Eigen matrix.

Compressed storage of repeated rejected states is an interesting possible optimization, but it is not required for the first implementation and should not complicate the kernel API.

## Immediate implementation sequence

1. Finish `LinearPolytope`.
   - defensive constructor
   - metadata
   - `contains`
   - workspace creation
   - dense `line_domain`
   - coordinate-specialized `line_domain`
   - efficient state/workspace updates

2. Add exact Catch2 tests for every `LinearPolytope` operation using a simple box.

3. Implement and test `CoordinateDirectionSampler`.

4. Implement and test `UniformDirectionSampler`.

5. Implement and test `UniformLineKernel`.

6. Assemble Coordinate Hit-and-Run:

```cpp
HitAndRunKernel<
    CoordinateDirectionSampler,
    UniformLineKernel
>
```

7. Assemble ordinary Hit-and-Run by replacing only the direction sampler:

```cpp
HitAndRunKernel<
    UniformDirectionSampler,
    UniformLineKernel
>
```

8. Add deterministic/invariant end-to-end tests on the box before adding statistical convergence tests.

## Performance principles

The current design should continue to preserve the following properties:

- no virtual dispatch in the sampling hot path;
- compile-time composition through templates;
- immutable geometries shareable across chains;
- mutable per-chain workspaces;
- cached linear-polytope slacks;
- \(O(m)\) Coordinate Hit-and-Run line-domain calculation and accepted-step update;
- no unnecessary state mutation before MH acceptance;
- no unnecessary allocation in the eventual hot path;
- expensive invariant checks restricted to debug builds;
- storage policy kept separate from transition logic.

These principles should be used as the benchmark when reviewing later architectural changes.



## Native target interface

### Motivation

Python-defined target densities are convenient, but they introduce a potentially severe performance bottleneck during parallel sampling. A typical sampling loop may evaluate the target millions of times. If each evaluation requires a transition from the hopsy C++ backend into Python and then into another compiled extension, the Python boundary remains part of the hot loop.

This is particularly relevant for packages such as 13CFLUXv3. Its computational backend is implemented in C++, but its current hopsy-facing interface is exposed through Python. A call path may therefore look like

```text
hopsy C++
    -> Python target callback
    -> x3cflux Python binding
    -> 13CFLUX C++
    -> Python
    -> hopsy C++
```

even though the actual likelihood evaluation is native code.

Free-threaded Python alleviates the GIL bottleneck for Python targets, but compiled target implementations should ideally avoid Python entirely during sampling.

The goal is therefore to provide an optional **native target interface** that allows independently compiled extension modules to expose a target directly to the hopsy C++ backend.

### Design principle

Python should act only as the composition and setup layer.

A target provider such as x3cflux performs a one-time handshake with hopsy through Python. After this handshake, hopsy stores a native function table and opaque target context. All target evaluations during sampling are then native calls between shared libraries.

```text
Python
    |
    | one-time setup
    v
hopsy C++
    |
    | native function pointer
    v
external target backend
    |
    v
native target implementation
```

Python is not involved in the sampling hot loop.

### ABI boundary

Hopsy should define a small C-compatible, versioned ABI rather than expose hopsy C++ classes or require external packages to link against hopsy internals.

A minimal first version could be

```cpp
struct HopsyNativeTargetV1 {
    std::uint64_t abi_version;
    std::uint64_t struct_size;

    void* context;

    double (*log_density)(
        void* context,
        const double* state,
        std::size_t dimension
    );

    void (*destroy)(
        void* context
    );
};
```

The `context` pointer is opaque to hopsy.

For example, x3cflux may internally store a pointer to a 13CFLUX model or simulator. Hopsy must never cast or dereference this pointer itself. It only passes the pointer back to functions supplied by the target provider:

```cpp
return api_->log_density(
    api_->context,
    state.data(),
    state.size()
);
```

Only code compiled inside the external target provider needs to understand the concrete type stored in `context`.

This avoids exposing Eigen types, C++ templates, pybind11/nanobind types, or external implementation classes across the binary interface.

### Python handshake

The native target can be exported through a standard `PyCapsule`.

For example, an external package may implement a Python-level protocol such as

```python
model.__hopsy_native_target__()
```

which returns a capsule named

```text
hopsy.native_target.v1
```

Hopsy inspects the target once during chain construction. If the protocol is available, it extracts the native interface pointer and wraps it in a C++ `NativeTarget`.

Conceptually:

```python
model = x3cflux.HopsyModel(...)

chain = hopsy.MarkovChain(
    model=model
)
```

During construction:

```text
x3cflux Python object
        |
        | __hopsy_native_target__()
        v
PyCapsule
        |
        v
HopsyNativeTargetV1*
        |
        v
hopsy::NativeTarget
```

During sampling:

```text
hopsy kernel
    |
    v
NativeTarget::log_density()
    |
    v
function pointer
    |
    v
13CFLUX C++
```

No Python callback is performed for individual target evaluations.

### Binding-library independence

The native target interface must not depend on either pybind11 or nanobind.

This allows, for example,

```text
x3cflux:
    pybind11

hopsy:
    nanobind
```

while still sharing native targets.

Both sides use the CPython `PyCapsule` mechanism only for the initial pointer exchange. The actual ABI consists solely of plain C-compatible data and function pointers.

The binding libraries therefore disappear entirely from the runtime target interface.

### Lifetime management

The target context must remain valid for the lifetime of the hopsy chain.

A simple implementation is for hopsy to retain the Python capsule object:

```cpp
class NativeTarget {
    nb::object capsule_;
    const HopsyNativeTargetV1* api_;
};
```

The capsule is only retained for lifetime management. It is not touched during sampling.

The lifecycle is therefore

```text
construction:
    Python interaction

sampling:
    native interface only

destruction:
    capsule released under Python
```

The exact ownership contract should be documented explicitly. In particular, it must be clear whether the capsule owns the underlying context and which component
