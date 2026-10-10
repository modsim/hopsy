# Experimental hit-and-run bindings

This directory builds `_hopsy_core` with nanobind. It contains the bindings,
Python tests, and a usage example. No core headers are replaced.

## Integration

In the root CMakeLists.txt, after the core target and dependency setup:

```cmake
if(HOPSY_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
```

Remove any earlier root-level `nanobind_add_module(_hopsy_core ...)` call:
the target is now defined in `bindings/CMakeLists.txt`, with `bindings.cpp`
relative to that directory. The stub's `src/python/bindings.cpp` path was fixed.

Python and nanobind must already be configured (as in the dependency setup we
discussed). Discover Python in the parent scope, before configuring dependencies:

```cmake
if(HOPSY_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
endif()
```

For local tests, install NumPy and pytest into that same Python environment and
ensure `enable_testing()` is called in the root CMakeLists.txt. Then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DHOPSY_BUILD_PYTHON=ON -DHOPSY_BUILD_PYTHON_TESTS=ON
cmake --build build -j 3
ctest --test-dir build -R hopsy_python_bindings --output-on-failure
```

CTest supplies the extension directory automatically. For direct use with a
single-configuration generator, set `PYTHONPATH=build/bindings`; multi-configuration
generators may place the extension in `build/bindings/Release` instead.

`NOMINSIZE` retains Release optimization for instantiated sampling algorithms.
The install destination defaults to `hopsy/experimental`, configurable with
`HOPSY_PYTHON_INSTALL_DIR`. This directory does not provide wheel packaging or a
Python facade; those can be added to the existing hopsy package separately.

## Python API

```python
import numpy as np
import _hopsy_core as h

n = 3
geometry = h.LinearPolytope(np.vstack((np.eye(n), -np.eye(n))), np.ones(2*n))
kernel = h.UniformCoordinateHitAndRunKernel(geometry, np.zeros(n), seed=42)

accepted = h.step(kernel)  # bool
samples = h.draw(kernel, 10_000, thinning=3)
# shape: (dimension, num_samples), float64, samples stored as columns

# Equivalent methods:
accepted = kernel.step()
samples = kernel.draw(10_000, thinning=3)

output = np.empty((n, 1000), dtype=np.float64, order='F')
h.draw_into(kernel, output, thinning=3)
# kernel.draw_into(output, thinning=3) is equivalent.

current = kernel.state  # detached copy, not an editable view
kernel.reset(np.zeros(n))  # rebuilds candidate + geometry workspace
kernel.seed(123)  # resets RNG and direction-generator caches
```

All bound numeric types use double precision. Inputs must be NumPy arrays (or
compatible arrays accepted by the Eigen caster), not Python lists. Geometry
construction copies `A` and `b`; kernel construction copies `initial_state`.

Each Python kernel is a small owning wrapper around one concrete core template
instantiation. It owns the accepted state, candidate, geometry workspace, and RNG.
Geometry is immutable through this API and shared between kernels without another
matrix copy. The wrapper intentionally does not expose the raw propose/accept
lifecycle or mutable state buffers: coordinate cache synchronization and dense
buffer swaps remain internal. There is no virtual kernel hierarchy.

| Direction generator | Uniform chord | Gaussian displacement | Overrelaxed chord |
|---|---|---|---|
| Random dense | `UniformHitAndRunKernel` | `GaussianHitAndRunKernel` | `OverRelaxedHitAndRunKernel` |
| Cycling coordinate | `UniformCoordinateHitAndRunKernel` | `GaussianCoordinateHitAndRunKernel` | `OverRelaxedCoordinateHitAndRunKernel` |
| Random coordinate | `UniformRandomCoordinateHitAndRunKernel` | `GaussianRandomCoordinateHitAndRunKernel` | `OverRelaxedRandomCoordinateHitAndRunKernel` |

Every constructor takes `(geometry, initial_state, seed=0)`. Gaussian variants
also accept `standard_deviation=1.0` as their fourth argument. This controls the
proposal distribution, not the target. The geometry and starting state must be
finite, dimensionally compatible, and the starting state feasible. No rounding
or initial-point search is performed. The core line kernels retain their domain
requirements (e.g. positive-length bounded chords for uniform sampling).

## Targets and threads

Omitting `target` (or passing `None` or an exact `UniformTarget()` instance) uses
the native uniform target and releases the GIL around the C++ sampling operation.
The same API also accepts a Python object with a `log_density(state)` method:

```python
class LinearTarget:
    def log_density(self, state):
        return float(state[0])

samples = kernel.draw(10_000, thinning=3, target=LinearTarget())
```

Python targets keep the GIL held. The attribute is resolved once per sampling
call, while density evaluations happen as needed in the C++ loop. Each evaluation
receives a detached NumPy snapshot, so retaining or modifying it cannot damage the
chain. Those snapshots incur copies and callbacks; use the native uniform path
for core throughput benchmarks. UniformTarget subclasses are treated as Python
targets so overrides are respected. Native nonuniform targets and affine target
wrappers can be bound later.

Independent native chains can run from multiple Python threads. A nonblocking
per-chain lock rejects simultaneous or reentrant operations on the same chain.
The lock is acquired once per batch, not once per sample. This build does not
opt into nanobind's free-threaded CPython mode; that requires separate validation.
The caller must not mutate output arrays from another thread while draw_into runs.

If target evaluation raises, the exception propagates and the pending core
proposal is discarded by rebuilding caches from the last accepted state. The
kernel remains usable. Completed transitions and consumed RNG values are retained;
coordinate cycling and direction-distribution caches restart. Partially written
`draw_into` output is not rolled back. Invalid counts, dimensions, and reset states
are rejected before advancing the chain.

## Output semantics

One step means one attempted transition. Thinning counts attempts, including
rejections. Stored samples may therefore repeat. The initial state is not stored
as a separate sample. Output remains in sampling coordinates.

`draw` returns an owning matrix whose NumPy view remains valid after the kernel is
advanced or deleted. `draw_into` requires writable float64 storage and disables
implicit conversion, so it cannot silently write into a temporary. C- and
F-contiguous arrays and ordinary positive-stride slices are supported. Overlapping
or negative-stride output layouts are rejected. Getters (`state`, `A`, `b`) return
detached arrays; use `reset` to change chain state safely.

## Validation scope

Tests cover all nine instantiations, native and Python targets, sample shapes and
thinning, deterministic seeded replay, ownership/lifetimes, reset/reseed,
C/F/sliced output buffers, invalid arguments, rejection, Python exceptions,
reentrancy, and independent chains in Python threads. Fixed-seed nonuniform tests
check the known mean of exp(x0) on [-1,1]^2.

The supplied archive contained only the binding CMake stub. Validation therefore
used the core from this session with the agreed void-propose and synchronized
coordinate accept/reject changes applied in the test workspace. The bindings do
not depend on the return value of propose() and also work with its earlier
reference-returning signature. Tested environment: Linux x86-64, GCC 13.3,
Python 3.12, NumPy 2.3.5, nanobind 2.12.0, Eigen 5.0.1, and Boost 1.89.0.

Validation result: 43 Python tests passed. The installed module import and
example execution also passed; all 66 C++ tests passed in the reconstructed core
test workspace. The example's uniform second moments were approximately 0.332, 0.332,
and 0.330 (expected 1/3). The Python test count excludes the separate install and
example smoke checks.

Nanobind references:
- https://nanobind.readthedocs.io/en/latest/eigen.html
- https://nanobind.readthedocs.io/en/latest/ownership.html
