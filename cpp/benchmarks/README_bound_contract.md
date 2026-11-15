# Bound problems, kernel contracts, and coordinates

`problem/problem.h`, `problem/coordinates.h`, and `sampling/chain.h` introduce
a typed binding around the existing caller-owned state and kernel. This is a
view, not a new kernel hierarchy. For example:

```cpp
auto problem = hopsy::Problem(std::cref(geometry), std::cref(target));
auto chain = hopsy::bind_chain(problem, kernel, state);
chain.step(rng);
auto samples = chain.draw(rng, 1000);
```

Problem owns its constructor arguments by value unless borrowing is explicitly
requested with `std::cref`. Its public accessors are const. Borrowed objects must
outlive the problem and remain mathematically unchanged. A BoundChain exclusively
borrows its problem, kernel, and state; do not use another binding or the raw
kernel concurrently. It owns its workspace unless an existing workspace is
supplied as the fourth argument to `bind_chain`.

Binding clears target-dependent caches once. Subsequent steps and batches keep
them. To change target, finish the current binding and bind a new problem.
Geometry or externally changed state requires a fresh kernel/workspace pair.
The low-level free `step` API remains available, but its caller must maintain
these cache invariants explicitly. It no longer checks target addresses.

The kernels retain their existing proposal algorithms and state ownership.
Commit is `accept(state, geometry, workspace)`; cancellation is `reject(state)`.
Unused target/workspace arguments were removed. Symmetry and always-acceptance
remain distinct compile-time properties. Hit-and-Run retains its line proposal
correction and coordinate updates; Gaussian reverse-density work remains lazy.

## Coordinate convention

Geometry, initial state, metric, and step size use sampling coordinates `z`.
The physical target uses `x = T*z + shift`. Supply the already-rounded geometry
and `AffineCoordinates(T, shift)` from PolyRound; constraints are never silently
transformed. The sampling density is

```
log pi_z(z) = log pi_x(T*z + shift) + log sqrt(det(T.transpose()*T))
grad_z log pi_z = T.transpose() * grad_x log pi_x
```

Full-column-rank rectangular T supports equality elimination. Its density uses
intrinsic Euclidean volume on the affine subspace, not ambient marginalization.
QR computes the constant volume once. Identity coordinates return the original
target directly, preserving compile-time uniform traits without mapping buffers.
Mapped targets use separate per-chain buffers, allowing a shared immutable problem.

`chain.draw` emits physical coordinates; `draw_sampling` emits raw sampling
coordinates. `sampling_state()` and `target_state()` make this distinction explicit.
CSmMALA clips the pulled-back gradient in sampling coordinates. Dikin currently
requires linear constraints; general geometry does not imply a general metric.

The mapping protocol also permits nonconstant volume: implement `to_target`,
`log_volume`, `pullback_gradient`, and `log_volume_gradient` (for gradient kernels).
A nonlinear map must represent a valid change of coordinates; its injectivity
and measure are the implementer's responsibility. A test-only exponential map
checks the Jacobian correction. No production nonlinear map is added here.
A disconnected test geometry with no matrix interface checks that Hit-and-Run
and Gaussian random walk do not acquire a polytope dependency. Bilinear geometry
is a separate feature, not implemented in this change.

## Python compatibility

Existing `target=` arguments remain supported. The owning wrapper binds the
target once per call, outside the native loop. It avoids repeating geometry
validation during every Python `.step()` call. Persistent C++ bindings preserve
caches across batches; the compatibility Python facade conservatively invalidates
target caches once per call because it permits mutable Python targets.

All Python kernel constructors accept optional `coordinates=AffineCoordinates`.
Without it behavior is unchanged. `state`, `sampling_state`, `initial_state`,
`reset`, and `dimension` use sampling coordinates; `target_state`,
`target_dimension`, and mapped `draw`/`draw_into` use physical coordinates.
`draw_sampling` explicitly returns sampling coordinates. Output rows must match
the chosen space. Tests exercise actual PolyRound rounding with equality reduction.

```python
# rounded is the result of PolyRound's transform/round operations.
coordinates = core.AffineCoordinates(rounded.transformation.to_numpy(),
                                     rounded.shift.to_numpy())
geometry = core.LinearPolytope(rounded.A.to_numpy(), rounded.b.to_numpy())
chain = core.CSmMALAKernel(geometry, initial_z, step_size=0.3,
                         coordinates=coordinates)
samples_x = chain.draw(1000, target=physical_target)
```

## Contract benchmark

`contract_before` was compiled from `contract_benchmark.cpp` before source edits.
Current direct and bound variants use identical cases and RNG seeds. Each reports
the median of five 100k-transition measurements after 1k warm-up transitions.
The runner alternates executable order over three rounds and checks exact state
checksums. Pinning controls CPU placement, but timings still have frequency noise.
These are transition-throughput checks, not ESS or old/new binding comparisons.

```sh
c++ -std=c++20 -O3 -DNDEBUG -Icpp/include -Icpp/build/_deps/eigen3-src \
  -Icpp/build/_deps/pcg_cpp-src/include cpp/benchmarks/contract_benchmark.cpp \
  -o cpp/build/contract_direct
# Repeat with -DHOPSY_BOUND_CONTRACT and -o cpp/build/contract_bound.
taskset -c 0 python3 cpp/benchmarks/run_contract_benchmark.py
```

Raw paired timings and summarized ratios are in `results/contract_quiet.csv`
and `.json` (final run without concurrent compilation/tests). Earlier runs are
retained as `contract_paired`, `contract_confirmation`, and `contract_final`.
The first two identified a small dense 10D regression; removing unnecessary
exception cleanup from the always-accepted path addressed that extra code path.
The `contract_final` run overlapped compilation, so it must not be used to claim
a speedup. To reproduce the final runner command, add
`--output cpp/benchmarks/results/contract_quiet` above.
The frozen executable is a local artifact; reproducing its source
requires the pre-edit working tree, not the repository HEAD (which predates WIP).

Validation commands:

```sh
cmake --build cpp/build -j 2
cpp/build/tests/hopsy_core_tests
cmake --build cpp/build-python -j 2
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  -m pytest -q cpp/bindings/tests
```

Observed: 110 core cases / 522969 assertions, 88 Python tests (including the
installed PolyRound integration), and the existing legacy CSmMALA oracle's
1 case / 8 assertions passed. Protected WIP/legacy hashes matched the initial
snapshot. No dependency changes were needed.

Final quiet run (Ryzen 9 5950X, CPU 0, GCC `-O3 -DNDEBUG`, M transitions/s):

| Kernel | Dimension | Frozen before | Bound after |
|---|---:|---:|---:|
| Coordinate HR, uniform | 10 | 41.797 | 41.407 |
| Dense HR, uniform | 10 | 3.755 | 3.725 |
| Coordinate HR, uniform | 100 | 5.547 | 5.541 |
| Dense HR, uniform | 100 | 0.231 | 0.228 |
| Coordinate HR, Gaussian target | 10 | 22.130 | 22.184 |
| Gaussian random walk | 10 | 4.452 | 4.926 |
| Dikin | 10 | 0.792 | 0.788 |
| CSmMALA | 10 | 0.645 | 0.663 |

HR latency differences range from -0.24% to +1.27%, within observed run-to-run
variation; this does not prove literally zero cost. Gaussian random walk improved
about 10.6% in throughput, CSmMALA about 2.9%; Dikin was effectively unchanged.
Every executable/round matched exact state checksums. These results measure the
contract refactor with identity coordinates; actual affine mapping necessarily
adds matrix-vector work and was validated for correctness, not claimed free.
