# Partitioned bilinear geometry

## Exact model and supported combinations

The name denotes a possibly nonconvex bilinear-constraint region, not necessarily
a convex polytope. The state is `z = [x, y]`, with a single explicit split `p = dim(x)` and
`q = dim(y) = dimension - p`, both positive. The geometry is

```
A z <= b
x.transpose() B[k] y + affine.row(k) z <= bilinear_bounds[k]
```

Every `B[k]` has shape `(p,q)`. The affine matrix has shape `(k,p+q)` and defaults
to zero. Bounds include constant offsets. No within-block square/cross terms are
allowed: an arbitrary quadratic form is not accepted as a bilinear matrix.
Different constraints share the same partition. Inputs are copied, finite,
immutable float/double data. The Python binding uses double precision.

C++ and Python use the same constructor order:
`(A,b,split,cross_terms,bilinear_bounds,affine=zero)`. Python uses:

```python
geometry = core.BilinearPolytope(
    A, b, split=1, cross_terms=[np.ones((1, 1))],
    bilinear_bounds=np.array([0.2]), affine=None,
)
problem = core.Problem(geometry, core.UniformTarget())
```

Geometry is already in sampling coordinates. The existing problem coordinate
map can transform target evaluation/output, including rectangular maps, but does
not transform constraints. Arbitrary affine rounding that mixes the two blocks
can introduce within-block quadratic terms; do not feed such transformed
constraints to this representation. Block-preserving transformations can be
represented by appropriately transformed B, affine terms, and linear bounds.

`kernel_supports_geometry_v<K,G>` in `kernel/geometry_support.h` is the single
compatibility contract. The geometry marks itself `hit_and_run_only`; that marker
is inherited by C++ subclasses. Its audited allow-list is Hit-and-Run with a
uniform or Gaussian-displacement line kernel. The same trait drives C++ binding
validation, proposal static assertions, Python registration and
`core.supports_geometry(kernel, geometry)`. Unsupported Python bindings fail
with a descriptive error naming the kernel and supported choices. Gaussian random
walk, Dikin, CSmMALA, and all overrelaxed variants are rejected for this geometry.
This is a conservative restriction even for coordinate overrelaxation, whose
individual bilinear slices would be linear. There is no runtime capability check
inside a transition.

## Complete slices and MH

Along `z(t)=z0+t*d`, a constraint gives

```
a = dx.transpose() B dy
b = dx.transpose() B y0 + x0.transpose() B dy + affine.row(k) d
c = x0.transpose() B y0 + affine.row(k) z0 - bound
```

The solver retains the closed solution of `a*t*t+b*t+c<=0`: bounded interval,
two rays, full line, singleton, or empty. It intersects each solution with the
linear slice and all preceding bilinear constraints, retaining **all** components.
The result is sorted and nonoverlapping; touching intervals are merged. There
is no selection of just the component containing zero. With k bilinear
constraints there are at most k+1 components, so both interval buffers reserve
that capacity at workspace creation. Raw workspace calls require a valid paired origin/workspace,
like LinearPolytope; creation/reset and the Python query validate the origin.
The returned C++ range belongs to the workspace and is overwritten by its next
line query; the Python query returns a detached snapshot.
Tiny reused buffers have isolated aligned
allocation to avoid false sharing between independent chains.

The existing line-kernel abstraction already accepts forward ranges. Uniform
sampling chooses a component in proportion to length and then a uniform point;
its correction is zero because translation leaves total feasible line length
unchanged. The Gaussian-displacement kernel chooses by Gaussian mass and uses
`log Z(D,0)-log Z(D,t)` for the reverse correction over the **whole union**.
Nonuniform targets additionally use their log-density difference. The existing
Gaussian normalization implementation is audited with an independent erfc oracle
on a disconnected slice; no new MH rule is introduced.

Direction selection remains state independent. Random dense directions and
full-slice selection give the intended invariant distribution and permit jumps
between positive-volume components. Coordinate updates also preserve the
invariant measure, but may be reducible: in `x*y<0`, they cannot move from quadrant
II to quadrant IV. The example therefore uses dense directions. Capability support
is not a certificate of global mixing. Overrelaxation's single-interval law is
not extended or silently applied to a union.

Uniform kernels reject unbounded, empty, and zero-total-length domains. A
Gaussian-displacement proposal can use unbounded slices for a proper nonuniform
target; the geometry itself does not prove global boundedness/normalizability.
Start sampling at a feasible interior point of a positive-volume target. Singular
sets/lower-dimensional bilinear equalities are not supported as ambient uniform
distributions. A boundary start may produce a zero-measure tangent slice and an
explicit line-kernel error.

## Numerical contract

- `is_feasible(z,tolerance)` uses the existing signed-slack convention: positive
  tolerance shrinks support; negative tolerance is a diagnostic relaxation.
  It never changes the geometry used by `line_domain`.
- Boundaries are closed. The closure of `x*y<0` has the same continuous uniform
  law as the strict region, differing only on a zero-measure set.
- A nonzero quadratic coefficient is never dropped just because it is small.
  Zero direction components produce exactly linear/constant contributions.
- Ordinary coefficients use hardware extended arithmetic and absolute-term
  roundoff filters. Uncertain cancellation/discriminant cases reconstruct the
  polynomial exactly from the binary input values using Boost `cpp_rational`.
  Exact reconstruction is rare and can allocate; normal slice generation reuses
  workspace buffers. Severe cancellation, such as `1e30 + 1 - 1e30`, is tested.
- The discriminant uses compensated products (Dekker splitting); extreme ranges
  use exact power-of-two scaling. Stable q/c-over-q roots avoid cancellation.
  A tiny negative discriminant is not blindly clamped to zero. Exact fallback
  distinguishes true tangencies from tiny forbidden gaps/empty intersections.
- The validated special case of one product with zero RHS/no affine term is
  solved directly from the two factors. It avoids forming a discriminant while
  retaining both components and the correct sign. Ambiguous coincident ratios
  are checked exactly.
- Finite roots are rounded inward when converted to Scalar. A nonrepresentable
  isolated point may disappear (zero measure). A **positive-width** component
  that cannot be represented raises a range error rather than being silently
  removed. Unresolvable distinct roots, finite endpoints outside scalar range,
  coefficient/discriminant overflow or underflow, and invalid inputs fail clearly.
- Linear slacks reuse the established cache, refreshed from the actual state
  every 256 accepted transitions to limit accumulated roundoff. Nonlinear values
  are evaluated from the actual current state rather than recursively cached;
  this avoids an additional source of drifting slice topology. Ordinary floating
  linear updates still warrant feasibility checks with documented roundoff margins.
- Arithmetic assumes binary, round-to-nearest operation without fast-math.

The compensated-product approach follows the error-free arithmetic described by
[Shewchuk](https://www.cs.cmu.edu/~quake/robust.html). Root cancellation is discussed
by [Kahan](https://people.eecs.berkeley.edu/~wkahan/MxMulEps.pdf).
This is finite-precision sampling, not an exact-real-arithmetic claim.

## Independent validation and examples

`tests/bilinear_polytope_test.cpp` covers linear/constant degeneracy, two rays,
three-component intersections, tangencies/repeated roots, near-linear far roots,
scaling, empty/unbounded slices, exact fallback, scalar range failures, float
geometry, workspace allocation guards and dynamic-singleton RNG equivalence.
Random slice tests compare membership with direct constraint evaluation and check
translation invariance. Python tests use an independent `numpy.roots` solver and
midpoint classification, then compare interval endpoints and dense-grid membership.

For `[0,1]^2` and `xy<=c`, with c=.2:

```
Z = c*(1-log(c)) = 0.5218875824868201
E[X] = E[Y] = (c-c*c/2)/Z = 0.34490186400352946
F_X(s) = s/Z                         if 0<=s<=c
       = (c+c*log(s/c))/Z            if c<s<=1
```

The tests use 100k samples, thinning 2, 3k warm-up, and fixed seeds. Means must be
within .012 and marginal CDFs within .015 of these analytic values. These are
conservative deterministic regression tolerances for correlated MCMC samples, not
an iid confidence claim. The line `x+y=1` has intervals `[0,r]` and `[1-r,1]`,
`r=(1-sqrt(1-4c))/2`, explicitly tested at c=.2.

The complement `xy>=.2` has area `1-Z`, mean
`(.5-c+c*c/2)/(1-Z)`, and CDF `(s-c-c*log(s/c))/(1-Z)` for s>c. A separate 80k-sample
chain checks these values; independent box Monte Carlo checks its area.
On `[-1,1]^2`, `xy<=0` has area 2, uniform coordinate marginals, zero means,
`E[X^2]=1/3`, `E[XY]=-1/4`, and equal mass in both sign components. A separate
80k-sample dense chain checks these properties. Nonuniform tests use the known
moments of a standard Gaussian restricted to the same two sign components.

```sh
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  cpp/examples/bilinear_product.py
MPLCONFIGDIR=cpp/build/matplotlib-cache MPLBACKEND=Agg \
  PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  cpp/examples/bilinear.py
```

The second example saves `cpp/build/bilinear.png` and reports independently known
moments/component weights. Matplotlib was installed only in the isolated environment.

## Benchmark protocol

`benchmark_bilinear.py` compares the same native uniform Hit-and-Run kernels on
linear boxes, the dynamic interval container with zero cross constraints, one
opposite-sign constraint, and a dense cross matrix. Dimensions are 2 and 10.
It measures one chain versus four independent native chains in threads, with
1k warm-up, 1M attempted transitions/chain, thinning 20, five timing repetitions
per process and three alternating process-order rounds (15 timings per case),
and CPU affinity 0..3 on the Ryzen 9 5950X. Outputs and thread-pool startup/shutdown
are timed; setup/warm-up/checks are excluded. Coordinate rows for the disconnected
sign geometry are excluded because they do not mix its components.

The `c49f538` extension was frozen before edits at ignored
`cpp/build/bilinear-before`. The baseline mode measures its linear paths; comparing
linear checksums before/after detects changes in RNG consumption/trajectories.
Initial and optimized checkpoints are retained in `results/bilinear_initial.*`
and `results/bilinear_optimized.*`, rather than hiding the slow first version.
Final results and commands are recorded below after validation. A dense B requires
more arithmetic than a linear box; near-equal cost for every bilinear model is not
assumed. Four-chain scaling is reported independently of that geometry overhead.


## Final validation commands

Using the existing configured build directories:

```sh
cmake --build cpp/build -j 2
cpp/build/tests/hopsy_core_tests
cmake --build cpp/build-python -j 2
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
  -m pytest cpp/bindings/tests -q
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -pthread \
  -Icpp/include -Icpp/build/_deps/eigen3-src -Icpp/build/_deps/pcg_cpp-src/include \
  cpp/tests/bilinear_sanitizer_smoke.cpp -o cpp/build/bilinear_sanitizer_smoke
cpp/build/bilinear_sanitizer_smoke
```

Observed: **129 C++ test cases / 555,555 assertions** and **172 Python tests** pass.
The standalone four-thread ownership/allocation smoke test passes ASan/UBSan,
including leak checking (run outside the sandbox because LeakSanitizer cannot
operate under its ptrace restrictions). It is separate from throughput timing.

The examples report `E[XY]=-0.25094355` (exact `-0.25`) and positive-X mass
`0.49725` (exact `0.5`) for the opposite-sign region. The product-bound example
reports means `[0.34720461, 0.34275317]` (exact `0.344901864`), with marginal CDF
errors below `0.0045` at its four diagnostic points. These are seeded checks,
not a guarantee of convergence for arbitrary bilinear constraints.


## Final measured results

```sh
cpp/.benchmark-venv/bin/python cpp/benchmarks/run_bilinear.py \
  --output cpp/benchmarks/results/bilinear_verified
cpp/.benchmark-venv/bin/python cpp/benchmarks/run_bilinear.py --worker-local \
  --output cpp/benchmarks/results/bilinear_verified_worker
```

Both runs use 15 timings per case and verify identical linear output checksums
across the frozen baseline and current module. Module SHA256 hashes, raw times,
and baseline rows are retained in the adjacent JSON files. Rates are **millions
of attempted transitions per second**, summed over chains. These throughput
measurements do not claim equal ESS across different geometries or kernels.

### Main-thread setup; pool startup/shutdown included

| Dimension | Geometry | Direction | One chain | Four chains | Scaling |
|---|---|---|---:|---:|---:|
| 2 | Linear box | Dense | 12.859 | 43.590 | 3.39x |
| 2 | Linear box | Coordinate | 70.042 | 215.998 | 3.08x |
| 2 | Bilinear, no cross terms | Dense | 11.153 | 36.821 | 3.30x |
| 2 | Bilinear, no cross terms | Coordinate | 50.981 | 120.200 | 2.36x |
| 2 | One opposite-sign product | Dense | 6.784 | 22.962 | 3.38x |
| 2 | One dense cross matrix | Dense | 4.314 | 15.890 | 3.68x |
| 2 | One dense cross matrix | Coordinate | 10.146 | 34.484 | 3.40x |
| 10 | Linear box | Dense | 3.909 | 14.371 | 3.68x |
| 10 | Linear box | Coordinate | 36.978 | 122.020 | 3.30x |
| 10 | Bilinear, no cross terms | Dense | 3.673 | 13.423 | 3.65x |
| 10 | Bilinear, no cross terms | Coordinate | 29.766 | 107.040 | 3.60x |
| 10 | One opposite-sign product | Dense | 3.007 | 10.935 | 3.64x |
| 10 | One dense cross matrix | Dense | 1.843 | 6.825 | 3.70x |
| 10 | One dense cross matrix | Coordinate | 7.479 | 25.249 | 3.38x |

### Worker-local setup; persistent pool

The second protocol constructs and warms independent chains on distinct worker
threads (using a barrier), then times only draws with an existing pool. Setup and
pool shutdown are excluded. This is a separate protocol, compared to an equally
prepared frozen baseline, not an optimization secretly applied to only one side.
It helps distinguish sampling work from allocation placement and executor costs.

| Dimension | Geometry | Direction | One chain | Four chains | Scaling |
|---|---|---|---:|---:|---:|
| 2 | Linear box | Dense | 12.944 | 49.686 | 3.84x |
| 2 | Linear box | Coordinate | 71.388 | 257.524 | 3.61x |
| 2 | Bilinear, no cross terms | Dense | 11.490 | 43.625 | 3.80x |
| 2 | Bilinear, no cross terms | Coordinate | 51.806 | 199.386 | 3.85x |
| 2 | One opposite-sign product | Dense | 6.925 | 26.183 | 3.78x |
| 2 | One dense cross matrix | Dense | 4.388 | 16.478 | 3.76x |
| 2 | One dense cross matrix | Coordinate | 10.181 | 34.667 | 3.41x |
| 10 | Linear box | Dense | 3.931 | 15.049 | 3.83x |
| 10 | Linear box | Coordinate | 36.990 | 142.421 | 3.85x |
| 10 | Bilinear, no cross terms | Dense | 3.691 | 14.197 | 3.85x |
| 10 | Bilinear, no cross terms | Coordinate | 30.060 | 114.386 | 3.81x |
| 10 | One opposite-sign product | Dense | 3.020 | 11.489 | 3.80x |
| 10 | One dense cross matrix | Dense | 1.832 | 6.978 | 3.81x |
| 10 | One dense cross matrix | Coordinate | 7.457 | 26.809 | 3.59x |

### Interpretation and remaining performance limits

- Linear single-chain rates are within about 2% of baseline in the main protocol;
  all checked linear trajectories are unchanged. Worker-local dense linear rates
  are unchanged in 2D and approximately 3% faster in 10D.
- Main-protocol four-chain linear coordinate rates are **16.6% slower in 2D and
  11.3% slower in 10D**. We cannot claim uniformly regression-free four-chain
  performance. With worker-local construction these differences shrink to -4.3%
  and +0.6%, respectively. Allocation placement, executor costs and scheduling
  affect these very short transitions; this is consistent with false sharing,
  but hardware counters were not collected to prove that cause. Existing Eigen
  internal buffers still use their original allocation strategy. No global Eigen
  ABI/alignment changes or relocation of user-owned objects were introduced.
- Worker-local dense bilinear paths scale **3.76–3.85x** with four chains;
  main-protocol dense bilinear paths scale **3.30–3.70x**. Near-4x scaling is
  attainable but is not guaranteed for every setup/direction combination.
- In 10D the extra dynamic slice/workspace path alone costs about 6% versus the
  current linear dense kernel. A sparse opposite-sign constraint costs about
  23%, and a dense cross matrix about 53%. In 2D the arithmetic dominates more:
  the sparse constraint is about 47% slower and the dense case about 66% slower.
  The requested near-equal single-chain throughput is therefore **not achieved
  for general bilinear constraints**. The coordinate baseline is especially cheap,
  so evaluating even one nonlinear residual produces a much larger relative cost.
- The initial generic implementation used software long-double FMA on this host
  and was much slower. Replacing that ordinary path with hardware extended
  arithmetic, compensated discriminants and an exact fallback, plus validated
  factored/coordinate special cases, recovered most of that avoidable cost.
  The ordinary path allocates nothing; exact rational fallback may allocate.
- Nonlinear values are recomputed from the accepted origin. Recursively caching
  them could reduce coordinate cost but risks drifting signs/discriminants and
  changing slice topology. That additional numerical contract is deliberately
  left for review rather than trading correctness for the benchmark target.
