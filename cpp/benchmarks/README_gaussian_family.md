> Cache/lifecycle follow-up: [README_csmala_cache.md](README_csmala_cache.md)
> records the accepted-state caching implementation and the matched single-chain
> benchmark. Measurements and implementation descriptions below are historical.

> Historical checkpoint: legacy CSmMALA clipping is now fixed and Gaussian-family
> nanobind bindings are available. See [the Python/profiling follow-up](README_python_profile.md)
> for matched-clipping native results and the four-chain multiprocessing/thread benchmarks.

# Gaussian-family modernization checkpoint

The new core implements:

- `GaussianKernel<State, ZeroDrift, IdentityMetric>`: covariance `sigma^2 I`.
- `DikinWalkKernel<State>`: covariance `sigma^2 G(x)^-1`.
- `CSmMALAKernel<State>`: the same covariance and mean
  `x + sigma^2/2 G(x)^-1 clip_1(gradient log pi(x))`.

Here `G(x) = A^T diag((b-Ax)^-2) A`. CSmMALA currently uses the Dikin
metric only (legacy `fisher_weight=0`). Neither Fisher mixing nor tempering is
part of this checkpoint. All scales are positive, finite standard-deviation
multipliers in metric coordinates; no implicit dimension normalization is used.

The accepted state and geometry workspace belong to the caller. Current and
candidate metric factors belong to the kernel and are initialized lazily. A
noninterior or singular current metric throws; an unusable candidate metric
rejects. Accepted candidates promote their factor; rejections preserve the
current factor. Factors do not depend on sigma, and drift is recomputed with
each proposal's scale. Setters may be called before sampling or after a completed
transition. Target, state, workspace and scale must stay fixed while a proposal
is pending. Arbitrary external state replacement needs reinitialization.

`gaussian_proposal_is_symmetric` is independent of `always_accept`: even a
symmetric random walk rejects infeasible candidates and applies the target MH
ratio. The identity/zero-drift specialization compiles out reverse factors and
reverse-density calculations. For other policies the kernel computes the full
Gaussian reverse-minus-forward correction after feasibility checks. Its
`Proposal::log_proposal_ratio` is NaN until evaluation has completed for an
asymmetric proposal. `log_acceptance_ratio()` is non-const because it prepares
candidate caches. No Gaussian interface is required by Hit-and-Run.

## Reproduce the native comparison

From the repository root, using the dependencies already present in `cpp/build`:

```sh
c++ -std=c++20 -O3 -DNDEBUG \
  -Icpp/include -Icpp/build/_deps/eigen3-src \
  -Icpp/build/_deps/pcg_cpp-src/include -Iextern/hops/src \
  cpp/benchmarks/gaussian_family_benchmark.cpp \
  extern/hops/src/hops/Utility/DefaultDimensionNames.cpp \
  -o cpp/build/gaussian_family_after
python3 cpp/benchmarks/run_gaussian_family.py \
  --samples 500000 --repeats 5 \
  --output cpp/benchmarks/results/gaussian_family_final
```

No legacy source is changed or patched. Both backends use identical
`-0.5*x.squaredNorm()` target arithmetic; legacy CSmMALA receives the corresponding
`hops::Model` adapter. Gaussian and Dikin legacy kernels use a cached current
target density, as an MH wrapper can; the new kernel currently evaluates both
target densities. Native timings exclude nanobind/Python overhead.

The target is standard `N(0,I)` truncated to `[-3,3]^10`, initialized at zero.
Each run has 20,000 untimed warm-up steps and 500,000 retained steps, without
thinning. Five independent seeds (42..46) are used. Execution order reverses
on alternating repeats. Timings include sample storage, but exclude setup,
file I/O, and ESS computation. Numeric-library threading is set to one.
New kernels use `EmptyWorkspace`: legacy proposals likewise do not maintain an
accepted-slack workspace for use by other kernels. Correctness of cached
geometry workspaces is covered separately by unit tests.

Scale is fixed, not tuned: GRW sigma=0.7, Dikin/CSmMALA sigma=0.3. Legacy scale
conversion matches proposal covariance: GRW `sigma`, Dikin `10*sigma^2`,
CSmMALA `sqrt(10)*sigma`. On this axis-aligned box the legacy Dikin orientation
issue does not alter covariance; an off-diagonal unit test independently checks
the new implementation. CSmMALA deliberately differs from legacy near zero
because it clips at norm 1 rather than normalizing every nonzero gradient.

ESS is calculated by FFT autocovariance with Geyer's initial-positive,
monotone paired autocorrelation estimator. The minimum ESS over the ten
coordinates is reported for each run. ESS/sample includes repeated rejected
states; ESS/second is that minimum ESS divided by sampling time. Summary columns
are medians over repeats, separately, so multiplying summary medians need not
reproduce the median ESS/second. Individual means and second moments are saved.
For this target, each coordinate's second moment is approximately 0.9733369.
These are diagnostic estimates on one fixed workload, not a general mixing or
speed guarantee, and no claim of optimal step sizes is made.

## Hit-and-Run baseline

The existing source is `cpp/benchmarks/hit_and_run_benchmark.cpp`. Before core
source changes it was compiled with the same `c++ -std=c++20 -O3 -DNDEBUG`
flags and Eigen/PCG include paths above (no legacy include path needed), then
run as `cpp/build/hit_and_run_before`. Its output is
`results/hit_and_run_before.csv`: five timing repetitions of 1,000,000
iterations per measured component, reporting the median.

A shorter reproducibility variant substitutes `100'000` for `1'000'000` in
that source without other changes. It was compiled before/after and run as
`hit_and_run_100k_before` and `hit_and_run_100k_after`. Results are stored in
`results/hit_and_run_100k_{before,repeat_before,after}.csv`. The repeat-before
file was rerun from the frozen pre-change binary shortly before the after run.
The after run is shorter than the original baseline; use the paired 100k
results for timing comparisons.

Both full-size before/after executables are byte-identical SHA256:
`9b344f5c71eead37f906f1d663f1574545086fc83f4bddc6c2e038e14a22fcdc`.
Both 100k executables are byte-identical SHA256:
`097f0993ded17f2884a363e0e440dbe31b83b86ab5c560893cda6dcb62ec7e92`.
Thus these Gaussian changes introduce no generated-code change to that
Hit-and-Run workload. Small timing differences between these identical binaries
reflect environment variation. Machine: AMD Ryzen 9 5950X; GCC 16.2.1;
Linux x86_64. CPU frequency scaling was left enabled.

`results/gaussian_family.csv` and `.json` retain the initial three-repeat
comparison before scratch-buffer reuse and before selecting EmptyWorkspace.
`results/grw_before.csv` records an additional pre-change GRW run using the
cached workspace. Those timings have different configurations and should not
be presented as paired comparisons with the final run.

## Validation

```sh
cmake --build cpp/build --target hopsy_core_tests -j2
cpp/build/tests/hopsy_core_tests
```

Tests cover independent off-diagonal precision/covariance identities, full
forward/reverse density ratios, rejection before candidate target evaluation,
current factor reuse after acceptance/rejection and scale changes, singularity,
strict interior, clipping at zero/below/above one, float/fixed-size states,
post-warmup Dikin Eigen buffer reuse, and fixed-seed truncated-normal/uniform
stationary moments. Existing random-walk and Hit-and-Run tests remain enabled.

Fisher mixing, configurable boundary-aware gradient clipping, nanobind exposure,
and cross-kernel interoperability remain follow-up work. The paper linked in
[the brief](https://arxiv.org/html/2602.14616v2) clips the drift using distance to constraints, which is different
from clipping gradient norm; no configurable clipping API has been introduced.

## Observed final results

Medians of five 500,000-sample runs (minimum coordinate ESS):

| Kernel | Steps/s | ESS/sample | ESS/s |
|---|---:|---:|---:|
| Legacy GRW | 4,628,623 | 0.02952 | 136,572 |
| New GRW | 4,297,917 | 0.02952 | 126,571 |
| Legacy Dikin | 780,471 | 0.01061 | 8,081 |
| New Dikin | 722,141 | 0.01061 | 7,432 |
| Legacy CSmMALA | 642,165 | 0.01469 | 9,290 |
| New CSmMALA | 558,657 | 0.01300 | 6,980 |

GRW and Dikin have identical retained trajectories for matched seeds in this
workload. The new implementations remain approximately 7.1% and 7.5% slower,
respectively, than legacy; new CSmMALA is approximately 13.0% slower in raw
throughput and 24.9% lower in median ESS/s. The CSmMALA trajectory changes with
the requested clipping fix, and ESS is an estimate, so its ESS difference is not
an isolated runtime regression. No performance parity with legacy is claimed.

The Gaussian family currently reevaluates accepted-state density and drift;
legacy caches current density/drifted means. These are concrete differences,
not a profiling-based attribution of the entire gap. Caching scaled drifted
means would require scale invalidation; caching unscaled gradients would require
an explicit target-lifetime/change contract. Neither was added to hide the
benchmark gap. Further profiling and a separately reviewed caching feature are
appropriate next steps.

Paired 100k Hit-and-Run timings (ns/transition):

| Direction | Dimension | Before | After |
|---|---:|---:|---:|
| Coordinate | 8 | 21.6 | 20.9 |
| Coordinate | 64 | 128.3 | 129.3 |
| Coordinate | 256 | 428.6 | 433.4 |
| General | 8 | 197.0 | 204.2 |
| General | 64 | 2,434.9 | 2,484.7 |
| General | 256 | 30,163.7 | 30,005.7 |

The measured differences range from -3.2% to +3.7%; generated code is identical.
