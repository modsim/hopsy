# Representative benchmarks

Build with `-DCMAKE_BUILD_TYPE=Release -DHOPSY_BUILD_CPP_BENCHMARKS=ON`.
Run benchmarks sequentially, without competing builds, using the same compiler,
flags and machine for before/after comparisons.

- `build/hopsy_sampling_benchmark`: coordinate/dense Hit-and-Run at dimensions
  10 and 100, target-dependent coordinate MH, Gaussian random walk, Dikin and
  CSmMALA. Five seeded batches of 100,000 transitions after warm-up; reports
  median ns/transition, throughput and final-state checksums.
- `build/hopsy_bound_sampling_benchmark`: the same workload through `BoundChain`.
- `build/hopsy_hit_and_run_benchmark`: direction, chord, materialization and MH
  component costs at dimensions 8, 64 and 256. Includes a fused in-place baseline,
  which has different rejection capabilities. Component timings are not additive.

Native benchmarks exclude output storage; they measure attempted transitions,
including rejections. Statistical correctness belongs to the retained tests;
throughput alone does not measure mixing or effective sample size.

For Python, select the built extension with `PYTHONPATH` and use an interpreter
with numpy installed:

```sh
PYTHONPATH=cpp/build-python/core/bindings python cpp/benchmarks/benchmark_bound_python.py --output /tmp/chains.json
PYTHONPATH=cpp/build-python/core/bindings python cpp/benchmarks/benchmark_bilinear.py --output /tmp/bilinear.json
```

These scripts exercise one and four threads with independent kernels and RNGs.
They include output storage and thread-pool setup. The bilinear benchmark includes
linear, empty-bilinear, disconnected and dense-cross constraints. Coordinate-only
sampling of disconnected sign components is excluded because it is reducible.
Use `--worker-local` to compare worker-local construction. CPU affinity is left
to the caller; no machine-specific CPU numbering is assumed.

### Native E. coli likelihood

```sh
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
    cpp/benchmarks/benchmark_ecoli_nrpt.py
```

This benchmark requires the `x3cflux` wheel with `make_hopsy_target()` and
`cpp/benchmark-python/Ecoli_NatProt_GOX.fml`. It keeps 100,000 cold samples
from standalone and NRPT runs. The default `--kernel hit-and-run` uses dense
random directions with uniform chord proposals. Use `--kernel csmmala` to
use CSmMALA with the capsule's native gradient. `--step-size` sets the
CSmMALA proposal scale for the standalone chain and every replica and
defaults to 0.15. Both modes use the same FluxML configuration and
free-parameter polytope. The default NRPT run explores eight replicas
concurrently, using one independent
x3cflux capsule per physical replica and taking one local step per slot
before exchanging. Use `--nrpt-mode both` to also time serial NRPT with the
same seeds and replica setup. Use `--betas` to supply a custom
inverse-temperature ladder in increasing order from 0 to 1. The JSON
report includes local transition counts, timed sampling, approximate minimum
coordinate ESS and ESS per second from batch means, and NRPT exchange counts
and round trips per second. Use `--local-steps` for positive-beta slots and
`--reference-steps` for beta zero. The report counts all attempted local
transitions. Each scan still retains one cold sample. Use `--num-samples` to
reduce the number of scans for pilot runs.
Set `--standalone-thinning` to the same value as `--local-steps` to time
both modes at matched local steps per retained draw.

Use `--round` to compute a rounding map with legacy hopsy. The native
sampler applies the map to every original constraint and returns draws in
the original FluxML parameter coordinates. Rounding is outside the timer.
For a tunable dense Hit-and-Run proposal, pass `--kernel gaussian-hit-and-run`
and `--step-size`. This scale is the standard deviation of the Gaussian
line displacement before truncation to the feasible chord. After rounding,
it is expressed in rounded coordinates. Uniform Hit-and-Run samples the
entire chord and has no step-size setting.

With `--kernel csmmala`, pass `--reference-kernel rounded-coordinate` for
random-coordinate Hit-and-Run at beta zero. The directions are
columns of the legacy hopsy rounding matrix and CSmMALA uses the original
coordinates at every other beta. `--reference-max-beta` extends this walk
through the chosen cutoff. At positive beta, proposals use Metropolis
acceptance for the tempered target. Use `--reference-kernel both` to compare
it with CSmMALA at every temperature and with one standalone CSmMALA chain.
All three modes can appear in the same posterior figures.

For a FluxML model with an unbounded feasible region, a uniform beta-zero
reference requires extra bounds. `--unbounded-flux-caps NET EXCHANGE` adds caps
to only the unbounded sides of the free fluxes. The same capped target is used
for all modes and the JSON lists every added bound. This changes the support of
the posterior.

Pass `--replica-trace path.npz` to save replica IDs and untempered log
densities after each exchange, with the starting ensemble in row zero.
With `--round`, the cached densities include the affine map's constant
log-volume term.
The JSON report then shows one-way endpoint crossings, the coldest slot
reached after a hot visit, immediate swap-backs, and local log-density
changes by slot. It also checks the exchange and round-trip counts against
the trace. Sampling time includes trace capture. Trace analysis and file
output are outside the timer.

Pass `--posterior-plots DIR` to write `net_flux_posteriors.svg` and
`exchange_flux_posteriors.svg` with Matplotlib. Each panel overlays the
retained, target-temperature draws for one flux using shared histogram bins
and density normalization. The plots include free and dependent fluxes.
The JSON report lists fluxes fixed by the model and their values. Flux
reconstruction and plotting are outside the sampling timer.

Use `--kernel both` to overlay Hit-and-Run and CSmMALA in the same figures.
The JSON report contains both benchmark runs. The optional NPZ uses
kernel-prefixed trace keys. This command saves both figures.

```sh
PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python \
    cpp/benchmarks/benchmark_ecoli_nrpt.py --kernel both --num-samples 10000 \
    --betas 0 0.005 0.02 0.06 0.16 0.36 0.65 1 \
    --posterior-plots cpp/build/ecoli-posteriors
```

### Replicated FluxML convergence comparison

`benchmark_flux_convergence.py run` runs one independently seeded replicate
of rounded standalone coordinate Hit-and-Run, rounded all-Hit-and-Run NRPT,
and mixed NRPT with rounded coordinate Hit-and-Run through β = 0.5. Each mode
uses the same physical starting point for that replicate. Use `--fit-weight`
to disperse starts between the strict interior and the FluxML initial point.
Pass `--workers` to set the worker budget for each NRPT job.
Pass `--method` with `standalone`, `nrpt_parallel`, or
`nrpt_parallel_rounded_reference` to time a single strategy in isolation.
`--mixed-only` also selects the mixed strategy. `--hit-and-run-steps 10` runs
ten transitions per mixed Hit-and-Run slot before communication. The CSmMALA
slots each take one step. Set `--reference-max-beta` to the desired cutoff.
For the NatProt ladder, `--reference-max-beta 0.65` leaves CSmMALA at β = 1
only.

`benchmark_flux_convergence.py summarize` takes the replicate directories for
one model. It uses ArviZ to compute rank-normalized split R-hat for free fluxes
and varying reconstructed fluxes, and repeats the free-flux check on the last
half of each chain. The overview prints the maximum free-flux R-hat, total
sampler time across replicates, and pooled bulk ESS. Total sampler time adds
the times of concurrent replicates. For batch wall time, time the complete
launch through the end of the summary commands. Per-chain bulk ESS is
reported for the whole retained chain and its first and last halves, with
thresholds of 100, 50 and 50. Thresholds are checked separately for every
free flux. A free flux passes only when all three windows meet their
thresholds in every chain. A trajectory with no within-chain variation
contributes zero ESS. Timings distinguish retained sampling, warm-up and
full replicate elapsed time.

The SVG histograms pool draws within each method to show the retained
empirical distributions. Sampling reports and convergence diagnostics are
kept separately for each method.

#### NatProt GOX on the Ryzen 9 5950X

Each strategy ran separately with four seeds (42, 136, 271, and 314), starting
at weights 0, 0.33, 0.67, and 0.995 between the interior and FluxML points.
Each cold chain kept 120,000 draws after 2,000 warm-up steps or scans. NRPT used
the ladder `[0, 0.005, 0.02, 0.06, 0.16, 0.36, 0.65, 1]` and CSmMALA
used step size 0.15. Every mixed slot above the listed cutoff took one
CSmMALA step per scan.

| Strategy | Sampling wall (s) | Total trips | Max. R̂ | Min. / mean bulk ESS | Min. / mean ESS rate (s⁻¹) |
|---|---:|---:|---:|---:|---:|
| Standalone rounded Hit-and-Run | 42.0 | — | 1.7908 | 5.9 / 3,331.7 | 0.141 / 79.243 |
| Rounded Hit-and-Run NRPT, 1 step per slot | 119.5 | 77 | 1.0503 | 53.6 / 7,382.8 | 0.448 / 61.778 |
| Mixed, 1 Hit-and-Run step at β ≤ 0.5 | 2,297.4 | 85 | 1.0155 | 271.6 / 2,098.6 | 0.118 / 0.913 |
| Mixed, 10 Hit-and-Run steps at β ≤ 0.65 | 2,332.6 | 714 | 1.0017 | 951.9 / 19,633.2 | 0.408 / 8.417 |
| Mixed, 10 Hit-and-Run steps at β ≤ 0.5 | 2,767.0 | 726 | 1.0016 | 1,049.9 / 6,851.6 | 0.379 / 2.476 |

The standalone replicates used one core each. For NRPT, four concurrent
replicates with four workers each beat two waves of eight-worker replicates
in every short pilot. Each replicate process was pinned to a disjoint group
of physical cores, with numerical libraries capped at one thread. Sampling
wall time spans the first launch through completion of all four replicates,
including setup, warm-up and writing draws. ESS rates use this wall time.
The full session took 8,118.5 s including pilots and diagnostics. Both
ten-step schedules met the ESS targets in every replicate for all 27 free
fluxes.

Historical comparison drivers, frozen-API branches, legacy patches, profiler
experiments and accumulated result files have been removed. Git retains their
history. The optional `python_build` CMake entry point builds current bindings
using an installed nanobind package.
