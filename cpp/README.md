# hopsy C++ core

hopsy 2.0 is a header-only C++20 library for constrained Markov chain Monte Carlo. The optional nanobind module exposes the same samplers and a native target capsule interface to Python.

## How non-reversible parallel tempering works

For a bounded, positive-volume feasible region (P), slot β targets a density proportional to `1_P(x) exp(β ℓ(x))`, where ℓ is the entire supplied target log density. The hottest slot at β = 0 is uniform on (P), even where the supplied target gives zero mass. The cold slot at β = 1 targets the requested posterior. Intermediate slots use strictly increasing β. A constant-volume affine coordinate map preserves these semantics.

Each scan explores every slot, saves one cold draw, then offers disjoint adjacent swaps. With zero-based slots the first scan offers edges 1, 3, … and the second offers edges 0, 2, …. The log acceptance ratio for slots (i, j) is `(β[j] - β[i]) (ℓ(x[i]) - ℓ(x[j]))`, capped at zero. Each local transition and each matching of Metropolis swaps preserves the product of the slot target distributions. The deterministic composition is generally non-reversible.

Positive β always scales the log density, even when β is close to zero. The supplied target may return negative infinity to exclude a point from slots with β > 0. Initial states at positive β must have finite log densities, and target evaluations of NaN or positive infinity are errors.

Slots hold temperatures and point to physical replicas. A replica retains its state, RNG stream, kernel workspaces, native target context and untempered density cache when it changes slots. Temperature-dependent caches are invalidated on a β change. A round trip is a completed hot → cold → hot journey by one physical replica within the current diagnostics window. Warm-up diagnostics can inform a new ladder through `TemperatureLadder::adapted`. Fix the ladder for retained sampling.

Hit-and-Run combines random dense, random coordinate or cycling coordinate directions with uniform, Gaussian or overrelaxed line proposals. The Gaussian kernel family includes Gaussian random walk, MALA, Dikin walk and CSmMALA. The mixed FluxML sampler uses rounded random-coordinate Hit-and-Run at or below a chosen β and CSmMALA in the model coordinates above it. Columns of the rounding matrix give equivalent chord directions in model coordinates, so swaps need no coordinate conversion. Switching walks rebuilds the incoming geometry workspace from the current state. Positive-beta Hit-and-Run still uses tempered Metropolis acceptance.

| Source | Responsibility |
|---|---|
| [non_reversible_parallel_tempering.h](include/hopsy/sampling/non_reversible_parallel_tempering.h) | Replica ownership, local exploration, swaps and trip counters |
| [temperature_ladder.h](include/hopsy/sampling/temperature_ladder.h) | Ladder validation and warm-up adaptation |
| [basis_reference.h](include/hopsy/sampling/basis_reference.h), [log_density_cache.h](include/hopsy/target/log_density_cache.h) | Rounded chord directions and cached accepted and proposed densities |
| [tempering.cpp](bindings/tempering.cpp), [tempering.h](bindings/tempering.h) | Python API, native worker assignment and the per-scan exchange barrier |
| [benchmark_ecoli_nrpt.py](benchmarks/benchmark_ecoli_nrpt.py), [benchmark_flux_convergence.py](benchmarks/benchmark_flux_convergence.py), [flux_benchmark_common.py](benchmarks/flux_benchmark_common.py) | Single-seed comparisons, replicated convergence diagnostics, and shared model and trace utilities |

See the [complete C++ example](examples/non_reversible_parallel_tempering.cpp) and the [Python, C++ and capsule API guide](bindings/README.md).

### Chain and geometry contracts

Each chain needs its own mutable kernel, workspace and RNG stream. Geometry and targets can be shared when their const operations support concurrent use. Borrowed C++ objects must outlive the chain. A pending proposal reserves its workspace until acceptance or rejection. The bound-chain API manages this lifecycle and exposes the last evaluated accepted and proposed log densities.

The bilinear geometry supports uniform and Gaussian-displacement Hit-and-Run over unions of intervals. The Gaussian family and overrelaxed line kernel require linear geometry. Bilinear slices use numerical filters with an exact fallback for ambiguous boundaries. Use binary round-to-nearest arithmetic without fast-math.

## Build, install and test

Use CMake 3.26 or newer, Git, a C++20 compiler and standard library with thread support, and a CMake build tool such as Make or Ninja. The build looks for Boost 1.89 and Eigen 5.0 or newer and fetches them if needed. It fetches PCG and uses Catch2 3.8.1 or newer for native tests. A first build without these dependencies needs network access.

From the repository root, build and install the C++ headers and CMake package into a local prefix.

```sh
cmake -S cpp -B cpp/build/native -DCMAKE_BUILD_TYPE=Release \
  -DHOPSY_BUILD_CPP_TESTS=ON
cmake --build cpp/build/native --parallel 4
ctest --test-dir cpp/build/native --output-on-failure
cmake --install cpp/build/native --prefix "$PWD/cpp/build/native-install"
```

A downstream CMake project uses `find_package(hopsy CONFIG REQUIRED)` and links `hopsy::core`. Pass the installation prefix in `CMAKE_PREFIX_PATH` along with any nonstandard Boost and Eigen installation prefixes. The installed package needs those dependencies' CMake packages. Run just the Dirichlet stationary-distribution checks with `ctest --test-dir cpp/build/native -L '^dirichlet$' --output-on-failure`.

### Python bindings

Use Python 3.10 or newer with its development headers. Use the same interpreter for CMake, pip and the Python tests. The handover was tested with Python 3.14 and nanobind 3.1.0.

```sh
python3 -m venv cpp/.benchmark-venv
hopsy_python="$PWD/cpp/.benchmark-venv/bin/python"
"$hopsy_python" -m pip install nanobind pytest numpy
cmake -S cpp -B cpp/build/python -DCMAKE_BUILD_TYPE=Release \
  -DHOPSY_BUILD_CPP_TESTS=OFF -DHOPSY_INSTALL_CPP=OFF \
  -DHOPSY_BUILD_PYTHON=ON -DHOPSY_BUILD_PYTHON_TESTS=ON \
  -DPython_EXECUTABLE="$hopsy_python" \
  -Dnanobind_DIR="$("$hopsy_python" -m nanobind --cmake_dir)"
cmake --build cpp/build/python --parallel 4
ctest --test-dir cpp/build/python --output-on-failure
cmake --install cpp/build/python --prefix "$PWD/cpp/build/python-install"
PYTHONPATH="$PWD/cpp/build/python-install" "$hopsy_python" -c 'import hopsy; print(hopsy.TemperatureLadder)'
```

## Benchmarks

See [Non-reversible parallel tempering benchmarks](benchmarks/README.md#non-reversible-parallel-tempering) for native build and run commands, 13CFLUX setup, model selection, smoke tests and measured results.
