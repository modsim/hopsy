#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: $0 CPP_SOURCE_DIR VARIANT_OUTPUT_DIR" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cpp_dir=$(cd -- "$script_dir/../.." && pwd)
source_dir=$(realpath -- "$1")
mkdir -p -- "$2"
output=$(realpath -- "$2")
compiler=${CXX:-/usr/bin/g++}
python=${HOPSY_PERF_PYTHON:-"$cpp_dir/.benchmark-venv/bin/python"}
dependencies=${HOPSY_PERF_DEPS:-"$cpp_dir/build/_deps"}

cmake -S "$source_dir" -B "$output/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$compiler" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DHOPSY_BUILD_CPP_TESTS=OFF -DHOPSY_BUILD_PYTHON=ON \
  -DHOPSY_BUILD_PYTHON_TESTS=OFF -DHOPSY_INSTALL_CPP=OFF \
  "-DPython_EXECUTABLE=$python" "-Dnanobind_DIR=$("$python" -m nanobind --cmake_dir)" \
  "-DFETCHCONTENT_SOURCE_DIR_EIGEN3=$dependencies/eigen3-src" \
  "-DFETCHCONTENT_SOURCE_DIR_PCG_CPP=$dependencies/pcg_cpp-src" \
  > "$output/configure.log" 2>&1
cmake --build "$output/build" --parallel "${HOPSY_PERF_BUILD_JOBS:-4}" \
  > "$output/build.log" 2>&1
"$compiler" -std=c++20 -O3 -DNDEBUG -pthread \
  -I"$source_dir/include" -I"$dependencies/pcg_cpp-src/include" \
  -isystem "$dependencies/eigen3-src" "$script_dir/native.cpp" \
  -o "$output/native" > "$output/native-build.log" 2>&1
