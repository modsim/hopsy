#!/usr/bin/env python3
"""Build both extensions locally and copy the unmodified legacy Python facade.

Run with the isolated benchmark interpreter after installing declared runtime
requirements. Reuses Eigen and PCG sources from cpp/build; does not fetch sources,
install into src/, or overwrite the pre-existing legacy extension.
"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import nanobind
import pybind11


def main():
    cpp = Path(__file__).resolve().parents[1]
    root = cpp.parent
    eigen = cpp / "build/_deps/eigen3-src"
    pcg = cpp / "build/_deps/pcg_cpp-src"
    if not eigen.is_dir() or not pcg.is_dir():
        raise RuntimeError("configure the existing cpp build dependencies first")
    legacy = cpp / "build-legacy"
    new = cpp / "build-python"

    def run(*args):
        subprocess.run([str(a) for a in args], check=True)

    run(
        "cmake",
        "-S",
        root,
        "-B",
        legacy,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        f"-DFETCHCONTENT_SOURCE_DIR_EIGEN3={eigen}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DHOPS_DOCS=OFF",
        "-DHOPSY_BUILD_INFO=clipping-benchmark",
        "-DHOPSY_VERSION_INFO=experimental",
    )
    run("cmake", "--build", legacy, "--target", "core", "-j2")
    native_package = Path(nanobind.__file__).parent
    run(
        "cmake",
        "-S",
        cpp / "benchmarks/python_build",
        "-B",
        new,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        f"-DFETCHCONTENT_SOURCE_DIR_EIGEN3={eigen}",
        f"-DFETCHCONTENT_SOURCE_DIR_PCG_CPP={pcg}",
        f"-DFETCHCONTENT_SOURCE_DIR_NANOBIND={native_package}",
        f'-Dnanobind_DIR={native_package / "cmake"}',
        f"-DPython_EXECUTABLE={sys.executable}",
    )
    run("cmake", "--build", new, "--target", "_hopsy_core", "-j2")
    package = cpp / "benchmark-python/hopsy"
    package.mkdir(parents=True, exist_ok=True)
    for source in (root / "src/hopsy").glob("*.py"):
        shutil.copy2(source, package / source.name)
    extension = "core" + sysconfig.get_config_var("EXT_SUFFIX")
    shutil.copy2(legacy / extension, package / extension)
    print(f'PYTHONPATH={package.parent}:{new / "core/bindings"}')


if __name__ == "__main__":
    main()
