# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(".version", "r", encoding="utf-8") as fh:
    version = fh.read().strip()

try:
    with open(".commit", "r", encoding="utf-8") as fh:
        commit = fh.read().strip()
except Exception:
    commit = "dirty"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # get_ext_fullpath returns the path to the folder where the .so/.pyd should go
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "").strip()
        with_mpi = os.environ.get("HOPS_MPI", "OFF")

        # Keep build_temp simple to ensure relative paths work correctly
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            # 1. Modern CMake (3.12+) - High priority
            f"-DPython_EXECUTABLE={sys.executable}",
            # 2. Legacy/Pybind11 fallback - All caps
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # 3. Force it to stay in the right place (don't go hunting for the wrong python!)
            "-DPython_FIND_STRATEGY=LOCATION",
            f"-DHOPSY_VERSION_INFO={self.distribution.get_version()}",
            f"-DHOPSY_BUILD_INFO={commit}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DHOPS_MPI={with_mpi}",
        ]
        build_args = ["--config", cfg] if os.name == "nt" else []

        # Handle Build ID
        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=ext.sourcedir
                )
                .decode("utf-8")
                .strip()
            )
            cmake_args.append(f"-DHOPSY_BUILD_ID={commit_hash}")
        except Exception:
            cmake_args.append(f"-DHOPSY_BUILD_ID={commit}")

        # Generator Logic
        ninja_available = shutil.which("ninja") is not None
        if not cmake_generator:
            if ninja_available:
                cmake_args += ["-G", "Ninja"]
            elif os.name != "nt":
                cmake_args += ["-G", "Unix Makefiles"]

        # Windows-specific Clang/MSVC logic (keeping your improvements)
        if os.name == "nt":
            if "Ninja" in (cmake_generator or ("Ninja" if ninja_available else "")):
                cmake_args += [
                    "-DCMAKE_C_COMPILER=clang-cl",
                    "-DCMAKE_CXX_COMPILER=clang-cl",
                ]
            elif "Visual Studio" in cmake_generator or not cmake_generator:
                cmake_args += ["-T", "ClangCL"]
                if self.plat_name in PLAT_TO_CMAKE:
                    cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j16"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


setup(
    name="hopsy",
    version=version,
    author="Richard D. Paul",
    author_email="r.paul@fz-juelich.de",
    description="A python interface for hops, the highly optimized polytope sampling toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://modsim.github.io/hopsy/",
        "Gitlab (Sources&CI)": "https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy",
        "GitHub (Mirror)": "https://github.com/modsim/hopsy/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    ext_modules=[CMakeExtension("hopsy/core")],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PolyRound>=0.4.0",
        "optlang>=1.7.0",
        "arviz",
        "numpy",
        "pandas",
        "tqdm",
        "scipy",
        "scikit-learn",
    ],
    zip_safe=False,
)
