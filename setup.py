# -*- coding: utf-8 -*-
import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(".version", "r") as fh:
    version = fh.read().split("\n")[0]

try:
    with open(".commit", "r") as fh:
        commit = fh.read().split("\n")[0]
except:
    commit = "dirty"


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary 'native' libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DHOPSY_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DHOPSY_BUILD_INFO={}".format(commit),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=self.build_temp
                )
                .decode("utf-8")
                .split("\n")[0]
            )
            print("Build id:", commit_hash)
            cmake_args.append("-DHOPSY_BUILD_ID={}".format(commit_hash))
        except Exception as e:
            commit_hash = commit
            print("ERROR retrieving commit hash. No build ID will be set.")

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                cmake_args += ["-GNinja"]

        else:

            # Single config generators are handled 'normally'
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        if os.name == "nt":  # os.name == nt is True for windows only
            # Use clang because MSVC is bad
            cmake_args += ["-T ClangCL"]

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
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
    ],
    ext_modules=[CMakeExtension("hopsy/core")],
    # ext_modules=[Extension('hopsy._hopsy', sources=['src/hopsy/hopsy.cpp'])],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PolyRound==0.2.10",
        "sympy<1.12.0",
        "arviz",
        "numpy<1.24.0",
        "mcbackend",
        "pandas",
        "tqdm",
        "matplotlib",
        "scikit-learn",
    ],
    zip_safe=False,
)
