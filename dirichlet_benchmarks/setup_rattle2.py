"""Build cpp_rattle_2 (algebraic squashing).

    python setup_rattle2.py build_ext --inplace

Must be run from the directory that Python imports from, i.e. the one that
cpp_rattle_2.__file__ points at.  Restart the kernel afterwards.

If Eigen is not found automatically, set EIGEN_INCLUDE.
"""

import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

MODULE_NAME = "cpp_rattle_2"
SOURCE = "cpp_rattle_2.cpp"


def find_eigen():
    candidates = []

    env = os.environ.get("EIGEN_INCLUDE")
    if env:
        candidates.append(env)

    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(os.path.join(conda, "include", "eigen3"))
        candidates.append(os.path.join(conda, "include"))

    candidates += [
        os.path.join(sys.prefix, "include", "eigen3"),
        os.path.join(sys.prefix, "include"),
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]

    for path in candidates:
        if os.path.isfile(os.path.join(path, "Eigen", "Dense")):
            return path

    raise SystemExit(
        "Eigen was not found. Set EIGEN_INCLUDE to the directory containing the "
        "'Eigen' folder.\nSearched:\n  " + "\n  ".join(candidates)
    )


ext_modules = [
    Pybind11Extension(
        MODULE_NAME,
        [SOURCE],
        include_dirs=[find_eigen()],
        cxx_std=14,
        extra_compile_args=["-O3"],
    )
]

setup(
    name=MODULE_NAME,
    version="0.1.0",
    description="RATTLE integrator, algebraic squashing, Gram term, reversibility",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
