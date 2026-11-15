#!/usr/bin/env python3
"""Sample a correlated Gaussian restricted to a box using the new C++ core.

From the repository root, after building the nanobind extension:
    PYTHONPATH=cpp/build-python/core/bindings \
        cpp/.benchmark-venv/bin/python cpp/examples/gaussian.py
"""

import _hopsy_core as core
import numpy as np


def main():
    dimension = 2
    geometry = core.LinearPolytope(
        np.vstack((np.eye(dimension), -np.eye(dimension))),
        np.full(2 * dimension, 4.0),
    )
    target = core.GaussianTarget(
        np.zeros(dimension), np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    chain = core.GaussianRandomWalkKernel(
        geometry, initial_state=np.zeros(dimension), seed=42, step_size=0.7
    )  # step_size is a standard deviation in sampling coordinates.

    chain.draw(2_000, target=target)  # Discard warm-up samples.
    samples = chain.draw(20_000, thinning=3, target=target)

    # Each column is one sample. The box slightly truncates the Gaussian tails.
    print("Sample shape:", samples.shape)
    print("Mean (expected near zero):", samples.mean(axis=1))
    print("Covariance (approximately the target covariance):")
    print(np.cov(samples))


if __name__ == "__main__":
    main()
