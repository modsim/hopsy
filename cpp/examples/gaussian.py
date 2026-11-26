#!/usr/bin/env python3
"""Sample a correlated Gaussian restricted to a box using the new C++ core.

From the repository root, after building the nanobind extension:
    PYTHONPATH=cpp/build/python/bindings \
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
    problem = core.Problem(geometry, target)
    initial_state = np.zeros(dimension)
    kernel = core.GaussianRandomWalkKernel(initial_state, step_size=0.7)
    chain = core.bind_chain(problem, kernel, initial_state)
    rng = core.RandomNumberGenerator(seed=42)

    chain.draw(rng, 2_000)  # Discard warm-up samples.
    samples = chain.draw(rng, 20_000, thinning=3)

    # Each column is one sample. The box slightly truncates the Gaussian tails.
    print("Sample shape:", samples.shape)
    print("Mean (expected near zero):", samples.mean(axis=1))
    print("Covariance (approximately the target covariance):")
    print(np.cov(samples))


if __name__ == "__main__":
    main()
