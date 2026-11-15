#!/usr/bin/env python3
"""Independent analytic check: uniform on [0,1]^2 with x*y <= 0.2."""

import _hopsy_core as core
import numpy as np


def main():
    c = 0.2
    geometry = core.BilinearPolytope(
        np.vstack((np.eye(2), -np.eye(2))),
        np.array([1.0, 1.0, 0.0, 0.0]),
        split=1,
        cross_terms=[np.ones((1, 1))],
        bilinear_bounds=np.array([c]),
    )
    initial = np.array([0.1, 0.1])
    problem = core.Problem(geometry, core.UniformTarget())
    kernel = core.UniformHitAndRunKernel(initial)
    chain = core.bind_chain(problem, kernel, initial)
    rng = core.RandomNumberGenerator(seed=42)
    chain.draw(rng, 3000)
    samples = chain.draw(rng, 100000, thinning=2)
    Z = c * (1 - np.log(c))
    mean = (c - c * c / 2) / Z
    print("Exact area:", Z)
    print("Exact E[X]=E[Y]:", mean, "sample means:", samples.mean(axis=1))
    assert np.max(np.abs(samples.mean(axis=1) - mean)) < 0.012
    assert np.max(samples[0] * samples[1]) <= c + 1e-12
    for t in (0.05, 0.2, 0.5, 0.8):
        exact = t / Z if t <= c else (c + c * np.log(t / c)) / Z
        observed = np.mean(samples <= t, axis=1)
        print("CDF at", t, "exact:", exact, "observed:", observed)
        assert np.max(np.abs(observed - exact)) < 0.015


if __name__ == "__main__":
    main()
