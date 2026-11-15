#!/usr/bin/env python3
"""Uniform sampling in [-1,1]^2 with x*y < 0 (using its measure-equivalent closure).

PYTHONPATH=cpp/build-python/core/bindings cpp/.benchmark-venv/bin/python cpp/examples/bilinear.py

Dense directions and the complete feasible slice let the chain jump between
quadrants II and IV. Coordinate directions alone cannot mix these components.
"""

from pathlib import Path

import _hopsy_core as core
import matplotlib.pyplot as plt
import numpy as np


def main():
    geometry = core.BilinearPolytope(
        np.vstack((np.eye(2), -np.eye(2))),
        np.ones(4),
        split=1,
        cross_terms=[np.ones((1, 1))],
        bilinear_bounds=np.zeros(1),
    )
    problem = core.Problem(geometry, core.UniformTarget())
    initial = np.array([-0.5, 0.5])
    kernel = core.UniformHitAndRunKernel(initial)
    chain = core.bind_chain(problem, kernel, initial)
    rng = core.RandomNumberGenerator(seed=42)
    chain.draw(rng, 2000)
    samples = chain.draw(rng, 20000, thinning=2)
    assert np.all(samples[0] * samples[1] < 0)
    print("Mean (exact: [0,0]):", samples.mean(axis=1))
    print("E[XY] (exact: -1/4):", np.mean(samples[0] * samples[1]))
    print("Fraction with X>0 (exact: 1/2):", np.mean(samples[0] > 0))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(samples[0], samples[1], s=1, alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set(
        xlabel="x",
        ylabel="y",
        title="Uniform on the two regions x*y < 0",
        xlim=(-1, 1),
        ylim=(-1, 1),
        aspect="equal",
    )
    output = Path(__file__).resolve().parents[1] / "build/bilinear.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print("Plot:", output)


if __name__ == "__main__":
    main()
