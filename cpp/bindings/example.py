"""Run with the built extension directory on PYTHONPATH."""

import _hopsy_core as h
import numpy as np

n = 3
geometry = h.LinearPolytope(np.vstack((np.eye(n), -np.eye(n))), np.ones(2 * n))
problem = h.Problem(geometry, h.UniformTarget())
initial_state = np.zeros(n)
kernel = h.UniformCoordinateHitAndRunKernel(initial_state)
chain = h.bind_chain(problem, kernel, initial_state)
rng = h.RandomNumberGenerator(seed=42)
samples = chain.draw(rng, 10000, thinning=3)
print("shape:", samples.shape)
print("means:", samples.mean(axis=1))
print("second moments:", (samples**2).mean(axis=1))
