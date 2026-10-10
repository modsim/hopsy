"""Run with the built extension directory on PYTHONPATH."""

import _hopsy_core as h
import numpy as np

n = 3
geometry = h.LinearPolytope(np.vstack((np.eye(n), -np.eye(n))), np.ones(2 * n))
kernel = h.UniformCoordinateHitAndRunKernel(geometry, np.zeros(n), seed=42)
samples = h.draw(kernel, 10000, thinning=3)
print("shape:", samples.shape)
print("means:", samples.mean(axis=1))
print("second moments:", (samples**2).mean(axis=1))
