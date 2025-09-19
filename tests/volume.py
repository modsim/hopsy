import unittest

import matplotlib.pyplot as plt
import numpy as np

from hopsy import *


class VolumeTest(unittest.TestCase):
    def test_get_next_variance(self):
        variance = 1
        n_dim = 2
        next_variance = get_next_variance(variance, n_dim)
        self.assertLess(variance, next_variance)

    def test_sample_gaussian(self):
        n_dim = 3
        p = Problem(*examples.generate_unit_simplex(n_dim))
        variance = 1
        n_samples = 100
        samples = sample_gaussian(
            p,
            center=np.zeros(n_dim),
            variance=variance,
            random_seed=0,
            n_samples=n_samples,
            n_procs=4,
        )
        # sample variance should be less than variance because gaussian is constrained by simplex
        self.assertLess(np.mean(np.var(samples)), variance)

    def test_estimate_log_ratio(self):
        n_dims = 2
        A, b = examples.generate_unit_hypercube(n_dims)
        b = np.ones(n_dims * 2)
        p = Problem(A, 100 * b)
        narrow_variance = 0.1
        wide_variance = get_next_variance(narrow_variance, n_dims)
        print("narrow, wide", narrow_variance, wide_variance)
        theoretical_log_ratio = 0.5 * n_dims * (np.log(wide_variance / narrow_variance))
        print("theory", theoretical_log_ratio)

        n_samples = 10000
        center = np.zeros(n_dims)
        samples = sample_gaussian(
            p,
            center=center,
            variance=narrow_variance,
            random_seed=0,
            n_samples=n_samples,
            n_procs=4,
        )
        log_ratio = estimate_log_ratio(
            samples,
            center,
            variance_diff=(+1.0 / narrow_variance - 1.0 / wide_variance),
            n_dims=n_dims,
        )[0]
        print("estimation", log_ratio)

        self.assertTrue(np.isclose(log_ratio, theoretical_log_ratio))

    def test_estimate_cube_volume(self):
        n_dims = 2
        A, b = examples.generate_unit_hypercube(n_dims)
        p = round(Problem(A, b))
        start_variance = 1e-6
        n_samples = 10000
        center = np.zeros(n_dims)
        volume = estimate_polytope_volume(
            p, compute_rounding=False, n_procs=4, sample_batch_size=n_samples
        )
        print(volume)
