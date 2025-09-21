import unittest

import matplotlib.pyplot as plt
from scipy.special import gammaln
import numpy as np

from hopsy import *


class VolumeTest(unittest.TestCase):
    def test_get_next_variance(self):
        variance = 1
        n_dim = 2
        next_variance = get_next_variance(variance, n_dim)
        self.assertLess(variance, next_variance)

    def test_sample_constrained_gaussian(self):
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
        def test_n_dims(n_dims, variance, n_samples=10000):
            A, _ = examples.generate_unit_hypercube(n_dims)
            b = 100000 * variance * np.ones(A.shape[0])
            p = Problem(A, b)
            narrow_variance = variance
            wide_variance = get_next_variance(narrow_variance, n_dims)
            theoretical_log_ratio = 0.5 * n_dims * (np.log(wide_variance / narrow_variance))
            center = np.zeros(n_dims)
            samples = sample_gaussian(
                p,
                center=center,
                variance=narrow_variance,
                random_seed=0,
                n_samples=n_samples,
                n_procs=4,
            )
            log_ratio, mc_error = estimate_log_ratio(
                samples,
                center,
                variance_im1=narrow_variance,
                variance_i=wide_variance,
            )
            # check that the results are within 3 sigma
            self.assertTrue(np.abs(log_ratio - theoretical_log_ratio) < mc_error * 3)

        test_n_dims(2, 1)
        test_n_dims(2, 2)
        test_n_dims(2, 0.5)
        test_n_dims(3, 1)
        test_n_dims(3, 2)
        test_n_dims(3, 0.5)
        test_n_dims(4, 1)
        test_n_dims(4, 2)
        test_n_dims(4, 0.5)
        test_n_dims(10, 1)
        test_n_dims(10, 2)
        test_n_dims(10, 0.5)
        test_n_dims(15, 1)
        test_n_dims(15, 2)
        test_n_dims(15, 0.5)

    def test_estimate_cube_volume(self):
        def estimate_cube_volume(n_dims, n_samples):
            A, b = examples.generate_unit_hypercube(n_dims)
            p = Problem(A, b)
            log_volume, log_volume_error = estimate_polytope_log_volume(
                p, n_procs=16, sample_batch_size=n_samples, max_iterations=5 * n_dims,
            )
            log_theoretical = 0
            self.assertLess(np.abs(log_volume - log_theoretical), 3 * log_volume_error)

        estimate_cube_volume(15, 1000)

    def test_estimate_simplex_volume(self):
        def estimate_simplex_volume(n_dims, n_samples):
            A, b = examples.generate_unit_simplex(n_dims)
            p = Problem(A, b)
            log_volume, log_volume_error = estimate_polytope_log_volume(
                p, n_procs=16, sample_batch_size=n_samples, max_iterations=5 * n_dims,
            )
            log_theoretical = -gammaln(n_dims + 1)
            self.assertLess(np.abs(log_volume - log_theoretical), 3 * log_volume_error)

        estimate_simplex_volume(15, 1000)
