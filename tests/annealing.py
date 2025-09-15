import unittest

import numpy as np
import scipy

from hopsy import *

n_chains = 4
n_samples = 100
thinning = 10


class AnnealingTests(unittest.TestCase):
    def test_no_error_on_execution(self):
        problem = Problem(
            np.array([[1], [-1]]), np.array([3, 3]), Gaussian([0.0], [[1.0]])
        )

        mcs, rngs = setup(problem, proposal=hopsy.GaussianHitAndRunProposal, random_seed=42, n_chains=8, tempering=True)

        final_samples, result_stats = nrpt(
            mcs,
            rngs,
            n_rounds=5,
            thinning=1,
            progress_bar=True,
            seed=42,
        )

        self.assertTrue(True)

    def test_similar_wasserstein_to_single_chain(self):
        problem = Problem(
            np.array([[1], [-1]]), np.array([3, 3]), Gaussian([0.0], [[1.0]])
        )
        mcs, rngs = setup(problem, proposal=hopsy.GaussianHitAndRunProposal, random_seed=42, n_chains=8, tempering=True)

        samples_nrpt, result_stats = nrpt(
            mcs,
            rngs,
            n_rounds=5,
            thinning=1,
            progress_bar=True,
            seed=42,
        )

        samples_sc = sample(*setup(problem, 42, 1), 2**7)[1][0, :, 0]

        self.assertLess(scipy.stats.wasserstein_distance(samples_sc, samples_nrpt), 0.1)
