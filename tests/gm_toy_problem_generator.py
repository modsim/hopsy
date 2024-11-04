import unittest

from hopsy.examples import *


class TestGMToyProblemGenerator(unittest.TestCase):
    def test_random_gmm_on_box(self):
        generator = GaussianMixtureToyProblemGenerator(n_modes=3, dim=10, n_nonident=1)
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_spike_gmm(self):
        generator = GaussianMixtureToyProblemGenerator(
            polytope_type="spike", dim=10, angle=0.1, n_modes=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)
