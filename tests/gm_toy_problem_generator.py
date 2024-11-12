import unittest

from hopsy.examples import *


class TestGMToyProblemGenerator(unittest.TestCase):
    def test_box(self):
        generator = GaussianMixtureToyProblemGenerator(n_modes=3, dim=10, n_nonident=1)
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_spike(self):
        generator = GaussianMixtureToyProblemGenerator(
            polytope_type="spike", dim=10, angle=0.4, n_modes=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_cone(self):
        generator = GaussianMixtureToyProblemGenerator(
            polytope_type="cone", dim=10, angle=0.4, n_modes=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_diamond(self):
        generator = GaussianMixtureToyProblemGenerator(
            polytope_type="diamond", dim=10, angle=0.4, n_modes=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_generate_gaussian_mixture_toy_problem(self):
        problem = generate_gaussian_mixture_toy_problem(n_modes=3, dim=10, n_nonident=1)
        self.assertEqual(problem.A.shape[1], 10)
