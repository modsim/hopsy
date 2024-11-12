import unittest

from hopsy.examples import *


class TestGaussianMixtureGenerator(unittest.TestCase):
    def test_box(self):
        generator = GaussianMixtureGenerator(n_mix=3, dim=10, n_nonident=1)
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_spike(self):
        generator = GaussianMixtureGenerator(
            polytope_type="spike", dim=10, angle=0.4, n_mix=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_cone(self):
        generator = GaussianMixtureGenerator(
            polytope_type="cone", dim=10, angle=0.4, n_mix=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_diamond(self):
        generator = GaussianMixtureGenerator(
            polytope_type="diamond", dim=10, angle=0.4, n_mix=10
        )
        problem = generator.generate_problem()
        self.assertEqual(generator.dim, 10)

    def test_generate_gaussian_mixture(self):
        problem = generate_gaussian_mixture(n_mix=3, dim=10, n_nonident=1)
        self.assertEqual(problem.A.shape[1], 10)
