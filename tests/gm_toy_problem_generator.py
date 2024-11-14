import unittest
import numpy as np
from hopsy.examples import *





class TestGMToyProblemGenerator(unittest.TestCase):
    def setUp(self):
        # Define the parameter sets for each test case
        self.test_cases = [
            {"dim": 10, "n_modes": 3, "n_nonident": 1},
            {"polytope_type": "spike", "dim": 10, "angle": 0.4, "n_modes": 10},
            {"polytope_type": "cone", "dim": 10, "angle": 0.4, "n_modes": 10},
            {"polytope_type": "diamond", "dim": 10, "angle": 0.4, "n_modes": 10}
        ]

    def test_gm_toy_problem_generator(self):
        for params in self.test_cases:
            with self.subTest(params=params):
                generator = GaussianMixtureToyProblemGenerator(**params)
                problem = generator.generate_problem()
                self.assertEqual(problem.A.shape[1], params["dim"])

                for i in range(params["n_modes"]):
                    eigenvalues, eigenvectors = np.linalg.eig(generator.cov[i])
                    self.assertTrue(np.all(eigenvalues > 0))
                    self.assertTrue(np.allclose(sorted(eigenvalues), sorted(generator.scales[i]), atol=1e-6))
                    n_nonident = params["n_nonident"] if "n_nonident" in params else 0
                    self.assertEqual(np.sum(eigenvalues >= (1e6 - 1e-6)), n_nonident)