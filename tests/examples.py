import unittest

import numpy as np

from hopsy.examples import *


class TestGMGenerator(unittest.TestCase):
    def setUp(self):
        # Define the parameter sets for each test case
        self.test_cases = [
            {"dim": 10, "n_mix": 3, "n_nonident": 1},
            {"polytope_type": "spike", "dim": 10, "angle": 0.4, "n_mix": 10},
            {"polytope_type": "cone", "dim": 10, "angle": 0.4, "n_mix": 10},
            {"polytope_type": "diamond", "dim": 10, "angle": 0.4, "n_mix": 10},
        ]

    def test_gm_generator(self):
        for params in self.test_cases:
            with self.subTest(params=params):
                generator = GaussianMixtureGenerator(**params)
                problem = generator.get_problem()
                self.assertEqual(problem.A.shape[1], params["dim"])

                for i in range(params["n_mix"]):
                    self.assertTrue(np.all(np.diag(generator.covs[i]) > 0))
                    self.assertTrue(
                        np.allclose(
                            sorted(np.sqrt(np.diag(generator.covs[i]))), sorted(generator.scales[i]), atol=1e-6
                        )
                    )

                    n_nonident = params["n_nonident"] if "n_nonident" in params else 0

                    self.assertEqual(np.sum(np.diag(generator.covs[i]) >= (1e6 - 1e-6)), n_nonident)
