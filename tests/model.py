import pickle
import unittest

import numpy as np

from hopsy import *
import hopsy

class ModelTests(unittest.TestCase):
    def test_gaussian_pickling(self):
        model = Gaussian()
        data = pickle.dumps(model)
        new = pickle.loads(data)
        self.assertListEqual(model.mean.tolist(), new.mean.tolist())
        self.assertListEqual(model.covariance.tolist(), new.covariance.tolist())
        self.assertListEqual(model.inactives, new.inactives)

    def test_mixture_pickling(self):
        model = Mixture([Gaussian()])
        data = pickle.dumps(model)
        new = pickle.loads(data)
        self.assertListEqual(model.weights, new.weights)
        self.assertListEqual(
            model.components[0].mean.tolist(), new.components[0].mean.tolist()
        )
        self.assertListEqual(
            model.components[0].covariance.tolist(),
            new.components[0].covariance.tolist(),
        )
        self.assertListEqual(model.components[0].inactives, new.components[0].inactives)

    def test_pymodel_pickling(self):
        model = hopsy.core.PyModel("abc")
        data = pickle.dumps(model)
        new = pickle.loads(data)
        self.assertEqual(model.model, new.model)

    def test_rosenbrock_pickling(self):
        model = Rosenbrock()
        data = pickle.dumps(model)
        new = pickle.loads(data)
        self.assertEqual(model.scale, new.scale)
        self.assertListEqual(model.shift.tolist(), new.shift.tolist())

    def test_gaussian_properties(self):
        model = Gaussian()
        model.mean = [1, 2]
        self.assertListEqual(model.mean.tolist(), [1, 2])

        with self.assertRaises(RuntimeError):
            model.mean = [1, 2, 3]

        model = Gaussian(dim=3)
        with self.assertRaises(RuntimeError):
            model.log_density([0, 0])

        self.assertIsNotNone(model.log_density(model.mean))
        self.assertTrue(
            np.all(
                np.isclose(
                    model.log_density(model.mean),
                    -model.compute_negative_log_likelihood(model.mean),
                )
            )
        )
        self.assertIsNotNone(model.log_gradient(model.mean))
        self.assertTrue(
            np.all(
                np.isclose(
                    model.log_gradient(model.mean),
                    model.compute_log_likelihood_gradient(model.mean),
                )
            )
        )
        self.assertIsNotNone(model.log_curvature(model.mean))
        self.assertTrue(
            np.all(
                np.isclose(
                    model.log_curvature(model.mean),
                    model.compute_expected_fisher_information(model.mean),
                )
            )
        )

    def test_mixture_properties(self):
        model = Mixture([Gaussian()])
        model.components[0].mean = [1, 2]
        self.assertListEqual(model.components[0].mean.tolist(), [1, 2])

        self.assertIsNotNone(
            model.compute_log_likelihood_gradient(model.components[0].mean)
        )
        self.assertIsNone(
            model.compute_expected_fisher_information(model.components[0].mean)
        )

    def test_implementing_model(self):
        class Uniform:
            def log_density(self, x):
                return 0

            def log_gradient(self, x):
                raise RuntimeError("Method not implemented.")

            def log_curvature(self, x):
                raise RuntimeError("Method not implemented.")

            def __copy__(self):
                return Uniform()

            def dimension_names(self):
                # implementation could also return empty list theoretically, but then RJMCMC becomes less useful.
                return ["x0", "x1"]

        model = Uniform()
        problem = Problem([[1, 1]], [1], model, starting_point=[0, 0])
        markovChain = MarkovChain(problem, GaussianProposal)
        state = markovChain.draw(RandomNumberGenerator())

    def test_implementing_derived_model(self):
        class Uniform:
            def log_density(self, x):
                return 0

            def log_gradient(self, x):
                raise RuntimeError("Method not implemented.")

            def log_curvature(self, x):
                raise RuntimeError("Method not implemented.")

            def __copy__(self):
                return Uniform()

            def dimension_names(self):
                # implementation could also return empty list theoretically, but then RJMCMC becomes less useful.
                return ["x0", "x1"]

        model = Uniform()
        problem = Problem([[1, 1]], [1], model, starting_point=[0, 0])
        markovChain = MarkovChain(problem, GaussianProposal)
        state = markovChain.draw(RandomNumberGenerator())

    def test_py_model(self):
        class Uniform:
            def log_density(self, x):
                return 0

        model = Uniform()
        problem = Problem([[1, 1]], [1], model, starting_point=[0, 0])
        markovChain = MarkovChain(problem, GaussianProposal)
        state = markovChain.draw(RandomNumberGenerator())

        # tests switching models works
        problem.model = Gaussian()
        problem.model = Uniform()
