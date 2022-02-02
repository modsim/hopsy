import unittest
import pickle

from hopsy import *

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
        self.assertListEqual(model.components[0].mean.tolist(), new.components[0].mean.tolist())
        self.assertListEqual(model.components[0].covariance.tolist(), new.components[0].covariance.tolist())
        self.assertListEqual(model.components[0].inactives, new.components[0].inactives)

    def test_pymodel_pickling(self):
        model = PyModel("abc")
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

        model.mean = [1, 2, 3]
        with self.assertRaises(RuntimeError):
            model.compute_negative_log_likelihood([0, 0])

    def test_mixture_properties(self):
        model = Mixture([Gaussian()])
        model.components[0].mean = [1, 2]
        self.assertListEqual(model.components[0].mean.tolist(), [1, 2])

