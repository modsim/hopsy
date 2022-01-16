import unittest
import pickle

from .. import *

class ModelTests(unittest.TestCase):
    def test_gaussian_pickling(self):
        model = Gaussian()
        data = pickle.dumps(model)
        new = pickle.loads(data)
        self.assertListEqual(model.mean.tolist(), new.mean.tolist())
        self.assertListEqual(model.covariance.tolist(), new.covariance.tolist())
        self.assertListEqual(model.inactives, new.inactives)

    def test_gaussian_properties(self):
        model = Gaussian()
        model.mean = [1, 2]
        self.assertListEqual(model.mean.tolist(), [1, 2])

        model.mean = [1, 2, 3]
        with self.assertRaises(RuntimeError):
            model.compute_negative_log_likelihood([0, 0])

