import unittest

from .. import *

class ModelTests(unittest.TestCase):
    def test_gaussian_pickling(self):
        model = Gaussian()
        print(model)
        print(model.__getstate__())

