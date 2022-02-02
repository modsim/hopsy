import unittest
import pickle

from hopsy import *

class RandomTests(unittest.TestCase):
    
    def test_randomnumbergenerator_pickling(self):
        rng = RandomNumberGenerator()
        dump = pickle.dumps(rng)
        new_rng = pickle.loads(dump)

        self.assertEqual(rng(), new_rng())

