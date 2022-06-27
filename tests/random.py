import unittest
import pickle

from hopsy import *

class RandomTests(unittest.TestCase):
    
    def test_randomnumbergenerator_pickling(self):
        rng = RandomNumberGenerator()
        # calls rng once to advance it
        rng()
        dump = pickle.dumps(rng)
        new_rng = pickle.loads(dump)

        self.assertEqual(rng(), new_rng())

