import unittest

from .. import *

class TuningTests(unittest.TestCase):

    def test_tuning_call(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0]); 
        mcs = [MarkovChain(GaussianProposal(problem), problem) for i in range(5)]; 

        ts = ThompsonSamplingTuning()
        target = ExpectedSquaredJumpDistanceTarget(mcs)

        # run twice to check that tune() doesnt break anything in target
        tune(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])
        tune(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(True, True)


    def test_target_setup(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0]); 
        mcs = [MarkovChain(GaussianProposal(problem), problem) for i in range(5)]; 

        ts = ThompsonSamplingTuning()
        target = ExpectedSquaredJumpDistanceTarget(mcs)
        tune(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        result = target([0], [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertLessEqual(result[0], 0.9)
        self.assertGreaterEqual(result[0], 0.75)
        self.assertLessEqual(result[1], 0.05)
        self.assertGreaterEqual(result[1], 0.005)

        tune(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(True, True)


    def test_target_setup(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0]); 
        mcs = [MarkovChain(GaussianProposal(problem), problem) for i in range(5)]; 

        ts = ThompsonSamplingTuning()
        target = AcceptanceRateTarget(mcs)
        tune(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        result = target([0], [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertLessEqual(result[0], 0.9)
        self.assertGreaterEqual(result[0], 0.75)
        self.assertLessEqual(result[1], 0.05)
        self.assertGreaterEqual(result[1], 0.005)

        tune(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(True, True)


