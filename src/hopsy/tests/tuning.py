import unittest

from .. import *

class TuningTests(unittest.TestCase):

    def test_tuning_call(self):
        problem = hopsy.Problem([[1, 1]], [1], hopsy.Gaussian(), starting_point=[0, 0]); 
        mcs = [hopsy.MarkovChain(hopsy.GaussianProposal(problem), problem) for i in range(5)]; 

        ts = hopsy.ThompsonSamplingTuning()
        target = hopsy.ExpectedSquaredJumpDistanceTarget(mcs)

        # run twice to check that tune() doesnt break anything in target
        hopsy.tune(ts, target, [hopsy.RandomNumberGenerator(0, i) for i in range(5)])
        hopsy.tune(ts, target, [hopsy.RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(True, True)


    def test_target_setup(self):
        problem = hopsy.Problem([[1, 1]], [1], hopsy.Gaussian(), starting_point=[0, 0]); 
        mcs = [hopsy.MarkovChain(hopsy.GaussianProposal(problem), problem) for i in range(5)]; 

        ts = hopsy.ThompsonSamplingTuning()
        target = hopsy.ExpectedSquaredJumpDistanceTarget(markov_chains=mcs)
        hopsy.tune(ts, target, [hopsy.RandomNumberGenerator(0, i) for i in range(5)])
        hopsy.tune(ts, target, [hopsy.RandomNumberGenerator(0, i) for i in range(5)])

        self.assertListEqual(target([0], [hopsy.RandomNumberGenerator(0, i) for i in range(5)]), (0.8263665206373976, 0.005821290885441588))


