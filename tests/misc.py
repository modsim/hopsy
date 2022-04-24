import unittest

import numpy

from hopsy import *

n_chains = 4
n_samples = 100
thinning = 10

seed = 0

problem = Problem([[1, 1]], [1], Gaussian(), starting_point = [0, 0])

ProposalTypes = [
            AdaptiveMetropolisProposal,
            BallWalkProposal,
            BilliardMALAProposal,
            CSmMALAProposal,
            DikinWalkProposal,
            GaussianCoordinateHitAndRunProposal,
            GaussianHitAndRunProposal,
            GaussianProposal,
            TruncatedGaussianProposal,
            UniformCoordinateHitAndRunProposal,
            UniformHitAndRunProposal,
        ]

class MiscTests(unittest.TestCase):
    def test_sequential_sampling(self):
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])


    #@unittest.expectedFailure
    def test_parallel_sampling(self):
        n_threads = 4

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning, n_threads)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

    def test_add_box_constraints(self):
        uniform_problem = Problem([[1, 1,]], [1])
        uniform_problem = add_box_constraints(uniform_problem, -2, 1)

        for ProposalType in ProposalTypes:
            if ProposalType in [BilliardMALAProposal, CSmMALAProposal, TruncatedGaussianProposal]:
                continue
            chain = MarkovChain(uniform_problem, ProposalType, starting_point=[.1, .1])

        gaussian_problem = Problem([[1, 1,]], [1], Gaussian())
        gaussian_problem = add_box_constraints(gaussian_problem, -1, 1)

        for ProposalType in ProposalTypes:
            chain = MarkovChain(gaussian_problem, ProposalType, starting_point=[.1, .1])


    def test_rounding(self):
        problem = Problem([[1, 1,]], [1])
        problem = add_box_constraints(problem, -2, 1)
        problem = round(problem)

        problem = Problem([[1, 1,]], [1], starting_point=[0, 0])
        problem = add_box_constraints(problem, -2, 1)
        problem = round(problem)


    def test_ess(self):
        states = [[[0, 1, 2, 3, 4]]*100]*4
        neff = ess(states)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        states = numpy.concatenate([[[[0, 1, 2, 3, 4]]*100]*4, numpy.random.rand(4, 100, 5)], axis=1)
        neff = ess(states, series=100)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        rel_ess = 1 / 400

        states = [[[0, 1, 2, 3, 4]]*100]*4
        neff = ess(states, relative=True)
        self.assertListEqual([rel_ess]*5, neff[0].tolist())

        states = numpy.concatenate([[[[0, 1, 2, 3, 4]]*100]*4, numpy.random.rand(4, 100, 5)], axis=1)
        neff = ess(states, series=100, relative=True)
        self.assertListEqual([rel_ess]*5, neff[0].tolist())

