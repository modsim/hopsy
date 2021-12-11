import unittest

from .. import *

ProposalTypes = [
            AdaptiveMetropolisProposal,
            BallWalkProposal,
            CSmMALAProposal,
            DikinWalkProposal,
            GaussianCoordinateHitAndRunProposal,
            GaussianHitAndRunProposal,
            GaussianProposal,
            UniformCoordinateHitAndRunProposal,
            UniformHitAndRunProposal,
        ]

class MarkovChainTests(unittest.TestCase):

    def test_automatic_downcasting(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0])

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(proposal, problem)
            self.assertIsInstance(mc.proposal, ProposalType)

