import unittest
import pickle

from hopsy import *

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

    #def test_initialization(self):
    #    problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0])

    #    for ProposalType in ProposalTypes:
    #        proposal = ProposalType(problem)
    #        mc = MarkovChain(proposal, problem)
    #        self.assertIsInstance(mc.proposal, ProposalType)

    @unittest.expectedFailure
    def test_empty_proposal_initialization(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0])

        for ProposalType in ProposalTypes:
            mc = MarkovChain(ProposalType(), problem)
            self.assertIsInstance(mc.proposal, ProposalType)

    def test_markovchain_pickling(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0])

        for ProposalType in ProposalTypes:
            print(ProposalType)
            proposal = ProposalType(problem)
            mc = MarkovChain(proposal, problem)
            dump = pickle.dumps(mc)
            new_mc = pickle.loads(dump)

