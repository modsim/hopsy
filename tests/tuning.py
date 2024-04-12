import unittest

import matplotlib.pyplot as plt
import numpy as np

from hopsy import *


class TuningTests(unittest.TestCase):
    def test_tuning_call(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        mcs = [MarkovChain(problem, GaussianProposal) for i in range(5)]

        ts = ThompsonSamplingTuning()
        target = ExpectedSquaredJumpDistanceTarget(mcs)

        # run twice to check that tune() doesnt break anything in target
        tune_old(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])
        tune_old(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(True, True)

    def test_esjd_target_setup(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        mcs = [MarkovChain(problem, GaussianProposal) for i in range(5)]

        ts = ThompsonSamplingTuning()
        target = ExpectedSquaredJumpDistanceTarget(mcs)
        tune_old(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        result = target([0], [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertLessEqual(result[0], 0.9)
        self.assertGreaterEqual(result[0], 0.75)
        self.assertLessEqual(result[1], 0.05)
        self.assertGreaterEqual(result[1], 0.005)

        tune_old(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

    def test_accrate_target_setup(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        mcs = [MarkovChain(problem, GaussianProposal) for i in range(5)]

        ts = ThompsonSamplingTuning()
        target = AcceptanceRateTarget(mcs, acceptance_rate=0.825)
        stepsize, posterior = tune_old(
            ts, target, [RandomNumberGenerator(0, i) for i in range(5)]
        )

        result = target(stepsize, [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(len(result), 2)

        tune_old(ts, target, [RandomNumberGenerator(0, i) for i in range(5)])

        self.assertEqual(True, True)

    def test_python_tuning(self):
        for target in ["accrate", "esjd", "esjd/s"]:
            A, b = examples.generate_unit_hypercube(10)
            problem = Problem(
                A, b, Gaussian(mean=np.zeros(10), covariance=0.001 * np.eye(10))
            )
            problem.starting_point = compute_chebyshev_center(problem)

            proposals = [
                GaussianProposal,
                GaussianHitAndRunProposal,
                BallWalkProposal,
                CSmMALAProposal,
                DikinWalkProposal,
            ]

            mcs = [
                [MarkovChain(problem, proposal=proposal) for i in range(4)]
                for proposal in proposals
            ]
            rngs = [
                [RandomNumberGenerator(42, 4 * j + i) for i in range(4)]
                for j, _ in enumerate(proposals)
            ]

            _, _, (gprs, domains) = tune(
                mcs, rngs, target=target, n_tuning=10000, n_rounds=5
            )

            proposal = GaussianProposal

            mcs = [MarkovChain(problem, proposal=proposal) for i in range(4)]
            rngs = [RandomNumberGenerator(42, i) for i in range(4)]

            _, _, (gprs, domains) = tune(
                mcs, rngs, target=target, n_tuning=10000, n_rounds=5
            )

            mcs = MarkovChain(problem, proposal=proposal)
            rngs = RandomNumberGenerator(42)

            _, _, (gprs, domains) = tune(
                mcs, rngs, target=target, n_tuning=10000, n_rounds=5
            )

        self.assertEqual(True, True)
