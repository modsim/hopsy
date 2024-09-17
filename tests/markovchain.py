import pickle
import unittest

import numpy as np

from hopsy import *

ProposalTypes = [
    AdaptiveMetropolisProposal,
    BallWalkProposal,
    BilliardMALAProposal,
    BilliardWalkProposal,
    CSmMALAProposal,
    DikinWalkProposal,
    GaussianCoordinateHitAndRunProposal,
    GaussianHitAndRunProposal,
    GaussianProposal,
    TruncatedGaussianProposal,
    UniformCoordinateHitAndRunProposal,
    UniformHitAndRunProposal,
]


class MarkovChainTests(unittest.TestCase):
    def test_automatic_downcasting(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(problem, proposal)
            self.assertIsInstance(mc.proposal, ProposalType)

    def test_state_log_density(self):
        model = Gaussian()
        start = [0, 0]
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], model, starting_point=start
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(problem, proposal)
            self.assertAlmostEqual(
                mc.state_log_density, -mc.state_negative_log_likelihood
            )
            self.assertAlmostEqual(mc.state_log_density, model.log_density(start))
            self.assertAlmostEqual(
                mc.state_negative_log_likelihood, -model.log_density(start)
            )

    def test_empty_proposal_initialization(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            mc = MarkovChain(problem, ProposalType)
            self.assertIsInstance(mc.proposal, ProposalType)

    def test_markovchain_pickling(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(problem, proposal)
            dump = pickle.dumps(mc)
            new_mc = pickle.loads(dump)

            self.assertTrue(np.all(np.isclose(mc.state, new_mc.state)))
            mc.state = mc.state
            mc.state = new_mc.state
            self.assertTrue(np.all(np.isclose(mc.state, new_mc.state)))

    def test_markovchain_pickling_with_equality_constraints(self):
        problem = Problem([[1, 1, 0], [-1, 0, 0], [0, -1, 0]], [1, 0, 0])
        S = np.array([[2, 1, 0]])
        h = np.array([0.5])

        problem = add_box_constraints(problem, lower_bound=-10, upper_bound=10)
        problem = add_equality_constraints(problem, A_eq=S, b_eq=h)
        problem.starting_point = compute_chebyshev_center(problem)

        for ProposalType in ProposalTypes:
            try:
                proposal = ProposalType(problem)
                mc = MarkovChain(problem, proposal)
                dump = pickle.dumps(mc)
                new_mc = pickle.loads(dump)

                self.assertTrue(np.all(np.isclose(mc.state, new_mc.state)))
                # tests setting of state with equality constraints. This triggers a transform internally,
                # which introduced a bug in the past. Therefore, we test it specifically.
                mc.state = mc.state
                mc.state = new_mc.state
                new_mc.state = mc.state
                self.assertTrue(np.all(np.isclose(mc.state, new_mc.state)))
            except:
                # skip proposals that require the model
                pass

    def test_even_chains_parallel_tempering_markovchains_with_multiprocessing(self):
        replicates = 1
        n_temps = 4
        n_samples = 25_000
        thinning = 10
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.array([2, 2, 2, 2])

        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        mu1 = np.ones(2).reshape(2, 1)
        gauss1 = Gaussian(mean=mu1, covariance=cov)
        mu2 = -np.ones(2).reshape(2, 1)
        gauss2 = Gaussian(mean=mu2, covariance=cov)
        model = Mixture([gauss1, gauss2])

        problem = Problem(A, b, model)

        sync_rngs = [RandomNumberGenerator(seed=511 + r) for r in range(replicates)]
        temperature_ladder = [1.0 - float(n) / (n_temps - 1) for n in range(n_temps)]

        for ProposalType in ProposalTypes:
            if ProposalType == TruncatedGaussianProposal:
                continue
            mcs = [
                MarkovChain(
                    proposal=ProposalType,
                    problem=problem,
                    starting_point=1 * np.ones(2),
                )
                for r in range(replicates)
            ]

            mcs = create_py_parallel_tempering_ensembles(
                markov_chains=mcs,
                temperature_ladder=temperature_ladder,
                sync_rngs=sync_rngs,
                draws_per_exchange_attempt=20,
            )

            rngs = [RandomNumberGenerator(i + 511511) for i, _ in enumerate(mcs)]

            _, samples = sample(
                markov_chains=mcs,
                rngs=rngs,
                n_samples=n_samples,
                thinning=thinning,
                n_procs=len(mcs),
            )

            # mean should be 0 within 4 standard errors for every temp
            # if parallel tempering fails, mean is -5 or 5 with the coldest chain.
            for t in temperature_ladder:
                temp_samples = get_samples_with_temperature(
                    t, temperature_ladder, samples
                )
                expected_std_error = np.std(temp_samples) / np.sqrt(
                    np.min(ess(temp_samples))
                )
                assert np.abs(np.mean(temp_samples)) < 4 * expected_std_error

    def test_odd_chains_parallel_tempering_markovchains_with_multiprocessing(self):
        replicates = 1
        n_temps = 5
        n_samples = 25_000
        thinning = 10
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.array([2, 2, 2, 2])

        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        mu1 = np.ones(2).reshape(2, 1)
        gauss1 = Gaussian(mean=mu1, covariance=cov)
        mu2 = -np.ones(2).reshape(2, 1)
        gauss2 = Gaussian(mean=mu2, covariance=cov)
        model = Mixture([gauss1, gauss2])

        problem = Problem(A, b, model)

        sync_rngs = [RandomNumberGenerator(seed=511 + r) for r in range(replicates)]
        temperature_ladder = [1.0 - float(n) / (n_temps - 1) for n in range(n_temps)]

        for ProposalType in ProposalTypes:
            if ProposalType == TruncatedGaussianProposal:
                continue
            mcs = [
                MarkovChain(
                    proposal=ProposalType,
                    problem=problem,
                    starting_point=1 * np.ones(2),
                )
                for r in range(replicates)
            ]

            mcs = create_py_parallel_tempering_ensembles(
                markov_chains=mcs,
                temperature_ladder=temperature_ladder,
                sync_rngs=sync_rngs,
                draws_per_exchange_attempt=20,
            )

            rngs = [RandomNumberGenerator(i + 511511) for i, _ in enumerate(mcs)]

            _, samples = sample(
                markov_chains=mcs,
                rngs=rngs,
                n_samples=n_samples,
                thinning=thinning,
                n_procs=len(mcs),
            )

            # mean should be 0 within 4 standard errors for every temp
            # if parallel tempering fails, mean is -5 or 5 with the coldest chain.
            for t in temperature_ladder:
                temp_samples = get_samples_with_temperature(
                    t, temperature_ladder, samples
                )
                expected_std_error = np.std(temp_samples) / np.sqrt(
                    np.min(ess(temp_samples))
                )
                assert np.abs(np.mean(temp_samples)) < 4 * expected_std_error

    def test_even_chains_parallel_tempering_markovchains_with_rounding_and_multiprocessing(
        self,
    ):
        replicates = 1
        n_temps = 4
        n_samples = 25_000
        thinning = 10
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.array([2, 2, 2, 2])

        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        mu1 = np.ones(2).reshape(2, 1)
        gauss1 = Gaussian(mean=mu1, covariance=cov)
        mu2 = -np.ones(2).reshape(2, 1)
        gauss2 = Gaussian(mean=mu2, covariance=cov)
        model = Mixture([gauss1, gauss2])

        problem = round(Problem(A, b, model))

        sync_rngs = [RandomNumberGenerator(seed=511 + r) for r in range(replicates)]
        temperature_ladder = [1.0 - float(n) / (n_temps - 1) for n in range(n_temps)]

        for ProposalType in ProposalTypes:
            if ProposalType in [
                TruncatedGaussianProposal,
                CSmMALAProposal,
                BilliardMALAProposal,
            ]:
                continue

            mcs = [
                MarkovChain(
                    proposal=ProposalType,
                    problem=problem,
                    starting_point=1 * np.ones(2),
                )
                for r in range(replicates)
            ]

            mcs = create_py_parallel_tempering_ensembles(
                markov_chains=mcs,
                temperature_ladder=temperature_ladder,
                sync_rngs=sync_rngs,
                draws_per_exchange_attempt=20,
            )

            rngs = [RandomNumberGenerator(i + 511511) for i, _ in enumerate(mcs)]

            _, samples = sample(
                markov_chains=mcs,
                rngs=rngs,
                n_samples=n_samples,
                thinning=thinning,
                n_procs=len(mcs),
            )

            # mean should be 0 within 4 standard errors for every temp
            # if parallel tempering fails, mean is -5 or 5 with the coldest chain.
            for t in temperature_ladder:
                temp_samples = get_samples_with_temperature(
                    t, temperature_ladder, samples
                )
                expected_std_error = np.std(temp_samples) / np.sqrt(
                    np.min(ess(temp_samples))
                )
                assert np.abs(np.mean(temp_samples)) < 4 * expected_std_error

    def test_odd_chains_parallel_tempering_markovchains_with_rounding_and_multiprocessing(
        self,
    ):
        replicates = 1
        n_temps = 5
        n_samples = 25_000
        thinning = 10
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.array([2, 2, 2, 2])

        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        mu1 = np.ones(2).reshape(2, 1)
        gauss1 = Gaussian(mean=mu1, covariance=cov)
        mu2 = -np.ones(2).reshape(2, 1)
        gauss2 = Gaussian(mean=mu2, covariance=cov)
        model = Mixture([gauss1, gauss2])

        problem = round(Problem(A, b, model))

        sync_rngs = [RandomNumberGenerator(seed=511 + r) for r in range(replicates)]
        temperature_ladder = [1.0 - float(n) / (n_temps - 1) for n in range(n_temps)]

        for ProposalType in ProposalTypes:
            if ProposalType in [
                TruncatedGaussianProposal,
                CSmMALAProposal,
                BilliardMALAProposal,
            ]:
                continue
            mcs = [
                MarkovChain(
                    proposal=ProposalType,
                    problem=problem,
                    starting_point=1 * np.ones(2),
                )
                for r in range(replicates)
            ]

            mcs = create_py_parallel_tempering_ensembles(
                markov_chains=mcs,
                temperature_ladder=temperature_ladder,
                sync_rngs=sync_rngs,
                draws_per_exchange_attempt=20,
            )

            rngs = [RandomNumberGenerator(i + 511511) for i, _ in enumerate(mcs)]

            _, samples = sample(
                markov_chains=mcs,
                rngs=rngs,
                n_samples=n_samples,
                thinning=thinning,
                n_procs=len(mcs),
            )

            # mean should be 0 within 4 standard errors for every temp
            # if parallel tempering fails, mean is -5 or 5 with the coldest chain.
            for t in temperature_ladder:
                temp_samples = get_samples_with_temperature(
                    t, temperature_ladder, samples
                )
                expected_std_error = np.std(temp_samples) / np.sqrt(
                    np.min(ess(temp_samples))
                )
                assert np.abs(np.mean(temp_samples)) < 4 * expected_std_error
