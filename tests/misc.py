import unittest

import numpy
import numpy as np
import typing

from hopsy import *

n_chains = 4
n_samples = 100
thinning = 10

seed = 0

uniform = Problem([[1, 1]], [1], starting_point=[0, 0])
problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])

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

        chains = [MarkovChain(uniform, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

    def test_parallel_sampling(self):
        n_procs = 4

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning, n_procs)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

        chains = [MarkovChain(uniform, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning, n_procs)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

    def test_sampling_semigroup_property(self):
        # Test property for sequential chains
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states0 = sample(chains, rngs, int(n_samples // 2), thinning)
        _, states1 = sample(chains, rngs, int(n_samples // 2), thinning)

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states2 = sample(chains, rngs, n_samples, thinning)

        self.assertTrue(numpy.linalg.norm(states2[:, -1] - states1[:, -1]) < 1e-7)

        # Test property for parallel chains
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states0 = sample(chains, rngs, int(n_samples // 2), thinning, -1)
        _, states1 = sample(chains, rngs, int(n_samples // 2), thinning, -1)

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states2 = sample(chains, rngs, n_samples, thinning, -1)

        self.assertTrue(numpy.linalg.norm(states2[:, -1] - states1[:, -1]) < 1e-7)

    def test_add_box_constraints(self):
        uniform_problem = Problem([[1, 1, ]], [1])
        uniform_problem = add_box_constraints(uniform_problem, -2, 1)

        for ProposalType in ProposalTypes:
            if ProposalType in [BilliardMALAProposal, CSmMALAProposal, TruncatedGaussianProposal]:
                continue
            chain = MarkovChain(uniform_problem, ProposalType, starting_point=[.1, .1])

        gaussian_problem = Problem([[1, 1, ]], [1], Gaussian())
        gaussian_problem = add_box_constraints(gaussian_problem, -1, 1)

        for ProposalType in ProposalTypes:
            chain = MarkovChain(gaussian_problem, ProposalType, starting_point=[.1, .1])

    def test_rounding(self):
        problem = Problem([[1, 1, ]], [1])
        problem = add_box_constraints(problem, -2, 1)
        problem = round(problem)

        problem = Problem([[1, 1, ]], [1], starting_point=[0, 0])
        problem = add_box_constraints(problem, -2, 1)
        problem = round(problem)

        self.assertTrue((problem.b - problem.A @ problem.starting_point >= 0).all())

    def test_ess(self):
        states = [[[0, 1, 2, 3, 4]] * 100] * 4
        neff = ess(states)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        states = numpy.concatenate([[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1)
        neff = ess(states, series=100)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        states = numpy.concatenate([[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1)
        neff = ess(states, series=100, n_procs=-1)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        rel_ess = 1 / 400

        states = [[[0, 1, 2, 3, 4]] * 100] * 4
        neff = ess(states, relative=True)
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())

        states = numpy.concatenate([[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1)
        neff = ess(states, series=100, relative=True)
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())

        states = numpy.concatenate([[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1)
        neff = ess(states, series=100, relative=True, n_procs=-1)
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())

    def test_recording_meta_data(self):
        problem = Problem([[1, 0], [0, 1], [-1, 0], [0, -1]], [5, 5, 0, 0], Gaussian(dim=2))
        mcs = [MarkovChain(problem, proposal=GaussianHitAndRunProposal, starting_point=[.5, .5]) for i in range(2)]
        rngs = [RandomNumberGenerator(42, i) for i in range(2)]

        record_meta = ['state_negative_log_likelihood', 'proposal.proposal']
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))
        self.assertTrue(meta['proposal.proposal'].shape == (2, 100, 2))

        record_meta = ['state_negative_log_likelihood', 'proposal.proposal']
        meta, states = sample(mcs, rngs, n_samples=100, n_procs=2, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))
        self.assertTrue(meta['proposal.proposal'].shape == (2, 100, 2))

        record_meta = ['acceptance_rate']
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))

        record_meta = ['foo']  # obviously not an attribute
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))
        self.assertTrue(meta['foo'] is None)

        record_meta = False
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == 2)  # just usual acceptance rates
        self.assertTrue(type(meta) is list)  # just usual acceptance rates

    def test_backend(self):
        class TestBackend(Backend):
            def __init__(self, name: str = None):
                super(TestBackend, self).__init__()
                self.states = None
                self.state_idx = []
                self.meta = None

            def setup(self, n_chains: int, n_samples: int, n_dims: int, meta_names: typing.List[str]) -> None:
                super(TestBackend, self).setup(n_chains, n_samples, n_dims, meta_names)
                self.states = np.zeros((self.n_chains, self.n_samples, self.n_dims))
                self.state_idx = [0 for i in range(n_chains)]
                self.meta = [None for i in range(n_chains)]

            def record(self, chain_idx: int, state: numpy.ndarray, meta: typing.Dict[str, typing.Union[float, numpy.ndarray]]) -> None:
                self.states[chain_idx, self.state_idx[chain_idx]] = state
                self.state_idx[chain_idx] += 1
                if self.meta[chain_idx] is None:
                    self.meta[chain_idx] = {name: [meta[name]] for name in meta.keys()}
                else:
                    for name in meta.keys():
                        self.meta[chain_idx][name] += [meta[name]]

            def finish(self) -> None:
                for i in range(self.n_chain):
                    for field in self.meta:
                        self.meta[i][field] = numpy.array(self.meta[i][field])

        problem = Problem([[1, 0], [0, 1], [-1, 0], [0, -1]], [5, 5, 0, 0], Gaussian(dim=2))
        mcs = [MarkovChain(problem, proposal=GaussianHitAndRunProposal, starting_point=[.5, .5]) for i in range(2)]
        rngs = [RandomNumberGenerator(42, i) for i in range(2)]

        backend = TestBackend()
        record_meta = ['state_negative_log_likelihood', 'proposal.proposal']
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta, backend=backend)
        self.assertTrue(np.all(states == backend.states))
        self.assertTrue(meta.keys() == backend.meta[0].keys())
        self.assertTrue(np.all([np.all([meta[name][i] == backend.meta[i][name] for i in range(2)]) for name in meta.keys()]))

        backend = TestBackend()
        record_meta = False
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta, backend=backend)
        self.assertTrue(np.all(states == backend.states))
        self.assertTrue(np.all(meta == np.mean([backend.meta[i]["acceptance_rate"] for i in range(2)], axis=1)))
