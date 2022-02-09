import unittest

from hopsy import *

n_chains = 4
n_samples = 100
thinning = 10

seed = 0

problem = Problem([[1, 1]], [1], Gaussian(), starting_point = [0, 0])

class MiscTests(unittest.TestCase):
    def test_sequential_sampling(self):
        mcs = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(mcs, rngs, n_samples, thinning)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])


    #@unittest.expectedFailure
    def test_parallel_sampling(self):
        n_threads = 4

        mcs = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(mcs, rngs, n_samples, thinning, n_threads)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

