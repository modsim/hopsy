import pickle
import typing
import unittest

import numpy
import numpy as np

from hopsy import *

if __name__ == "__main__":
    replicates = 3
    n_temps = 4

    problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])

    sync_rngs = [RandomNumberGenerator(seed=1991 + r) for r in range(replicates)]

    temperature_ladder = [1.0 - float(n) / (n_temps - 1) for n in range(n_temps)]

    mcs = [
        MarkovChain(
            proposal=UniformCoordinateHitAndRunProposal,
            problem=problem,
            starting_point=np.zeros(2),
        )
        for r in range(replicates)
    ]

    # Creates one parallel tempering ensemble for each replicate.
    # Each ensemble will have len(temperature_ladder) chains.
    chains = create_py_parallel_tempering_ensembles(
        markov_chains=mcs,
        temperature_ladder=temperature_ladder,
        sync_rngs=sync_rngs,
        draws_per_exchange_attempt=100,
    )

    # dump = pickle.dumps(chains)
    # del chains
    # loaded = pickle.loads(dump)

    print("loaded")
    # while True:
    #     pass
