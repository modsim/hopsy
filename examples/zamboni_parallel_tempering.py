import numpy as np
import pandas as pd
import x3cfluxpy
from mpi4py import MPI

import hopsy

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    model = x3cfluxpy.ILESimulator(
        "../../models/zamboni/unimodal-1,2-GLC/zamboni-unimodal-1,2-GLC.fml",
        sim_method="emu",
    )
    A = model.polytope_matrix
    b = model.polytope_bounds
    # problem = hopsy.round(hopsy.Problem(A, b, model=model, starting_point=model.last_params))
    problem = hopsy.Problem(A, b, model=model)
    starting_point = (
        0.9 * model.last_params
        + 0.1 * hopsy.compute_chebyshev_center(problem).flatten()
    )

    syncRng = hopsy.RandomNumberGenerator(42)
    num_procs = 1
    mc = [
        hopsy.MarkovChain(
            proposal=hopsy.DikinWalkProposal,
            problem=problem,
            parallelTemperingSyncRng=syncRng,
            exchangeAttemptProbability=0.15,
            starting_point=starting_point,
        )
    ]

    for m in mc:
        m.proposal.stepsize = 0.000025

    rng = [hopsy.RandomNumberGenerator(rank + 11, i) for i in range(num_procs)]

    acc, samples = hopsy.sample(
        markov_chains=mc, rngs=rng, n_samples=10000, thinning=50, n_procs=num_procs
    )

    print("ess", hopsy.ess(samples))
    print("acc", acc)

    samples = pd.DataFrame(data=samples[0], index=None, columns=model.param_names)
    samples.to_csv("zamboni_samples_" + str(rank) + ".csv")
