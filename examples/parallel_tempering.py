import numpy as np
import pandas as pd
from mpi4py import MPI

import hopsy


class MGaussian:
    def __init__(self, mu1, mu2):
        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        self.model1 = hopsy.Gaussian(mean=mu1, covariance=cov)
        self.model2 = hopsy.Gaussian(mean=mu2, covariance=cov)

    def compute_negative_log_likelihood(self, x):
        return -np.log(
            np.exp(-self.model1.compute_negative_log_likelihood(x))
            + np.exp(-self.model2.compute_negative_log_likelihood(x))
        )


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array(
        [
            1,
            1,
            1,
            1,
        ]
    )

    model = MGaussian(np.ones(2).reshape(2, 1), -np.ones(2).reshape(2, 1))
    problem = hopsy.Problem(A, b, model=model)

    syncRng = hopsy.RandomNumberGenerator(42)
    num_procs = 1
    mc = [
        hopsy.MarkovChain(
            proposal=hopsy.GaussianCoordinateHitAndRunProposal,
            problem=problem,
            parallelTemperingSyncRng=syncRng,
            exchangeAttemptProbability=0.15,
            starting_point=0.9 * np.ones(2),
        )
        for i in range(num_procs)
    ]

    for m in mc:
        m.proposal.stepsize = 0.25

    rng = [hopsy.RandomNumberGenerator(rank + 11, i) for i in range(num_procs)]

    acc, samples = hopsy.sample(
        markov_chains=mc, rngs=rng, n_samples=550000, thinning=1, n_procs=num_procs
    )

    print("ess", hopsy.ess(samples))
    print("acc", acc)

    samples = pd.DataFrame(data=samples[0], index=None, columns=["x0", "x1"])
    samples.to_csv("samples_" + str(rank) + ".csv")
