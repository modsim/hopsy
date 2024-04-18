import time

import matplotlib.pyplot as plt
import numpy as np

import hopsy


class GaussianMixture:
    def __init__(self, mu1, mu2):
        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        self.model1 = hopsy.Gaussian(mean=mu1, covariance=cov)
        self.model2 = hopsy.Gaussian(mean=mu2, covariance=cov)

    def log_density(self, x):
        return np.log(
            np.exp(-self.model1.compute_negative_log_likelihood(x))
            + np.exp(-self.model2.compute_negative_log_likelihood(x))
        )


if __name__ == "__main__":
    replicates = 3
    n_temps = 4
    n_samples = 10_000
    thinning = 10

    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 1, 1])

    model = GaussianMixture(np.ones(2).reshape(2, 1), -np.ones(2).reshape(2, 1))
    problem = hopsy.Problem(A, b, model)

    sync_rngs = [hopsy.RandomNumberGenerator(seed=4321 + r) for r in range(replicates)]

    temperature_ladder = [1.0 - float(n) / (n_temps - 1) for n in range(n_temps)]

    mcs = [
        hopsy.MarkovChain(
            proposal=hopsy.UniformCoordinateHitAndRunProposal,
            problem=problem,
            starting_point=0.9 * np.ones(2),
        )
        for r in range(replicates)
    ]

    # Creates one parallel tempering ensemble for each replicate.
    # Each ensemble will have len(temperature_ladder) chains.
    chains = hopsy.create_py_parallel_tempering_ensembles(
        markov_chains=mcs,
        temperature_ladder=temperature_ladder,
        sync_rngs=sync_rngs,
        draws_per_exchange_attempt=50,
    )
    # chains = [m for m in mcs for beta in temperature_ladder]
    print("n chains", len(chains))

    rngs = [hopsy.RandomNumberGenerator(i + 1234) for i, _ in enumerate(chains)]

    start = time.perf_counter()
    acc_rate, samples = hopsy.sample(
        markov_chains=chains,
        rngs=rngs,
        n_samples=n_samples,
        thinning=thinning,
        n_procs=len(chains),
        # progress_bar=True,
    )
    end = time.perf_counter()
    print(
        f"sampling {n_samples} with thinning {thinning} and {len(chains)} chains took {end-start} seconds"
    )

    plt.figure()
    for i, chain in enumerate(chains):
        plt.hist(
            samples[i, :, 0], density=True, alpha=0.245, bins=100, label=f"chain {i}"
        )

    plt.legend()
    plt.show()
