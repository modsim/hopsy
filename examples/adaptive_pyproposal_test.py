import sys

import numpy as np

import hopsy


class GaussianProposal:
    def __init__(
        self, A: np.ndarray, b: np.ndarray, state: np.ndarray, cov: np.ndarray
    ):
        self.A = A
        self.b = b
        self.cov = cov
        self.stepsize = 1
        self.proposal = state.flatten()
        self.state = state.flatten()

    def propose(self, rng):
        mean = np.zeros((len(self.cov),))
        y = np.random.multivariate_normal(mean, self.cov).reshape(-1, 1).flatten()
        self.proposal = self.state + self.stepsize * y
        return self.proposal

    def log_acceptance_probability(self) -> float:
        if ((self.A @ self.proposal - self.b) >= 0).any():
            return -np.inf
        return 0


class AdaptiveGaussianProposal:
    def __init__(self, A: np.ndarray, b: np.ndarray, state: np.ndarray, eps=0.001):
        self.A = A
        self.b = b
        self.state = state.flatten()
        self.eps = eps
        self.stepsize = 1
        self.proposal = state.flatten()
        self.t = 0
        self.cov = np.identity(len(self.state))
        self.mean = np.zeros(self.state.shape)

    def propose(self, rng):
        new_mean = (self.t * self.mean + self.state) / (self.t + 1)
        self.cov = (
            (
                (self.t - 1) * self.cov
                + self.t * np.outer(self.mean, self.mean)
                - (self.t + 1) * np.outer(new_mean, new_mean)
                + np.outer(self.state, self.state)
                + self.eps * np.identity(len(self.state))
            )
            / self.t
            if self.t > 0
            else self.cov
        )
        self.t += 1

        proposal_mean = np.zeros((len(self.cov),))
        y = (
            np.random.multivariate_normal(proposal_mean, self.cov)
            .reshape(-1, 1)
            .flatten()
        )
        self.proposal = self.state + self.stepsize * y
        return self.proposal

    def log_acceptance_probability(self) -> float:
        if ((self.A @ self.proposal - self.b) >= 0).any():
            return -np.inf
        return 0


if __name__ == "__main__":
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([[1], [1], [0], [0]])

    x0 = np.array([[0.1], [0.1]])

    mu = np.zeros((2, 1))
    cov = 0.1 * np.identity(2)

    gaussian_proposal = GaussianProposal(A, b, x0, np.identity(2))
    adaptive_proposal = AdaptiveGaussianProposal(A, b, x0)

    model = hopsy.Gaussian(mu, cov)
    problem = hopsy.Problem(A, b, model)

    gaussian_run = hopsy.MarkovChain(problem, starting_point=x0)
    gaussian_run.proposal = gaussian_proposal
    adaptive_run = hopsy.MarkovChain(problem, starting_point=x0)
    adaptive_run.proposal = adaptive_proposal

    gaussian_run.proposal.stepsize = 0.9
    adaptive_run.proposal.stepsize = 0.9

    gaussian_stepsize = gaussian_run.proposal.stepsize
    adaptive_stepsize = adaptive_run.proposal.stepsize

    gaussian_acc_rate, gsamples = hopsy.sample(
        gaussian_run, hopsy.RandomNumberGenerator(42), n_samples=10000
    )
    adaptive_acc_rate, asamples = hopsy.sample(
        adaptive_run, hopsy.RandomNumberGenerator(42), n_samples=10000
    )

    gaussian_ess = np.min(hopsy.ess(gsamples))
    adaptive_ess = np.min(hopsy.ess(asamples))

    if len(sys.argv) == 1 or sys.argv[1] != "test":
        print("         | Gaussian proposal" + " | Adaptive proposal")
        print("---------+------------------" + "-+------------------")
        print(
            "Stepsize |                "
            + str(gaussian_stepsize)
            + "  |             "
            + str(adaptive_stepsize)
        )
        print(
            "Acc Rate |        "
            + str(gaussian_acc_rate)
            + "    |           "
            + str(adaptive_acc_rate)
        )
        print(
            "ESS      |             "
            + str(int(gaussian_ess))
            + "   |           "
            + str(int(adaptive_ess))
        )
