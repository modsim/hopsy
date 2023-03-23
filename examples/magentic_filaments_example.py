import matplotlib.pyplot as plt
import numpy as np

import hopsy


class UnconstrainedProposal:
    def __init__(self, x: np.ndarray, cov: np.ndarray):
        self.x = x
        self.cov = cov
        self.r = 1
        self.proposal = x

    def propose(self):
        mean = np.zeros((len(self.cov),))
        y = np.random.multivariate_normal(mean, self.cov).reshape(-1, 1)
        y[:2] = np.zeros(y[:2].shape)  # disallow the anchor to move
        self.proposal = self.x + self.r * y

    def accept_proposal(self):
        self.x = self.proposal

    def compute_log_acceptance_probability(self) -> float:
        return 0

    def get_state(self) -> np.ndarray:
        return self.x

    def set_state(self, new_state: np.ndarray):
        self.x = new_state.reshape(-1, 1)

    def get_proposal(self) -> np.ndarray:
        return self.proposal

    def get_stepsize(self) -> float:
        return self.r

    def set_stepsize(self, new_stepsize: float):
        self.r = new_stepsize

    def get_name(self) -> str:
        return "PyGaussianProposal"


class Potentials:
    def __init__(self, K_f, r_f, sigma, H, beta, eps):
        self.K_f = K_f
        self.r_f = r_f
        self.sigma = sigma
        self.r_cut = 2 ** (1 / 6) * sigma
        self.H = H
        self.beta = beta
        self.eps = eps
        self.U_LJ_cut = (
            4
            * self.eps
            * ((self.sigma / self.r_cut) ** 12 - (self.sigma / self.r_cut) ** 6)
        )

    def compute_negative_log_likelihood(self, x):
        return self.beta * (
            self.fene_potential(x) + self.wca_potential(x) + self.zeeman_potential(x)
        )

    def fene_potential(self, x):
        # every particle has 4 degrees of freedom: 2 space coordinates, 2 dipole moment coordinates
        assert len(x) % 4 == 2

        n_particles = int((len(x) - 2) / 4)

        r = np.linalg.norm(x[2:4])
        U_FENE = (
            (-self.K_f * self.r_f**2) * np.log(1 - (r / self.r_f) ** 2) / 2
            if (r / self.r_f) ** 2 < 1
            else np.inf
        )
        for i in range(1, n_particles):
            particle_j_coord = x[4 * (i - 1) + 2 : 4 * (i - 1) + 4]
            particle_i_coord = x[4 * i + 2 : 4 * i + 4]
            r = np.linalg.norm(particle_i_coord - particle_j_coord)
            U_FENE += (
                (-self.K_f * self.r_f**2) * np.log(1 - (r / self.r_f) ** 2) / 2
                if (r / self.r_f) ** 2 < 1
                else np.inf
            )

        return U_FENE

    def wca_potential(self, x):
        # every particle has 4 degrees of freedom: 2 space coordinates, 2 dipole moment coordinates
        assert len(x) % 4 == 2

        n_particles = int((len(x) - 2) / 4)

        U_WCA = 0
        for i in range(n_particles):
            r = np.linalg.norm(x[4 * i + 2 : 4 * i + 4])
            U_WCA += (
                0
                if r > self.r_cut
                else 4 * self.eps * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)
                - self.U_LJ_cut
            )
            for j in range(n_particles):
                if i == j:
                    continue
                particle_i_coord = x[4 * i + 2 : 4 * i + 4]
                particle_j_coord = x[4 * j + 2 : 4 * j + 4]
                r = np.linalg.norm(particle_i_coord - particle_j_coord)
                U_WCA += (
                    0
                    if r > self.r_cut
                    else 4 * self.eps * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)
                    - self.U_LJ_cut
                )

        return U_WCA

    def zeeman_potential(self, x):
        # every particle has 4 degrees of freedom: 2 space coordinates, 2 dipole moment coordinates
        assert len(x) % 4 == 2

        n_particles = int((len(x) - 2) / 4 + 1)

        U_H = np.dot(x[:2], self.H)
        for i in range(n_particles):
            particle_i_dipole_moment = x[4 * i : 4 * i + 2]
            U_H += np.dot(particle_i_dipole_moment, self.H)

        return U_H


def draw_state(x, sigma, ax, color, n_dof=4):
    assert len(x) % n_dof == 2
    n_particles = int(len(x) / n_dof)

    particle = plt.Circle((0, 0), sigma, clip_on=False, fill=False, color=color)
    ax.add_patch(particle)
    for i in range(n_particles):
        particle = plt.Circle(
            x[n_dof * i + 2 : n_dof * i + 2 + 4],
            sigma,
            clip_on=False,
            fill=False,
            color=color,
        )
        ax.add_patch(particle)

    particle_j_coord = x[2:4]
    ax.plot([0, particle_j_coord[0]], [0, particle_j_coord[1]], color=color)
    for i in range(n_particles - 1):
        particle_i_coord = x[4 * i + 2 : 4 * i + 4]
        particle_j_coord = x[4 * (i + 1) + 2 : 4 * (i + 1) + 4]
        ax.plot(
            [particle_i_coord[0], particle_j_coord[0]],
            [particle_i_coord[1], particle_j_coord[1]],
            color=color,
        )


def get_initial_state(n_particles, r_f, sigma):
    pre_x = np.array([0, 0])
    x0 = [np.random.rand((2))]
    for i in range(n_particles - 1):
        pre_x = pre_x + r_f / 2 + sigma
        x0.append(pre_x)
        x0.append(np.random.rand((2)))

    return np.array(x0).flatten()


def compute_sample_average(f, x):
    average = 0
    for i in range(len(x)):
        average += f(x[i])

    print(len(x))
    return average / len(x)


def main():
    n_particles = 10
    n_dof = 4

    r_f = 1
    sigma = 0.1

    x = get_initial_state(n_particles, r_f, sigma)
    U = Potentials(1, r_f, sigma, np.array([0, 0]), 1, 1)
    print(U.compute_negative_log_likelihood(x))

    A = np.hstack([np.array([[-1]]), np.zeros((1, len(x) - 1))])
    b = np.zeros((1, 1))

    problem = hopsy.Problem(A, b, U)

    # proposal = UnconstrainedProposal(x, 0.0002*np.identity(n_particles * n_dof - 2))
    run = hopsy.Run(problem, "Gaussian")
    run.stepsize = 0.02

    run.starting_points = [x]
    run.init()
    run.sample(1000, 10)

    state = run.data.states[-1][-1]
    # print(hopsy.compute_potential_scale_reduction_factor(run.data))

    print(x)
    print(state)

    print("acceptance rate", hopsy.compute_acceptance_rate(run.data)[-1])
    print(
        compute_sample_average(
            lambda x: np.linalg.norm(x[-4:-2]), np.array(run.data.states[-1])
        )
    )

    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    ax.set_aspect("equal")

    draw_state(x, sigma, ax=ax, color="C0")

    draw_state(state, sigma, ax=ax, color="C1")

    plt.show()


if __name__ == "__main__":
    main()
