import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hopsy


def hopsy_blackjax_gaussian_example():
    n_samples = 10000

    model = hopsy.Gaussian()

    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([100, 100, 100, 100])
    problem = hopsy.Problem(A, b, model=model)
    hopsy_mcs, hopsy_rngs = hopsy.setup(problem, random_seed=0, n_chains=1)
    _, hopsy_samples = hopsy.sample(
        hopsy_mcs, hopsy_rngs, backend="hops", n_samples=n_samples
    )

    mu = jnp.array(model.mean)
    # CAREFUL: INVERSE ONLY VALID IN THIS EXAMPLE!!
    inv_cov = jnp.array(model.covariance)

    # model.logdensity will automatically be jaxified in the future
    def jax_log_density(x):
        diff = x - mu
        return -0.5 * diff @ inv_cov @ diff

    step_size = 0.5
    inverse_mass_matrix = jnp.ones(2)
    num_integration_steps = 10
    # TODO blackjax.hmc will a hopsy type that understands polytopes in the future
    blackjax_mc = blackjax.hmc(
        jax.jit(jax_log_density), step_size, inverse_mass_matrix, num_integration_steps
    )

    _, blackjax_samples = hopsy.sample(
        blackjax_mc,
        rngs=hopsy_rngs,
        backend="blackjax",
        n_samples=n_samples,
        starting_point=mu,
    )

    plt.hist(hopsy_samples[0, :, 0], label="hopsy x_0", density=True, alpha=0.5)
    plt.hist(hopsy_samples[0, :, 1], label="hopsy x_1", density=True, alpha=0.5)
    plt.hist(blackjax_samples[0, :, 0], label="blackjax x_0", density=True, alpha=0.5)
    plt.hist(blackjax_samples[0, :, 1], label="blackjax x_1", density=True, alpha=0.5)
    plt.hist(hopsy_samples[0, :, 0], histtype="step", color="C0", density=True)
    plt.hist(hopsy_samples[0, :, 1], histtype="step", color="C1", density=True)
    plt.hist(blackjax_samples[0, :, 0], histtype="step", color="C2", density=True)
    plt.hist(blackjax_samples[0, :, 1], histtype="step", color="C3", density=True)
    plt.show(block=False)


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "gpu")
    hopsy_blackjax_gaussian_example()


def kinetic_example():
    from jax.experimental.ode import odeint

    def mm_rhs(y, t, params):
        Vmax, Km = params
        S, P = y
        rate = Vmax * S / (Km + S)
        return jnp.array([-rate, rate])

    def solve_mm(t, y0, params):
        return odeint(mm_rhs, y0, t, params)

    true_params = dict(Vmax=1.2, Km=0.35, sigma=0.05)  # ground-truth
    t_obs = jnp.linspace(0.0, 4.0, 41)  # 41 points
    y0 = jnp.array([1.0, 0.0])  # S0, P0

    clean_traj = solve_mm(t_obs, y0, (true_params["Vmax"], true_params["Km"]))
    S_clean = clean_traj[:, 0]

    key = jax.random.PRNGKey(0)
    key, subk = jax.random.split(key)
    noise = true_params["sigma"] * jax.random.normal(subk, S_clean.shape)
    S_obs = S_clean + noise  # observed substrate

    def unpack(theta):
        log_Vmax, log_Km, log_sigma = theta
        Vmax = jnp.exp(log_Vmax)
        Km = jnp.exp(log_Km)
        sigma = jnp.exp(log_sigma)
        return (Vmax, Km, sigma)

    def loglik(theta, t=t_obs, y0=y0, S_data=S_obs):
        Vmax, Km, sigma = unpack(theta)
        S_pred = solve_mm(t, y0, (Vmax, Km))[:, 0]
        resid = (S_data - S_pred) / sigma
        logpdf = -0.5 * jnp.sum(resid**2 + jnp.log(2 * jnp.pi * sigma**2))
        return logpdf

    logposterior_jit = jax.jit(loglik)

    step_size = 0.0001
    inverse_mass_matrix = 0.05 * jnp.ones(3)
    num_integration_steps = 10
    blackjax_mc = blackjax.hmc(
        logposterior_jit, step_size, inverse_mass_matrix, num_integration_steps
    )

    hopsy_rngs = hopsy.RandomNumberGenerator(0)
    n_samples = 50000
    initial_theta = jnp.log(jnp.array([0.8, 0.8, 0.1]))
    _, blackjax_samples = hopsy.sample(
        blackjax_mc,
        rngs=hopsy_rngs,
        backend="blackjax",
        n_samples=n_samples,
        starting_point=initial_theta,
    )
    blackjax_samples = np.exp(blackjax_samples)

    plt.hist(blackjax_samples[0, :, 0], label="blackjax x_0", density=True, alpha=0.5)
    plt.hist(blackjax_samples[0, :, 1], label="blackjax x_1", density=True, alpha=0.5)
    plt.hist(blackjax_samples[0, :, 2], label="blackjax x_2", density=True, alpha=0.5)
    plt.axvline(true_params["Vmax"], color="C0", linestyle="--", linewidth=2)
    plt.axvline(true_params["Km"], color="C1", linestyle="--", linewidth=2)
    plt.axvline(true_params["sigma"], color="C2", linestyle="--", linewidth=2)
    plt.show()
