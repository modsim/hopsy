from .misc import (  # TruncatedGaussianProposal,
    MarkovChain,
    _c,
    compute_chebyshev_center_and_radius,
    ess,
    round,
    sample,
    setup,
)


class _submodules:
    import numpy
    from scipy.special import logsumexp


_s = _submodules


def get_first_variance(
    problem: _c.Problem, center: _s.numpy.ndarray, radius: float, tolerance: float
) -> float:
    return radius * 1e-5


def get_next_variance(variance: float, n_dim: int) -> float:
    return variance * (1.0 + 1.0 / n_dim**2)


def sample_gaussian(
    problem: _c.Problem, center, variance, random_seed, n_samples, n_procs
):
    gaussian_problem = _c.Problem(
        problem.A,
        problem.b,
        _c.Gaussian(center, _s.numpy.identity(center.shape[0]) / variance),
    )
    mcs, rngs = setup(gaussian_problem, random_seed=random_seed)
    return sample(markov_chains=mcs, rngs=rngs, n_samples=n_samples, n_procs=n_procs)[1]


def estimate_log_ratio(samples, center, variance_diff: float, n_dims: int):
    square_norm = samples.flatten() ** 2
    # square_norm = ((samples-center)**2).sum(axis=-1)
    log_ratio_samples = -square_norm / 2 * variance_diff
    print("log_ratio_samples", log_ratio_samples.shape)
    print("samples shape", samples.shape)
    n_samples = samples.shape[0] * samples.shape[1]
    log_ratio = _s.logsumexp(log_ratio_samples) - _s.numpy.log(n_samples)
    # ess = _c.ess(samples)
    mc_error = 0
    return log_ratio, mc_error


def estimate_polytope_volume(
    problem,
    max_iterations=int(1e1),
    tolerance=1e-9,
    compute_rounding=True,
    n_procs=1,
    sample_batch_size=10000,
):
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    if compute_rounding:
        p = round(problem)
    else:
        p = problem

    n_dims = problem.A.shape[1]
    center, radius = compute_chebyshev_center_and_radius(p)
    variances = [get_first_variance(problem, center, radius, tolerance)]

    log_ratios = []

    sample_batch_size = 10000
    random_seed = 0
    while len(variances) < max_iterations:
        variances.append(get_next_variance(variances[-1], n_dims))
        converged = False
        while not converged:
            print("center, radius", center, radius)
            samples = sample_gaussian(
                problem,
                center=center,
                variance=variances[-2],
                n_procs=n_procs,
                n_samples=sample_batch_size,
                random_seed=random_seed,
            )
            log_ratio, mc_error_log_ratio = estimate_log_ratio(
                samples, center, variances[-2], n_dims
            )
            converged = True
        log_ratios.append(log_ratio)

    slogdet = _s.numpy.linalg.slogdet(p.transformation)
    rounding_factor = slogdet.sign * slogdet.logabsdet
    print(rounding_factor)
    log_volume_estimate = rounding_factor + _s.numpy.sum(log_ratios)
    print("log vol estimate", log_volume_estimate)
    volume_estimate = _s.numpy.exp(log_volume_estimate)
    print("vol estimate", volume_estimate)
    return volume_estimate
