from hopsy.core import RandomNumberGenerator

from .misc import (  # TruncatedGaussianProposal,
    _c,
    compute_chebyshev_center_and_radius,
    ess,
    rhat,
    round,
    sample,
    setup,
)


class _submodules:
    import numpy
    from scipy.special import logsumexp
    from scipy.stats import chi2


_s = _submodules


def get_first_variance(
       n_dims: int, radius: float, tolerance: float, safety_factor: float = 0.5
) -> float:
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    variance = safety_factor * radius / _s.chi2.ppf(1-tolerance, n_dims)
    return variance


def get_next_variance(variance: float, n_dim: int) -> float:
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    return variance * (1.0 + 1.0 / n_dim)


def sample_gaussian(
        problem: _c.Problem, center, variance, random_seed, n_samples, n_procs
):
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    gaussian_problem = _c.Problem(
        problem.A,
        problem.b,
        _c.Gaussian(center, _s.numpy.identity(center.shape[0]) * variance),
    )
    mcs, rngs = setup(gaussian_problem, random_seed=random_seed, n_chains=n_procs)
    # TODO: set smart starting point in typical set
    return sample(markov_chains=mcs, rngs=rngs, n_samples=n_samples, n_procs=n_procs)[1]


def estimate_log_ratio(samples, center, variance_im1: float, variance_i: float):
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    square_norm = ((samples - center) ** 2).sum(axis=-1)
    if variance_i is None:
        # this is for tail-correction by comparing gauss to uniform
        log_weights = 0.5 * square_norm * (1. / variance_im1)
    else:
        log_weights = 0.5 * square_norm * (1. / variance_im1 - 1. / variance_i)
    log_ratio = _s.logsumexp(log_weights.ravel()) - _s.numpy.log(log_weights.size)
    log_squared_ratio = _s.logsumexp(2. * log_weights.ravel()) - _s.numpy.log(log_weights.size)
    log_variance = _s.numpy.exp(log_squared_ratio - 2. * log_ratio) - 1.
    log_weights = log_weights.reshape((*log_weights.shape, -1))
    effective_sample_size = ess(log_weights)[0][0]
    mc_error = _s.numpy.sqrt(max(log_variance, 0.) / effective_sample_size)
    return log_ratio, mc_error


def estimate_polytope_log_volume(
        problem,
        max_iterations=30,
        tolerance=1e-2,
        compute_rounding=True,
        n_procs=1,
        sample_batch_size=10000,
        get_first_variance_fn=get_first_variance,
        get_next_variance_fn=get_next_variance,
):
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    if compute_rounding:
        p = round(problem)
    else:
        p = problem

    n_dims = p.A.shape[1]
    center, radius = compute_chebyshev_center_and_radius(p)
    variances = [get_first_variance_fn(n_dims=n_dims, radius=radius, tolerance=tolerance)]

    log_ratios = []
    log_ratio_errors = []

    rng = _c.RandomNumberGenerator(0)
    # TODO: using the first log_ratio, it is possible to check whether the first gaussian was essentially unconstrained
    while len(variances) < max_iterations:
        random_seed = rng()
        variances.append(get_next_variance_fn(variances[-1], n_dims))

        samples = sample_gaussian(
            p,
            center=center,
            variance=variances[-2],
            n_procs=n_procs,
            n_samples=sample_batch_size,
            random_seed=random_seed,
        )
        log_ratio, mc_error_log_ratio = estimate_log_ratio(
            samples, center, variance_im1=variances[-2], variance_i=variances[-1]
        )
        log_ratios.append(log_ratio)
        log_ratio_errors.append(mc_error_log_ratio)

    # tail correction:
    samples = sample_gaussian(
        p,
        center=center,
        variance=variances[-1],
        n_procs=n_procs,
        n_samples=sample_batch_size,
        random_seed=random_seed,
    )
    log_ratio, mc_error_log_ratio = estimate_log_ratio(samples, center, variance_im1=variances[-1], variance_i=None)
    log_ratios.append(log_ratio)
    log_ratio_errors.append(mc_error_log_ratio)


    log_volume_estimate = _s.numpy.sum(_s.numpy.array(log_ratios)) + n_dims * 0.5 * _s.numpy.log(
        2 * _s.numpy.pi * variances[0])
    if compute_rounding:
        rounding_factor = _s.numpy.linalg.slogdet(p.transformation).logabsdet
        log_volume_estimate += rounding_factor
    log_volume_error = _s.numpy.sqrt(_s.numpy.sum(_s.numpy.square(_s.numpy.array(log_ratio_errors))))
    return log_volume_estimate, log_volume_error
