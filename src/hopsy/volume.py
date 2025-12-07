"""
module for volume estimation
"""

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
    variance = safety_factor * radius / _s.chi2.ppf(1 - tolerance, n_dims)
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


def _estimate_log_ratio_from_log_weights(log_weights):
    """
    Shared core for converting log-weights into a log-ratio estimate
    and its Monte Carlo error using the effective sample size.

    This preserves the original behavior of `estimate_log_ratio`:
    - log_ratio is computed from flattened weights,
    - ESS is computed on `log_weights` reshaped with an extra
      singleton dimension at the end.
    """
    log_weights = _s.numpy.asarray(log_weights)

    # log mean exp
    log_ratio = _s.logsumexp(log_weights.ravel()) - _s.numpy.log(log_weights.size)

    # log of E[w^2]
    log_squared_ratio = _s.logsumexp(2.0 * log_weights.ravel()) - _s.numpy.log(
        log_weights.size
    )

    # Variance of importance weights in log-space:
    # Var(w) / (E[w]^2) = E[w^2] / (E[w]^2) - 1
    log_variance = _s.numpy.exp(log_squared_ratio - 2.0 * log_ratio) - 1.0

    # Preserve chain/sample structure for ESS
    ess_input = log_weights.reshape((*log_weights.shape, -1))
    effective_sample_size = ess(ess_input)[0][0]

    mc_error = _s.numpy.sqrt(max(log_variance, 0.0) / effective_sample_size)
    return log_ratio, mc_error


def estimate_log_ratio(samples, center, variance_im1: float, variance_i: float):
    """Cooling Gaussians, see DOI 10.1007/s12532-015-0097-z"""
    square_norm = ((samples - center) ** 2).sum(axis=-1)
    if variance_i is None:
        # this is for tail-correction by comparing gauss to uniform
        log_weights = 0.5 * square_norm * (1.0 / variance_im1)
    else:
        log_weights = 0.5 * square_norm * (1.0 / variance_im1 - 1.0 / variance_i)

    return _estimate_log_ratio_from_log_weights(log_weights)


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
    variances = [
        get_first_variance_fn(n_dims=n_dims, radius=radius, tolerance=tolerance)
    ]

    log_ratios = []
    log_ratio_errors = []

    rng = _c.RandomNumberGenerator(0)
    # TODO: using the first log_ratio, it is possible to check whether the first gaussian was essentially unconstrained
    while len(variances) < max_iterations:
        variances.append(get_next_variance_fn(variances[-1], n_dims))

        samples = sample_gaussian(
            p,
            center=center,
            variance=variances[-2],
            n_procs=n_procs,
            n_samples=sample_batch_size,
            random_seed=rng(),
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
        random_seed=rng(),
    )
    log_ratio, mc_error_log_ratio = estimate_log_ratio(
        samples, center, variance_im1=variances[-1], variance_i=None
    )
    log_ratios.append(log_ratio)
    log_ratio_errors.append(mc_error_log_ratio)

    log_volume_estimate = _s.numpy.sum(
        _s.numpy.array(log_ratios)
    ) + n_dims * 0.5 * _s.numpy.log(2 * _s.numpy.pi * variances[0])
    if compute_rounding:
        rounding_factor = _s.numpy.linalg.slogdet(p.transformation).logabsdet
        log_volume_estimate += rounding_factor
    log_volume_error = _s.numpy.sqrt(
        _s.numpy.sum(_s.numpy.square(_s.numpy.array(log_ratio_errors)))
    )
    return log_volume_estimate, log_volume_error


def _prepare_projection(
    projection,
    observation,
    sigma,
    n_dims: int,
    transformation=None,
):
    """
    Prepare (H, y, sigma) for the projected Gaussian energy.

    Two usage modes are supported:

    1) Mask mode (projection is 1D of length n_dims):
       - Each non-zero entry in `projection` defines ONE observation.
       - If projection[i] != 0, we create a row H_j such that
             H_j x = projection[i] * x_i.
       - The number of observations n_obs is the number of non-zero entries.
       - `observation` and `sigma` must be scalar or length n_obs.
         Scalars are broadcast to all observations.

    2) Full matrix mode (projection is 2D, shape (n_obs, n_dims)):
       - Each row of `projection` is one measurement direction.
       - `observation` and `sigma` must be scalar or length n_obs.
         Scalars are broadcast to all observations.

    If `transformation` is not None, it is assumed that
        x = T @ z
    and the projection is transformed accordingly:
        H_z = H_x @ T.
    """
    proj = _s.numpy.asarray(projection, dtype=float)

    # --- build H_raw and determine n_obs ---
    if proj.ndim == 1:
        # Mask mode: one observation per non-zero entry in proj
        if proj.shape[0] != n_dims:
            raise ValueError(
                f"projection has wrong dimensionality: expected length {n_dims}, "
                f"got {proj.shape[0]}"
            )

        idx = _s.numpy.nonzero(proj)[0]
        n_obs = idx.size
        if n_obs == 0:
            raise ValueError(
                "projection vector has no non-zero entries; "
                "at least one observation is required."
            )

        # H_raw[j, :] = proj[idx[j]] * e_{idx[j]}
        H_raw = _s.numpy.zeros((n_obs, n_dims), dtype=float)
        H_raw[_s.numpy.arange(n_obs), idx] = proj[idx]

    elif proj.ndim == 2:
        # Full matrix mode
        if proj.shape[1] != n_dims:
            raise ValueError(
                f"projection has wrong dimensionality: expected {n_dims} columns, "
                f"got {proj.shape[1]}"
            )
        H_raw = proj
        n_obs = H_raw.shape[0]

        if n_obs == 0:
            raise ValueError(
                "projection matrix has zero rows; "
                "at least one observation is required."
            )
    else:
        raise ValueError(
            f"projection must be 1D (mask) or 2D (matrix), got array with "
            f"{proj.ndim} dimensions."
        )

    # --- handle observation / sigma shapes ---
    y = _s.numpy.asarray(observation, dtype=float).reshape(-1)
    if y.size == 1 and n_obs > 1:
        y = _s.numpy.full(n_obs, y.item())
    elif y.size != n_obs:
        raise ValueError(
            f"observation has wrong length: expected {n_obs}, got {y.size}"
        )

    sigma_arr = _s.numpy.asarray(sigma, dtype=float).reshape(-1)
    if sigma_arr.size == 1 and n_obs > 1:
        sigma_arr = _s.numpy.full(n_obs, sigma_arr.item())
    elif sigma_arr.size != n_obs:
        raise ValueError(
            f"sigma has wrong length: expected {n_obs}, got {sigma_arr.size}"
        )

    # --- apply rounding transformation, if any ---
    if transformation is not None:
        T = _s.numpy.asarray(transformation, dtype=float)
        if T.shape != (n_dims, n_dims):
            raise ValueError(
                f"transformation must be of shape ({n_dims}, {n_dims}), "
                f"got {T.shape}"
            )
        H = H_raw @ T
    else:
        H = H_raw

    return H, y, sigma_arr


def _projected_gaussian_energy(samples, H, y, sigma):
    """
    Compute the projected Gaussian energy

        E(x) = 1/2 * || (H x - y) / sigma ||^2

    for a batch of points x (here: 'samples').
    """
    # samples: (N, d)
    proj = samples @ H.T  # (N, n_obs)
    residual = (proj - y) / sigma  # broadcast over observations
    return 0.5 * _s.numpy.sum(residual * residual, axis=-1)  # (N,)


def _estimate_log_ratio_to_projected_gaussian(
    samples,
    center,
    variance_im1: float,
    H,
    y,
    sigma,
):
    """
    Estimate log Z_proj - log Z_gauss where

    - Z_gauss is the normalizer of the truncated Gaussian with variance_im1,
    - Z_proj is the normalizer of the projected Gaussian

        pi_proj(x) ∝ exp(-E_proj(x)) 1_{x ∈ P} ,

    using samples from the Gaussian.
    """
    # Energy of the previous Gaussian stage
    square_norm = ((samples - center) ** 2).sum(axis=-1)
    energy_gauss = 0.5 * square_norm / variance_im1

    # Energy of the projected Gaussian
    energy_proj = _projected_gaussian_energy(samples, H, y, sigma)

    # Importance weights: w ∝ exp(-E_proj + E_gauss)
    log_weights = -energy_proj + energy_gauss
    return _estimate_log_ratio_from_log_weights(log_weights)


def estimate_projected_gaussian_log_normalization(
    problem,
    projection,
    observation,
    sigma,
    max_iterations=30,
    tolerance=1e-2,
    compute_rounding=True,
    n_procs=1,
    sample_batch_size=10000,
    get_first_variance_fn=get_first_variance,
    get_next_variance_fn=get_next_variance,
):
    """
    Estimate the log-normalization constant of a (possibly degenerate)
    Gaussian likelihood restricted to a polytope.

    We consider densities of the form

        pi(x) ∝ exp( -1/2 || (H x - y) / sigma ||^2 )  1_{x ∈ P},

    where P is the polytope defined by ``problem`` and

    - ``projection`` specifies H (either a single projection vector
      of shape (n_dims,) or a matrix of shape (n_obs, n_dims)),
    - ``observation`` specifies y (scalar or vector of length n_obs),
    - ``sigma`` specifies per-observation standard deviations
      (scalar or vector of length n_obs).

    The algorithm reuses the Gaussian cooling path used for
    polytope volume estimation, but changes the limiting
    distribution from the uniform measure to the projected
    Gaussian defined above.

    Parameters
    ----------
    problem : hopsy.Problem
        Polytope definition.
    projection :
        Projection matrix H (array_like, shape (n_obs, n_dims) or (n_dims,)).
    observation :
        Observed values y (scalar or array_like with shape (n_obs,)).
    sigma :
        Standard deviations (scalar or array_like with shape (n_obs,)).
    max_iterations : int, optional
        Maximum number of cooling steps (including the last Gaussian stage).
    tolerance : float, optional
        Tolerance used for determining the initial variance.
    compute_rounding : bool, optional
        If True, perform affine rounding of the polytope and account
        for the corresponding Jacobian in the returned log-normalizer.
    n_procs : int, optional
        Number of Markov chains to use in sampling.
    sample_batch_size : int, optional
        Number of samples per cooling stage.
    get_first_variance_fn, get_next_variance_fn : callables, optional
        Functions that define the cooling schedule.

    Returns
    -------
    log_Z_estimate : float
        Estimated log-normalization constant.
    log_Z_error : float
        Monte Carlo error estimate for log_Z_estimate.
    """
    if compute_rounding:
        p = round(problem)
    else:
        p = problem

    n_dims = p.A.shape[1]
    center, radius = compute_chebyshev_center_and_radius(p)
    variances = [
        get_first_variance_fn(n_dims=n_dims, radius=radius, tolerance=tolerance)
    ]

    # Prepare projected Gaussian in the coordinate system of `p`
    transformation = getattr(p, "transformation", None) if compute_rounding else None
    H, y, sigma_arr = _prepare_projection(
        projection=projection,
        observation=observation,
        sigma=sigma,
        n_dims=n_dims,
        transformation=transformation,
    )

    log_ratios = []
    log_ratio_errors = []

    rng = _c.RandomNumberGenerator(0)

    # Gaussian cooling path
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

    # Final step: last Gaussian -> projected Gaussian
    samples = sample_gaussian(
        p,
        center=center,
        variance=variances[-1],
        n_procs=n_procs,
        n_samples=sample_batch_size,
        random_seed=rng(),
    )
    log_ratio, mc_error_log_ratio = _estimate_log_ratio_to_projected_gaussian(
        samples,
        center=center,
        variance_im1=variances[-1],
        H=H,
        y=y,
        sigma=sigma_arr,
    )
    log_ratios.append(log_ratio)
    log_ratio_errors.append(mc_error_log_ratio)

    log_Z_estimate = _s.numpy.sum(
        _s.numpy.array(log_ratios)
    ) + n_dims * 0.5 * _s.numpy.log(2 * _s.numpy.pi * variances[0])

    if compute_rounding:
        rounding_factor = _s.numpy.linalg.slogdet(p.transformation).logabsdet
        log_Z_estimate += rounding_factor

    log_Z_error = _s.numpy.sqrt(
        _s.numpy.sum(_s.numpy.square(_s.numpy.array(log_ratio_errors)))
    )

    return log_Z_estimate, log_Z_error


def integrate_over_polytope(
    problem,
    max_iterations=30,
    tolerance=1e-2,
    compute_rounding=True,
    n_procs=1,
    sample_batch_size=10000,
    get_first_variance_fn=get_first_variance,
    get_next_variance_fn=get_next_variance,
    target=None,
):
    """
    Backwards compatible alias for ``estimate_polytope_log_volume``.

    The ``target`` argument is currently unused and reserved for
    future extensions where generic integrands could be supported.
    """
    if target is not None:
        raise NotImplementedError(
            "Generic integration with a custom target is not implemented yet. "
            "For now, this function only supports volume estimation and "
            "ignores 'target'."
        )

    return estimate_polytope_log_volume(
        problem=problem,
        max_iterations=max_iterations,
        tolerance=tolerance,
        compute_rounding=compute_rounding,
        n_procs=n_procs,
        sample_batch_size=sample_batch_size,
        get_first_variance_fn=get_first_variance_fn,
        get_next_variance_fn=get_next_variance_fn,
    )
