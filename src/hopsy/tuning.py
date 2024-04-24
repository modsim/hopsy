"""
Tuning module for hyperparameter tuning of proposal distributions using Bayesian optimization.
"""


class _submodules:
    import time

    import numpy as np
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF


_s = _submodules

from .misc import *
from .misc import _c


def inv_sigmoid(x):
    r"""
    Inverse sigmoid function

    .. math::
        y = \log(x) - \log(1 - x)

    """
    return _s.np.log(x) - _s.np.log(1 - x)


def sigmoid(x):
    r"""
    Sigmoid function

    .. math::
        y = \frac{1}{1 + \exp(-x)}

    """
    return 1.0 / (1 + _s.np.exp(-x))


def set_params(mcs, params_map, param_values, transforms):
    """ """
    for mc in mcs:
        for i, param in enumerate(params_map):
            if param in transforms:
                setattr(mc.proposal, param, transforms[param](param_values[i]))
            else:
                setattr(mc.proposal, param, param_values[i])


default_transforms = {
    "stepsize": lambda x: 10**x,
    "fisher_weight": lambda x: sigmoid(x),
}


def estimate_accrate(mcs, rngs, n_samples, *args, **kwargs):
    r"""
    Estimate the acceptance rate of a Markov chain
    by computing average number of accepted proposals per sample drawn

    .. math::
        \alpha = \mathbb{E}\left[ 1_{x_i \neq x_{i+1}} \right]


    Parameters
    ----------
    mcs : list of hopsy.MarkovChain
        Markov chains, which will be used to generate samples.
    rngs : list of hopsy.RandomNumberGenerator
        Random number generators required for sampling from the Markov chains.
    n_samples : int
        Number of samples that will be drawn per chain.
    *args : tuple
        Additional arguments will be passed to `hopsy.sample`.
    **kwargs : dict
        Additional keyword arguments will be passed to `hopsy.sample`.

    Returns
    -------
        np.array
            Array with acceptance rates for every individual chain.
    """
    accrates, samples = sample(mcs, rngs, n_samples, *args, **kwargs)
    return _s.np.array(accrates)


def estimate_esjd(mcs, rngs, n_samples, k=[1], consider_time=False, *args, **kwargs):
    r"""
    Estimate the Expected Squared Jump Distance of a Markov chain
    by computing the average squared displacement of consecutive samples

    .. math::
        ESJD = \mathbb{E}\left[\|x_i - x_{i+t}\|_2^2\right]

    for lag :math:`t`.
    If more than one lag is specified, the resulting ESJDs are summed up.

    Parameters
    ----------
    mcs : list of hopsy.MarkovChain
        Markov chains, which will be used to generate samples.
    rngs : list of hopsy.RandomNumberGenerator
        Random number generators required for sampling from the Markov chains.
    n_samples : int
        Number of samples that will be drawn per chain.
    k : list of int, default=[1]
        Lags at which the ESJDs are computed. If more than one lag is passed,
        ESJDs across multiple lags will be summed up.
    consider_time : bool, default=False
        If set to `True`, the ESJD will be per second instead of being per sample.
        Note that runtimes are just measured as wall clock times.
    *args : tuple
        Additional arguments will be passed to `hopsy.sample`.
    **kwargs : dict
        Additional keyword arguments will be passed to `hopsy.sample`.

    Returns
    -------
        np.array
            Array of cumulative ESJD over specified lags `k` for every individual chain.
    """
    start = _s.time.perf_counter()
    _, samples = sample(mcs, rngs, n_samples, *args, **kwargs)
    time_per_sample = (_s.time.perf_counter() - start) / n_samples
    esjds = _s.np.sum(
        [
            _s.np.mean(
                _s.np.sum(_s.np.diff(samples[:, ::_k], axis=1) ** 2, axis=-1), axis=-1
            )
            for _k in k
        ],
        axis=0,
    )
    return (esjds / time_per_sample) if consider_time else esjds


def tune_acceptance_rate(
    mcs,
    rngs,
    accrate=0.234,
    seed=42,
    n_procs=1,
    n_rounds=100,
    n_samples=1000,
    lower=-5,
    upper=5,
    n_points=256,
    params=["stepsize"],
    transforms=default_transforms,
):
    """ """
    lower = [lower] if not hasattr(lower, "__len__") else lower
    upper = [upper] if not hasattr(upper, "__len__") else upper

    log_prior_var = 10**2
    log_prior_mean = _s.np.log(1e-5)

    domains = dict()
    param_map = dict()
    gprs = dict()
    data = dict()

    channel = 0

    for j, key in enumerate(mcs):
        channel += len(mcs[key])
        param_map[key] = list()

        a, b = [], []
        i = 0
        for param in params:
            if hasattr(mcs[key][0].proposal, param):
                param_map[key].append(param)
                a.append(lower[min(i, len(lower) - 1)])
                b.append(upper[min(i, len(upper) - 1)])

        a, b = _s.np.array(a), _s.np.array(b)

        sampler = _s.qmc.Sobol(d=len(param_map[key]), scramble=True)
        domains[key] = (b - a) * sampler.random_base2(m=int(_s.np.log2(n_points))) + a

        gprs[key] = _s.GaussianProcessRegressor(
            kernel=log_prior_var * _s.RBF(), random_state=seed + 1, optimizer=None
        )
        data[key] = {"X": [], "y": []}

    rng = _c.RandomNumberGenerator(seed, channel + 1)

    for i in range(n_rounds):
        for j, key in enumerate(gprs):
            sample = gprs[key].sample_y(domains[key], random_state=rng())
            idx = (-((sigmoid(sample) - accrate) ** 2)).argmax()
            x = domains[key][idx]

            set_params(mcs[key], param_map[key], x, transforms)

            _y = (
                estimate_accrate(
                    mcs[key], rngs[key], n_samples=n_samples, n_procs=n_procs
                )
                * 0.9998
                + 0.0001
            )
            y = inv_sigmoid(_y)

            data[key]["X"].append(x)
            data[key]["y"].append(y)

            std = _s.np.std(data[key]["y"], axis=-1)
            gprs[key].alpha = _s.np.max(
                _s.np.hstack([std, 1e-3 * _s.np.ones_like(std)]), axis=-1
            )
            gprs[key].fit(data[key]["X"], _s.np.mean(data[key]["y"], axis=-1))

    for j, key in enumerate(gprs):
        idx = gprs[key].predict(domains[key]).argmax()
        x = domains[key][idx]
        set_params(mcs[key], param_map[key], x, transforms)

    return mcs, rngs, (gprs, domains)


def tune_esjd(
    mcs,
    rngs,
    k=[1],
    seed=42,
    n_procs=1,
    n_burnin=50,
    n_rounds=100,
    n_samples=1000,
    consider_time=True,
    sample_proposal=True,
    lower=-5,
    upper=5,
    n_points=256,
    params=["stepsize"],
    transforms=default_transforms,
):
    """ """
    if not hasattr(k, "__len__"):
        k = [k]

    lower = [lower] if not hasattr(lower, "__len__") else lower
    upper = [upper] if not hasattr(upper, "__len__") else upper

    log_prior_var = 1
    log_prior_mean = _s.np.log(1e-5)

    domains = dict()
    param_map = dict()
    gprs = dict()
    data = dict()

    channel = 0

    for j, key in enumerate(mcs):
        channel += len(mcs[key])
        param_map[key] = list()

        a, b = [], []
        i = 0
        for param in params:
            if hasattr(mcs[key][0].proposal, param):
                param_map[key].append(param)
                a.append(lower[min(i, len(lower) - 1)])
                b.append(upper[min(i, len(upper) - 1)])

        a, b = _s.np.array(a), _s.np.array(b)

        sampler = _s.qmc.Sobol(d=len(param_map[key]), scramble=True)
        domains[key] = (b - a) * sampler.random_base2(m=int(_s.np.log2(n_points))) + a

        gprs[key] = _s.GaussianProcessRegressor(
            kernel=log_prior_var * _s.RBF(), random_state=seed + 1, optimizer=None
        )
        data[key] = {"X": [], "y": []}

    rng = _c.RandomNumberGenerator(seed, channel + 1)

    for i in range(n_rounds):
        if sample_proposal and i > n_burnin:
            maxkey = None
            maxsample = 0
            for j, key in enumerate(gprs):
                sample = gprs[key].sample_y(domains[key], random_state=rng())
                idx = sample.argmax()
                x = domains[key][idx]

                if _s.np.exp(sample[idx]) > maxsample:
                    maxsample = sample[idx]
                    maxkey = key

            key = maxkey

            set_params(mcs[key], param_map[key], x, transforms)

            _y = (
                estimate_esjd(
                    mcs[key],
                    rngs[key],
                    k=k,
                    consider_time=consider_time,
                    n_samples=n_samples,
                    n_procs=n_procs,
                )
                + 1e-5
            )
            y = _s.np.log(_y) - log_prior_mean

            data[key]["X"].append(x)
            data[key]["y"].append(y)

            log_prior_var = max(
                2 * _s.np.max(_s.np.array(data[key]["y"])), log_prior_var
            )

            std = _s.np.std(data[key]["y"], axis=-1)
            gprs[key].kernel = log_prior_var**2 * _s.RBF()
            gprs[key].alpha = _s.np.max(
                _s.np.vstack([std, 1e-3 * _s.np.ones_like(std)]), axis=0
            )
            gprs[key].fit(data[key]["X"], _s.np.mean(data[key]["y"], axis=-1))
        else:
            for j, key in enumerate(gprs):
                sample = gprs[key].sample_y(
                    domains[key], random_state=rng()
                )  # - prior_mean
                idx = sample.argmax()
                x = domains[key][idx]

                set_params(mcs[key], param_map[key], x, transforms)

                _y = (
                    estimate_esjd(
                        mcs[key],
                        rngs[key],
                        k=k,
                        consider_time=consider_time,
                        n_samples=n_samples,
                        n_procs=n_procs,
                    )
                    + 1e-5
                )
                y = _s.np.log(_y) - log_prior_mean

                data[key]["X"].append(x)
                data[key]["y"].append(y)

                log_prior_var = max(
                    2 * _s.np.max(_s.np.array(data[key]["y"])) ** 2, log_prior_var
                )

                std = _s.np.std(data[key]["y"], axis=-1)
                gprs[key].kernel = log_prior_var * _s.RBF()
                gprs[key].alpha = _s.np.max(
                    _s.np.hstack([std, 1e-3 * _s.np.ones_like(std)]), axis=-1
                )
                gprs[key].fit(data[key]["X"], _s.np.mean(data[key]["y"], axis=-1))

    for j, key in enumerate(gprs):
        idx = gprs[key].predict(domains[key]).argmax()
        x = domains[key][idx]
        set_params(mcs[key], param_map[key], x, transforms)

    return mcs, rngs, (gprs, domains)


def tune(
    mcs,
    rngs,
    target,
    n_tuning,
    k=[1],
    accrate=0.234,
    seed=42,
    n_procs=1,
    n_rounds=100,
    n_burnin=50,
    sample_proposal=True,
    lower=-5,
    upper=5,
    n_points=256,
    params=["stepsize"],
    transforms=None,
):
    r"""
    Thompson Sampling-based tuning method for specifying meaningful hyperparameters
    for Markov chain proposal distributions. This method supports three different tuning targets,
    namely the commonly used accpetance rate, the Expected Squared Jump Distance (ESJD) and the ESJD
    per second (ESJD/s) rather than per sample.

    Tuning happens over a fixed search domain created from a Sobol sequence of `n_points` grid points
    with lower and upper bound using `lower` and `upper`. Its dimension is automatically adapted to the
    number of parameters that are to be tuned per proposal. Note that constraints are handled by transforming
    the parameters using the transformations stored in the `transforms` dict. Default transforms are available
    for `stepsize`, where an exponential transform is applied (i.e. Thompson Sampling happens in log space)
    and for `fisher_weight`, where a sigmoid function is applied.

    Parameters
    ----------
    mcs : list of list or list of hopsy.MarkovChain
        Markov chains, which will be used to generate samples. If list of lists of ``hopsy.MarkovChain``s
        are passed, outer lists are expected to share the same proposal algorithm,
        inner lists to be replicas of the same chain.
    rngs : list of list or list of hopsy.RandomNumberGenerator
        Random number generators required for sampling from the Markov chains. Should match `mcs` in number
        and shape.
    target : str
        Tuning target. Choose one of `accrate`, `esjd` or `esjd/s`.
    n_tuning : int
        Total budget of samples that may be drawn for tuning per chain.
    k : list of int, default=[1]
        Lags at which the ESJDs are computed. If more than one lag is passed,
        ESJDs across multiple lags will be summed up.
    accrate : float, default=0.234
        Target accpetance rate for acceptance rate tuning.
    seed : int, default=42
        RNG seed for the Thompson Sampling procedure.
    n_procs : int, default=1
        Number of parallel chains. Is passed as is to `hopsy.sample`.
    n_rounds : int, default=100
        Number of Thompson Sampling rounds.
    n_burnin : int, default=50
        Number of Thompson Sampling rounds, where it is just naively applied to all proposal distributions.
        This option only makes any difference, if `sample_proposal==True`.
    sample_proposal : bool, default=True
        Choose whether to apply TS also on the discrete choice of proposal algorithm to use. This option
        is only supported if tuning w.r.t. to ESJD or ESJD/s.
    lower : float, default=-5
        Lower bound on the search domain of the transformed hyperparameters.
    upper : float, default=5
        Upper bound on the search domain of the transformed hyperparameters.
    n_points : int, default=256
        Number of points to draw from the Sobol sequence. Should ideally be some power of 2.
    params : list of str, default=['stepsize']
        Parameters that are to be tuned. They get accessed as attributes of `mc.proposal`.
        If a proposal doesn't provide the corresponding parameter, then it will just be ignored.
    transforms : dict of callable, default=None
        Dictonary containing transforms mapping unbounded values to the proper range for a given parameter.
        Default transforms exist for `stepsize`, which gets mapped using an exponential function,
        and `fisher_weight`, which gets mapped using a sigmoid. Default values are overridden if any other
        transforms are passed here for the same parameters.

    Returns
    -------
    mcs : dict of list of hopsy.MarkovChain
        List of tuned Markov chains, where each list of replicas is stored with `key==mc.proposal.__class__.__name__`.
    rngs : dict of list of hopsy.RandomNumberGenerator
        List of forwarded random number generators, where corresponding RNGs to the Markov chains in `mcs` are
        also stored with `key==mc.proposal.__class__.__name__`.
    (gprs, domains) : tuple of dict
        `gprs` contains the fitted sklearn `GaussianProcessRegressor`s, which were used as surrogate functions
        to the blackbox tuning target. `domains` contains the Sobol sequence domains on which the GPs were
        evaluated to obtain the next hyperparameter choice. Both dicts store using the respective proposal
        names as keys.
    """

    if isinstance(mcs, _c.MarkovChain):
        mcs = [[mcs]]
        rngs = [[rngs]]

    if isinstance(mcs, list) and isinstance(mcs[0], _c.MarkovChain):
        mcs = [mcs]
        rngs = [rngs]

    if isinstance(mcs, list):
        _mcs = dict()
        _rngs = dict()

        for i, group in enumerate(mcs):
            key = group[0].proposal.__class__.__name__

            for mc in group:
                if mc.proposal.__class__.__name__ != key:
                    raise RuntimeError(
                        """
                        `mcs` need to be list of list of type `hopsy.core.MarkovChain`, where every inner list of `hopsy.core.MarkovChain` objects is required to use the same proposal distribution.
                    """
                    )

            _mcs[key] = group
            _rngs[key] = rngs[i]

    mcs = _mcs
    rngs = _rngs

    n_samples = int(n_tuning / (n_rounds + (len(mcs) - 1) * n_burnin))

    # select transforms for params where we also want to perform tuning
    _transforms = {
        param: transform
        for param, transform in default_transforms.items()
        if param in params
    }

    # add user-defined transforms
    transforms = (
        {**_transforms, **transforms} if transforms is not None else _transforms
    )

    if target.lower() == "esjd":
        return tune_esjd(
            mcs,
            rngs,
            k=k,
            seed=seed,
            n_procs=n_procs,
            n_rounds=n_rounds,
            n_burnin=n_burnin,
            n_samples=n_samples,
            consider_time=False,
            sample_proposal=sample_proposal,
            lower=lower,
            upper=upper,
            n_points=n_points,
            params=params,
            transforms=transforms,
        )
    if target.lower() == "esjd/s":
        return tune_esjd(
            mcs,
            rngs,
            k=k,
            seed=seed,
            n_procs=n_procs,
            n_rounds=n_rounds,
            n_burnin=n_burnin,
            n_samples=n_samples,
            consider_time=True,
            sample_proposal=sample_proposal,
            lower=lower,
            upper=upper,
            n_points=n_points,
            params=params,
            transforms=transforms,
        )
    if target.lower() == "accrate":
        return tune_acceptance_rate(
            mcs,
            rngs,
            accrate=accrate,
            seed=seed,
            n_procs=n_procs,
            n_rounds=n_rounds,
            n_samples=n_samples,
            lower=lower,
            upper=upper,
            n_points=n_points,
            params=params,
            transforms=transforms,
        )
