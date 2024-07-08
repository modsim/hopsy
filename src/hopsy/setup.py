from .misc import MarkovChain, _c, _s
from .tuning import tune


def setup(
    problem: _c.Problem,
    random_seed: int,
    n_chains: int = None,
    proposal=None,
    n_tuning=0,
    n_tuning_rounds=100,
    target_accrate=0.234,
    tuning_target="accrate",
):
    """

    Parameters
    ----------
    problem
    random_seed: int
        To force easier reproducibility of scientific results, the user is forced to specify a random seed
    n_chains: int
        Optional, if none is given, will use all cores available on machine
    proposal: hopsy.ProposalType or proposal object (duck typing!)
        Optional: Will heuristically select fitting algorithm, see MarkovChain documentation
    n_tuning : int, default 1000
        Total budget of samples that may be drawn for tuning per chain.
        Only applicable if selected proposal is tunable, ignored otherwise
    n_tuning_rounds : int, default 0
        Number of Thompson Sampling rounds.
        Only applicable if selected proposal is tunable and n_tuning>0, ignored otherwise
    target_accrate : float, default=0.234
        Target accpetance rate for acceptance rate tuning.
    tuning_target: str
        What tuning_target to use, if n_tuning>0 Valid tuning targets are "accrate" for acceptance rate tuning,
        "esjd" for expected squared jump distance tuning and "esjd/s" for expected squared jump distance per second tuning.

    Returns
    -------
    tuple containing either 2 or 3 elements
    1) markov chains with sensible defaults
    2) random number generators
    3) tuning results only if tuning_target was not None
    """
    if n_chains is None or n_chains <= 0:
        n_chains = _s.multiprocessing.cpu_count()

    # set up random number generatores from single random number
    rng = _c.RandomNumberGenerator(random_seed)
    random_seeds = []
    while len(random_seeds) < n_chains:
        random_number = _c.UniformInt(0, 2_147_483_647)(rng)
        if random_number not in random_seeds:
            random_seeds.append(random_number)

    rngs = [_c.RandomNumberGenerator(rs) for rs in random_seeds]

    markov_chains = [MarkovChain(problem, proposal=proposal) for _ in rngs]
    assert len(markov_chains) == n_chains and len(random_seeds) == n_chains

    if n_tuning > 0 and hasattr(markov_chains[0].proposal, "stepsize"):
        mcs, rngs, tuning_results = tune(
            markov_chains,
            rngs,
            target=tuning_target,
            n_tuning=n_tuning,
            n_rounds=n_tuning_rounds,
            n_burnin=int(n_tuning_rounds / 2),
            accrate=target_accrate,
            n_procs=n_chains,
        )
        rngs = list(*rngs.values())
        return markov_chains, rngs, tuning_results

    return markov_chains, rngs
