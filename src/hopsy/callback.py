"""

"""


class _submodules:
    import abc
    from typing import Dict, List, Optional, Sequence, Union

    import numpy as np

    try:
        import hagelkorn
        import mcbackend
    except ImportError:
        _has_mcbackend = False
    else:
        _has_mcbackend = True


_s = _submodules


class Callback(_s.abc.ABC):
    r"""Abstract base class for observing states and metadata"""

    def __init__(self, name: str = None):
        r"""
        Construct backend.

        :param name : str
            Name of the backend
        """
        self.n_chain = -1
        self.n_samples = -1
        self.n_dims = -1
        self.meta_names = None
        self.meta_shapes = None
        self.name = name

    def setup(
        self,
        n_chains: int,
        n_samples: int,
        n_dims: int,
        meta_names: _s.Sequence[str],
        meta_shapes: _s.Sequence[_s.Sequence[int]],
    ) -> None:
        r"""
        Setup backend for a specific MCMC chain.

        :param n_chains : int
            Number of chains
        :param n_samples: int
            Number of samples to produce
        :param n_dims: int
            Number of dimensions of the sampling problem
        :param meta_names: Sequence[str]
            String identifiers for meta information
        :param meta_shapes: Sequence[Sequence[int]]
            Shapes of meta information (empty for scalars)
        """
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_dims = n_dims
        self.meta_names = meta_names
        self.meta_shapes = meta_shapes

    def record(
        self,
        chain_idx: int,
        state: _s.np.ndarray,
        meta: _s.Dict[str, _s.Union[float, _s.np.ndarray]],
    ) -> None:
        r"""
        Record new MCMC state and metadata.

        :param chain_idx : int
            Index of the recording chain
        :param state : numpy.ndarray
            New MCMC state
        :param meta: Dict[str, Union[float, numpy.ndarray]]
            Recorded metadata of the step
        """
        raise NotImplementedError()

    def finish(self) -> None:
        r"""
        Finish recording (e.g. close connection to database).
        """
        pass


if _s._has_mcbackend:

    class MCBackendCallback(Callback):
        r"""Adapter to create a hospy backend from any McBackend backend."""

        supports_sampler_stats = True

        def __init__(
            self, backend: _s.mcbackend.Backend, name: _s.Optional[str] = None
        ):
            r"""
            Construct mcbackend callback.

            :param backend : mcbackend.Backend
                Backend implementation from mcbackend
            :param name : typing.Optional
                Name of the run
            """
            super().__init__(name)
            self.run_id = _s.hagelkorn.random(digits=6)
            print(f"Backend run id: {self.run_id}")
            self._backend: _s.mcbackend.Backend = backend

            # Sessions created from the underlying backend
            self._run: _s.Optional[_s.mcbackend.Run] = None
            self._chains: _s.Optional[_s.List[_s.mcbackend.Chain]] = None

        def setup(
            self,
            n_chains: int,
            n_samples: int,
            n_dim: int,
            meta_names: _s.Sequence[str],
            meta_shapes: _s.Sequence[_s.Sequence[int]],
        ) -> None:
            r"""
            Setup backend for a specific MCMC chain.

            :param n_chains : int
                Number of chains
            :param n_samples: int
                Number of samples to produce
            :param n_dims: int
                Number of dimensions of the sampling problem
            :param meta_names: Sequence[str]
                String identifiers for meta information
            :param meta_shapes: Sequence[Sequence[int]]
                Shapes of meta information (empty for scalars)
            """
            super().setup(n_chains, n_samples, n_dim, meta_names, meta_shapes)

            # Initialize backend sessions
            if not self._run:
                variables = [
                    _s.mcbackend.Variable(
                        f"variable_{i}",
                        _s.np.dtype(float).name,
                        [],
                        [],
                        is_deterministic=False,
                    )
                    for i in range(n_dim)
                ]

                sample_stats = []
                for name, shape in zip(meta_names, meta_shapes):
                    sample_stats.append(
                        _s.mcbackend.Variable(
                            name=name, dtype=_s.np.dtype(float).name, shape=shape
                        )
                    )

                run_meta = _s.mcbackend.RunMeta(
                    self.run_id,
                    variables=variables,
                    sample_stats=sample_stats,
                )
                self._run = self._backend.init_run(run_meta)
            self._chains = [
                self._run.init_chain(chain_number=i) for i in range(n_chains)
            ]

        def record(
            self,
            chain_idx: int,
            state: _s.np.ndarray,
            meta: _s.Dict[str, _s.Union[float, _s.np.ndarray]],
        ) -> None:
            r"""
            Record new MCMC state and metadata.

            :param chain_idx : int
                Index of the recording chain
            :param state : numpy.ndarray
                New MCMC state
            :param meta: Dict[str, Union[float, numpy.ndarray]]
                Recorded metadata of the step
            """
            draw = dict(zip([f"variable_{i}" for i in range(self.n_dims)], state))

            self._chains[chain_idx].append(draw, meta)

        def finish(self) -> None:
            r"""
            Finish recording (e.g. close connection to database).
            """
            pass
