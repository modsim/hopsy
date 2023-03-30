"""

"""
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


class Backend(abc.ABC):
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
        meta_names: List[str],
        meta_shapes: List[List[int]],
    ) -> None:
        r"""
        Setup backend for a specific MCMC chain.

        :param n_chains : int
            Number of chains
        :param n_samples: int
            Number of samples to produce
        :param n_dims: int
            Number of dimensions of the sampling problem
        :param meta_names: List[str]
            String identifiers for meta information
        :param meta_shapes: List[List[int]]
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
        state: np.ndarray,
        meta: Dict[str, Union[float, np.ndarray]],
    ) -> None:
        r"""
        Record new MCMC state and metadata.

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


def enable_if(cond):
    if cond:

        def dec_enable_if(cls):
            return cls

        return dec_enable_if
    else:

        def dec_enable_if(cls):
            return

        return dec_enable_if


@enable_if(cond=_has_mcbackend)
class MCBBackend(Backend):
    """Adapter to create a hospy backend from any McBackend backend."""

    supports_sampler_stats = True

    def __init__(self, backend: Backend, name: Optional[str] = None):
        super().__init__(name)
        self.run_id = hagelkorn.random(digits=6)
        print(f"Backend run id: {self.run_id}")
        self._backend: Backend = backend

        # Sessions created from the underlying backend
        self._run: Optional[Run] = None
        self._chains: Optional[List[Chain]] = None

    def setup(
        self,
        n_chains: int,
        n_samples: int,
        n_dim: int,
        meta_names: Sequence[str],
        meta_shapes: Sequence[Sequence[int]],
    ) -> None:
        super().setup(n_chains, n_samples, n_dim, meta_names, meta_shapes)

        # Initialize backend sessions
        if not self._run:
            variables = [
                Variable(
                    f"variable_{i}",
                    np.dtype(float).name,
                    [],
                    [],
                    is_deterministic=False,
                )
                for i in range(n_dim)
            ]

            sample_stats = []
            for name, shape in zip(meta_names, meta_shapes):
                sample_stats.append(
                    Variable(name=name, dtype=np.dtype(float).name, shape=shape)
                )

            run_meta = RunMeta(
                self.run_id,
                variables=variables,
                sample_stats=sample_stats,
            )
            self._run = self._backend.init_run(run_meta)
        self._chains = [self._run.init_chain(chain_number=i) for i in range(n_chains)]

    def record(
        self,
        chain_idx: int,
        state: np.ndarray,
        meta: Dict[str, Union[float, np.ndarray]],
    ) -> None:
        draw = dict(zip([f"variable_{i}" for i in range(self.n_dims)], state))

        self._chains[chain_idx].append(draw, meta)

    def finish(self) -> None:
        pass
