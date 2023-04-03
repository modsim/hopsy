import logging
import os
import unittest

import clickhouse_driver
import hagelkorn
import mcbackend as mcb
import numpy

import hopsy

try:
    DB_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
    DB_PASS = os.environ.get("CLICKHOUSE_PASS", "")
    DB_PORT = os.environ.get("CLICKHOUSE_PORT", 9000)
    DB_KWARGS = dict(host=DB_HOST, port=DB_PORT, password=DB_PASS)
    client = clickhouse_driver.Client(**DB_KWARGS)
    client.execute("SHOW DATABASES;")
    HAS_REAL_DB = True
except:
    HAS_REAL_DB = False

_log = logging.getLogger(__file__)


def test_objects():
    A, b = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0, 0]
    model = hopsy.Gaussian(mean=[0, 0, 0])
    problem = hopsy.Problem(A, b, model)

    n_chains = 4
    chains = [
        hopsy.MarkovChain(problem, hopsy.GaussianProposal, starting_point=[0, 0, 0])
        for _ in range(n_chains)
    ]
    for chain in chains:
        chain.proposal.stepsize = 0.2

    rngs = [hopsy.RandomNumberGenerator(seed=42, stream=i) for i in range(n_chains)]

    return chains, rngs


@unittest.skipIf(
    condition=not HAS_REAL_DB,
    reason="Integration tests need a ClickHouse server on localhost:9000 without authentication.",
)
class TestAdapter(unittest.TestCase):
    def setUp(self):
        """Initializes a fresh database just for this test method."""
        self._db = "testing_" + hagelkorn.random()
        self._client_main = clickhouse_driver.Client(**DB_KWARGS)
        self._client_main.execute(f"CREATE DATABASE {self._db};")
        self._client = clickhouse_driver.Client(**DB_KWARGS, database=self._db)
        self.backend = mcb.clickhouse.ClickHouseBackend(self._client)
        return

    def tearDown(self):
        self._client.disconnect()
        self._client_main.execute(f"DROP DATABASE {self._db};")
        self._client_main.disconnect()
        return

    def test_write(self):
        backend = mcb.clickhouse.ClickHouseBackend(self._client)

        # To extract the run meta that the adapter passes to the backend:
        args = []
        original = backend.init_run

        def wrapper(meta: mcb.RunMeta):
            args.append(meta)
            return original(meta)

        backend.init_run = wrapper

        chains, rngs = test_objects()
        trace = hopsy.MCBBackend(backend)
        record_meta = ["state_negative_log_likelihood", "proposal.proposal"]
        meta, samples = hopsy.sample(
            chains,
            rngs,
            n_samples=50,
            thinning=10,
            n_procs=3,
            record_meta=record_meta,
            backend=trace,
        )

        if not len(args) == 1:
            _log.warning("Run was initialized multiple times.")
        rmeta = args[0]

        assert numpy.all(
            [var.name == f"variable_{i}" for i, var in enumerate(rmeta.variables)]
        )
        assert numpy.all(
            [var.name == record_meta[i] for i, var in enumerate(rmeta.sample_stats)]
        )
