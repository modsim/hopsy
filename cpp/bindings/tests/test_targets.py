import gc
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import _capsule_gaussian as native
import _hopsy_core as hopsy
import numpy as np
import pytest

P = np.array([[2.0, 1.0], [1.0, 3.0]])
x = np.array([0.4, -0.7])


def chain(target, gradient=False):
    g = hopsy.LinearPolytope(np.vstack([np.eye(2), -np.eye(2)]), np.ones(4) * 10)
    problem = hopsy.Problem(g, target)
    kernel = (hopsy.CSmMALAKernel if gradient else hopsy.GaussianRandomWalkKernel)(
        np.zeros(2), 0.2
    )
    return hopsy.bind_chain(problem, kernel, np.zeros(2))


class Spy(hopsy.PythonTarget):
    def __init__(self):
        super().__init__()
        self.calls = [0, 0, 0]

    def log_density(self, x):
        self.calls[0] += 1
        return -0.5 * x @ P @ x

    def log_gradient(self, x):
        self.calls[1] += 1
        return -P @ x

    def log_hessian(self, x):
        self.calls[2] += 1
        return -P


def test_dispatch_sampling_and_lifetime():
    target = Spy()
    result = hopsy.debug_target(target, x, 2)
    assert target.calls == [1, 1, 1]
    assert result["log_density"] == pytest.approx(-0.5 * x @ P @ x)
    np.testing.assert_allclose(result["log_gradient"], -P @ x)
    np.testing.assert_array_equal(result["log_hessian"], -P)
    c = chain(target, True)
    before = target.calls.copy()
    c.draw(hopsy.RandomNumberGenerator(1), 20)
    assert target.calls[0] > before[0] and target.calls[1] > before[1]
    ref = weakref.ref(target)
    del target
    gc.collect()
    assert ref() is not None
    del c
    gc.collect()
    assert ref() is None


class Density(hopsy.PythonTarget):
    def log_density(self, x):
        return -x @ x


def test_python_capabilities_errors():
    with pytest.raises(TypeError, match="log_density"):
        chain(hopsy.PythonTarget())
    chain(Density()).draw(hopsy.RandomNumberGenerator(2), 5)
    with pytest.raises(TypeError, match="log_gradient"):
        chain(Density(), True)
    with pytest.raises(ValueError, match="order"):
        hopsy.debug_target(Density(), x, 1)

    class Bad(Spy):
        def log_gradient(self, x):
            return np.zeros(3)

    with pytest.raises(ValueError, match="dimension"):
        hopsy.debug_target(Bad(), x, 1)

    class BadH(Spy):
        def log_hessian(self, x):
            return np.eye(3)

    with pytest.raises(ValueError, match="dimension"):
        hopsy.debug_target(BadH(), x, 2)

    class Fails(Density):
        def log_density(self, x):
            raise RuntimeError("callback sentinel")

    with pytest.raises(RuntimeError, match="callback sentinel"):
        chain(Fails()).draw(hopsy.RandomNumberGenerator(2), 1)

    class HessianOnly(Density):
        def log_hessian(self, x):
            return -P

    with pytest.raises(TypeError, match="gradient"):
        chain(HessianOnly())


@pytest.mark.parametrize("order", [0, 1, 2])
def test_native_capabilities(order):
    target = hopsy.CapsuleTarget(native.make(order))
    for _ in range(20):
        result = hopsy.debug_target(target, x, order)
        assert result["log_density"] == pytest.approx(-0.5 * x @ P @ x)
        if order >= 1:
            np.testing.assert_allclose(result["log_gradient"], -P @ x)
        if order == 2:
            np.testing.assert_array_equal(result["log_hessian"], -P)
    c = chain(target, order >= 1)
    c.draw(hopsy.RandomNumberGenerator(2), 20)
    if not order:
        with pytest.raises(TypeError, match="gradient"):
            chain(target, True)
    with pytest.raises(ValueError, match="dimension"):
        hopsy.debug_target(target, np.zeros(3))


@pytest.mark.parametrize("mode", range(2, 9))
def test_invalid_capsules(mode):
    with pytest.raises(ValueError):
        hopsy.CapsuleTarget(native.make(2, mode))


def test_failure_cleanup_parallel():
    with pytest.raises(RuntimeError, match="17"):
        hopsy.debug_target(hopsy.CapsuleTarget(native.make(2, 1)), x)
    gc.collect()
    before = native.destroyed()
    capsule = native.make()
    target = hopsy.CapsuleTarget(capsule)
    c = chain(target)
    del capsule, target
    gc.collect()
    assert native.destroyed() == before
    c.draw(hopsy.RandomNumberGenerator(1), 5)
    del c
    gc.collect()
    assert native.destroyed() == before + 1
    target = hopsy.CapsuleTarget(native.make())
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda _: chain(target, True).draw(hopsy.RandomNumberGenerator(3), 100),
                range(8),
            )
        )
    for result in results:
        np.testing.assert_array_equal(result, results[0])


@pytest.mark.parametrize("sampling", [False, True])
def test_native_releases_gil(sampling):
    native.reset()
    target = hopsy.CapsuleTarget(native.make(2, 9))
    with ThreadPoolExecutor(1) as pool:
        if sampling:
            c = chain(target)
            future = pool.submit(c.draw, hopsy.RandomNumberGenerator(2), 1)
        else:
            future = pool.submit(hopsy.debug_target, target, x)
        deadline = time.monotonic() + 3
        while not native.started() and time.monotonic() < deadline:
            time.sleep(0.001)
        try:
            assert native.started()
        finally:
            native.release()
        result = future.result(timeout=3)
        if not sampling:
            assert result["log_density"] == pytest.approx(-0.5 * x @ P @ x)


def test_direct_native_methods_and_affine_sampling():
    target = hopsy.CapsuleTarget(native.make())
    assert target.dimension == 2
    assert target.log_density(x) == pytest.approx(-0.5 * x @ P @ x)
    np.testing.assert_allclose(target.log_gradient(x), -P @ x)
    np.testing.assert_array_equal(target.log_hessian(x), -P)
    g = hopsy.LinearPolytope(np.vstack([np.eye(2), -np.eye(2)]), np.ones(4) * 3)
    coordinates = hopsy.AffineCoordinates(
        np.array([[1.0, 0.3], [0.0, 2.0]]), np.array([0.2, -0.1])
    )
    outputs = []
    for t in [target, Spy()]:
        p = hopsy.Problem(g, t, coordinates)
        k = hopsy.CSmMALAKernel(np.zeros(2), 0.1)
        c = hopsy.bind_chain(p, k, np.zeros(2))
        outputs.append(c.draw(hopsy.RandomNumberGenerator(3), 100))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-14)
    with pytest.raises(ValueError, match="dimension"):
        hopsy.Problem(hopsy.LinearPolytope(np.eye(3), np.ones(3)), target)
    density_only = hopsy.CapsuleTarget(native.make(0))
    with pytest.raises(ValueError, match="order"):
        density_only.log_gradient(x)


def test_python_gradient_only_and_callback_failures():
    class Gradient(Density):
        def log_gradient(self, x):
            return -2 * x

    chain(Gradient(), True).draw(hopsy.RandomNumberGenerator(2), 10)

    class Fails(Spy):
        def log_gradient(self, x):
            raise LookupError("gradient sentinel")

    with pytest.raises(LookupError, match="gradient sentinel"):
        chain(Fails(), True).step(hopsy.RandomNumberGenerator(2))

    class Noncallable(Density):
        log_gradient = 3

    with pytest.raises(TypeError, match="callable"):
        chain(Noncallable())


@pytest.mark.parametrize("mode,order,status", [(10, 1, 19), (11, 2, 20)])
def test_native_derivative_failures(mode, order, status):
    with pytest.raises(RuntimeError, match=str(status)):
        hopsy.debug_target(hopsy.CapsuleTarget(native.make(2, mode)), x, order)


def test_capsule_type_and_input_validation():
    with pytest.raises(ValueError, match="capsule"):
        hopsy.CapsuleTarget(object())
    target = hopsy.CapsuleTarget(native.make())
    for value in [np.zeros(0), np.array([np.nan, 1.0])]:
        with pytest.raises(ValueError, match="dimension|nonfinite"):
            hopsy.debug_target(target, value)


@pytest.mark.parametrize(
    "method,value,order,message",
    [
        ("log_density", np.zeros(2), 0, "scalar"),
        ("log_gradient", np.zeros((2, 2)), 1, "vector"),
        ("log_hessian", "invalid", 2, "matrix"),
    ],
)
def test_python_invalid_return_types(method, value, order, message):
    target_type = type("InvalidReturn", (Spy,), {method: lambda self, x: value})
    with pytest.raises(TypeError, match=message):
        hopsy.debug_target(target_type(), x, order)
