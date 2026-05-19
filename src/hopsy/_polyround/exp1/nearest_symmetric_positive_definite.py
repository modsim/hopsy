import cupy as cp


def as_float(value):
    return float(cp.asnumpy(value))


def positive_spacing(x, dtype):
    return cp.finfo(dtype).eps * cp.maximum(1.0, cp.abs(x))


def get_NSPD(A):
    assert A.shape[0] == A.shape[1]

    B = (A + A.transpose()) / 2
    U, S, _ = cp.linalg.svd(B)
    H = (U * S) @ U.transpose()
    A_hat = (B + H) / 2
    return correct_numerics(A_hat)


def correct_numerics(A_hat):
    k = 0
    while True:
        mineig = cp.min(cp.linalg.eigvalsh(A_hat))
        if as_float(mineig) > 0:
            return A_hat
        k += 1
        diag_min = cp.min(cp.diag(A_hat))
        jitter = positive_spacing(diag_min, A_hat.dtype)
        A_hat = A_hat + (-mineig * k**2 + jitter) * cp.eye(
            A_hat.shape[0], dtype=A_hat.dtype
        )
