import numpy as np


class NSPD:
    @staticmethod
    def get_NSPD(A):
        assert A.shape[0] == A.shape[1]

        B = (A + A.transpose()) / 2
        U, S, V = np.linalg.svd(B)
        US = np.multiply(U, S)
        H = np.matmul(US, U.transpose())
        A_hat = (B + H) / 2
        A_hat = NSPD.correct_numerics(A_hat)
        return A_hat

    @staticmethod
    def correct_numerics(A_hat):
        mineig = 0
        # p = 1
        k = 0
        # while p != 0:
        while mineig <= 0:
            # R, p = np.linalg.cholesky(A_hat)
            eigenvals = np.linalg.eig(A_hat)[0]
            mineig = np.min(eigenvals)
            k += 1
            # if p != 0:
            if mineig <= 0:

                A_hat = A_hat + (
                    -mineig * k**2
                    + np.spacing(np.min(np.diag(A_hat))) * np.eye(A_hat.shape[0])
                )
        return A_hat
