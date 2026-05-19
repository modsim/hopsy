import copy

import numpy as np
import pandas as pd


class Polytope:
    """
    Polytope class which holds a set of inequality constraints:
    Ax<=b
    and equality constraints:
    Sx=h
    A and S are Pandas Dataframes and b and h are Pandas Series. Column names of A and S are the reaction names
    (preserved). The polytope can be transformed with apply_shift and apply_transformation. Each of these
    transformations applied are stored in the attributes shift and transformation, allowing to back transform points x
    to the original space using the method back_transform.
    """

    def __init__(self, A, b, S=None, h=None):
        self.A = pd.DataFrame(A)
        self.b = pd.Series(b)
        self.A, self.b = Polytope.remove_zero_rows(self.A, self.b)
        self.shift = pd.Series(0, index=self.A.columns)
        self.transformation = pd.DataFrame(
            np.eye(len(self.A.columns)), index=self.A.columns
        )
        # Differentiate the case with equality constraints
        if S is None:
            if h is not None:
                # You can never have a RHS without a LHS
                raise ValueError
            self.inequality_only = True
            self.S = None
            self.h = None
        else:
            self.inequality_only = False
            self.S = pd.DataFrame(S)
            if h is None:
                self.h = pd.Series(np.zeros((S.shape[0])), index=self.S.index)
            else:
                self.h = pd.Series(h)
            self.S, self.h = Polytope.remove_zero_rows(self.S, self.h)

    def border_distance(self, x):
        """
        Computes shortest distance from x to polytope border
        :param x: Numpy array
        :return: Float
        """
        temp = (self.b.values - np.squeeze(np.matmul(self.A.values, x).T)).T
        return np.min(temp)

    def apply_shift(self, shift):
        """
        Shifts Polytope region without altering the shape
        :param shift: Numpy Array
        :return:
        """
        self.shift += np.matmul(
            self.transformation.values, shift.reshape((shift.size,))
        )
        self.b -= np.squeeze(np.matmul(self.A.values, shift))
        # self.center -= shift
        if not self.inequality_only:
            self.h -= np.squeeze(np.matmul(self.S.values, shift))

    def apply_transformation(self, transformation):
        """
        Applies transformation matrix to Polytope. If non-symmetric transformation, assumes reduction to system of only
        inequlity constraints (not checked).
        :param transformation: Numpy Array or Pandas Dataframe
        :return:
        """
        self.transformation = self.transformation.dot(transformation)
        self.A = self.A.dot(transformation)

        # if applying non-symmetric transform, reduce system to only inequality constraints
        if transformation.shape[0] != transformation.shape[1]:
            self.inequality_only = True
            self.S = None
            self.h = None

        if not self.inequality_only:
            self.S = self.S.dot(transformation)

    def back_transform(self, x):
        """
        Backtransforms x to the original space of the Polytope.
        :param x: Numpy array
        :return:
        """
        return (
            np.squeeze(np.matmul(self.transformation.values, x)).T + self.shift.values
        ).T

    def normalize(self):
        """
        Normalizes each row sum of self.A to 1. Does not change feasible space.
        :return:
        """
        self.A, self.b = Polytope.normalize_system(self.A, self.b)
        if not self.inequality_only:
            self.S, self.h = Polytope.normalize_system(self.S, self.h)

    def copy(self):
        """
        Deep copy of Polytope.
        :return: Polytope
        @rtype: object
        """
        return copy.deepcopy(self)

    def __eq__(self, other):
        if isinstance(other, Polytope):
            try:
                attribute_set = set(dir(self)).union(set(dir(other)))
                for attribute in attribute_set:
                    attr1 = getattr(self, attribute)
                    attr2 = getattr(other, attribute)
                    if (
                        isinstance(attr1, pd.DataFrame)
                        and isinstance(attr2, pd.DataFrame)
                    ) or (
                        isinstance(attr1, pd.Series) and isinstance(attr2, pd.Series)
                    ):
                        bool = (attr1 == attr2).all()
                        if bool is False:
                            return False
                return True
            except AttributeError:
                return False

        return False

    @staticmethod
    def remove_zero_rows(S, h):
        row_norm = np.linalg.norm(S, axis=1)
        S = S.iloc[row_norm != 0, :]
        assert (h.iloc[row_norm == 0] == 0).all()
        h = h.iloc[row_norm != 0]
        return S, h

    @staticmethod
    def normalize_system(A, b, row_norm=None):
        if row_norm is None:
            row_norm = np.linalg.norm(A, axis=1)
        row_norm[row_norm == 0] = 1
        A = (A.T / row_norm).T
        b = (b.T / row_norm).T
        return A, b
