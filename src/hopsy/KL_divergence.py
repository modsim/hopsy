from scipy.spatial import cKDTree as KDTree
import numpy as np



def _multiplicity_in_self(a):
    """Count how many times each line appears in a (including itself)."""
    a = np.ascontiguousarray(a)
    keys = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return cnt[inv]


def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)
  if np.array_equal(x,y):
    return 0


  mA = _multiplicity_in_self(x)  # Frequency of each point in x (inkl. self)
  idx_r = np.minimum(1 + mA - 1, n - 1)  # next neares neighbour = self + skip duplicates
  k_max = int(idx_r.max())

  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)


  # Get the first k_max+1 nearest neighbours for x, since the first k_max ones are the
  # sample itself.
  r = xtree.query(x, k=k_max+1, eps=0.0, p=2)[0][:,k_max]
  s = ytree.query(x, k=k_max, eps=.0, p=2)[0][:,k_max-1]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))