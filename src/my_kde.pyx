from cpython cimport bool
from libc cimport math
cimport cython
cimport numpy as np
from numpy.math cimport PI
from numpy cimport ndarray, int64_t, float64_t, intp_t
import warnings
import numpy as np
import scipy.stats, scipy.special
cimport scipy.special.cython_special as cs

ctypedef np.float64_t DTYPE

ctypedef fused real:
    float
    double
    long double


@cython.wraparound(False)
@cython.boundscheck(False)

def gauss(whitening, points_, xi_,values_, dtype, real _=0):
    """
    def gaussian_kernel_estimate(points, real[:, :] values, xi, precision)
    Evaluate a multivariate Gaussian kernel estimate.
    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimensions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    precision : array_like with shape (d, d)
        Precision matrix for the Gaussian kernel.
    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    cdef:
        real[:, :] estimate
        int i, j, k
        int n, d, m, p
        real arg, residual, norm

    n = points_.shape[0]
    d = points_.shape[1]
    m = xi_.shape[0]
    p = values_.shape[1]
    print(n,d,m,p)

    # Evaluate the normalisation
    norm = math.pow((2 * PI) ,(- d / 2))
    for i in range(d):
        norm *= whitening[i, i]

    # Create the result array and evaluate the weighted sum
    estimate = np.zeros((m, p), dtype)
    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])
                arg += residual * residual

            arg = math.exp(-arg / 2) * norm
            for k in range(p):
                estimate[j, k] += values_[i, k] * arg

    return np.asarray(estimate)