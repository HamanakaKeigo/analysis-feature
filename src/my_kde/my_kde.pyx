from libc cimport math
cimport cython
cimport numpy as np
from numpy.math cimport PI
from numpy cimport ndarray, float64_t
import numpy as np


ctypedef fused real:
    float
    double
    long double


@cython.wraparound(False)
@cython.boundscheck(False)

def gauss(whitening, points, xi,values, dtype, real _=0):
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
        real[:, :] xi_, values_,points_,estimate
        int i, k
        int n, d, p
        double arg, residual, norm
    

    n = points.shape[0]
    d = points.shape[1]
    p = values.shape[1]
    

    # Rescale the data
    xi_ = np.dot(xi, whitening).astype(dtype, copy=False)
    values_ = np.array(values).astype(dtype, copy=False)
    points_ = np.array(points).astype(dtype, copy=False)

    

    # Evaluate the normalisation
    norm = math.pow((2 * PI) ,<double>-d/2)
    for i in range(d):
        norm *= whitening[i, i]
    

    # Create the result array and evaluate the weighted sum
    estimate = np.zeros((1, p), dtype)
    for i in range(n):
        arg = 0
        for k in range(d):
            residual = (points_[i, k] - xi_[0, k])
            arg += residual * residual
            

        arg = math.exp(-arg / 2) * norm
        for k in range(p):
            estimate[0, k] += values_[i, k] * arg
        

    return np.asarray(estimate)