import sys
import numpy as np
import scipy as sp
import scipy.special
from sklearn.utils import resample
import operator as op
from functools import reduce
import itertools


def ctransform(x):
    """Copula transformation (empirical CDF)

    Parameters
    ----------
    x : numpy.ndarray
        Data to be transformed.

    Returns
    -------
    numpy.ndarray
        The empirical CDF value along the first axis of x.
        Data is ranked and scaled within [0 1] (open interval).
    """

    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr + 1).astype(np.float64) / (xr.shape[-1] + 1)
    return cx


def copnorm(x):
    """Copula normalization

    Parameters
    ----------
    x : numpy.ndarray
        Input data. If x>2D normalization is performed on each
        dimension separately.

    Returns
    -------
    numpy.ndarray
        Standard normal samples with rank ordering preserved.
        Operates along the last axis.
    """

    cx = sp.special.ndtri(ctransform(x))
    return cx


def ent_g(data, biascorrect=True, base=None):
    """Entropy of a continuous variable, with gaussian copula semi-parametric
    estimation

    Parameters
    ----------
    x : numpy.ndarray
        continuous variable with one or more dimensions. Columns of
        x correspond to samples, rows to dimensions/variables. (Samples last axis)
    biascorrect : bool (default=True)
        Specifies whether bias correction should be applied to
        computation of the entropy.
    base : float, int or None
        The logarithmic base to use for calculating the entropy.
        If None, the natural logarithm is used.
        Default is None.

    Returns
    -------
    float
        Entropy of the input continuous variable with Gaussian copula estimation
        and bias correction.
    """
    data = np.atleast_2d(data)
    if data.ndim > 2:
        raise ValueError("x must be at most 2d")

    data1 = data.copy()
    x = copnorm(data1)
    x = x - x.mean(axis=1)[:, np.newaxis]

    # Ntrl = x.shape[1]
    nvarx, ntrl = x.shape
    ln2 = np.log(2)

    # demean data

    Cov = np.dot(x, x.T) / float(ntrl - 1)
    # submatrices of joint covariance

    chc = np.linalg.cholesky(Cov)

    # entropies in nats
    # normalizations cancel for cmi
    hx = np.sum(np.log(np.diagonal(chc))) + 0.5 * nvarx * (np.log(2 * np.pi) + 1.0)

    if biascorrect:
        psiterms = (
            sp.special.psi((ntrl - np.arange(1, nvarx + 1).astype(float)) / 2.0) / 2.0
        )
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms.sum()

    if base is not None:
        hx /= np.log(base)

    return hx


def lin_ent(x, biascorrect=True, base=None):
    """Compute the entropy of a multivariate Gaussian variable X.

    Parameters
    ----------
    X : numpy.ndarray
        Gaussian variable with one or more dimensions. Each row
        represents a different dimension, and each column represents a
        different sample.
    biascorrect : bool (default=True)
        Specifies whether bias correction should be applied to
        computation of the entropy.
    base : float, int or None
        The logarithmic base to use for calculating the entropy.
        If None, the natural logarithm is used.
        Default is None.

    Returns
    -------
    float
        The entropy of the multivariate Gaussian variable X.
    """

    # X is of shape (num var, num timepoints)
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    nvarx, ntrl = x.shape
    ln2 = np.log(2)

    # demean data
    x = x - x.mean(axis=1)[:, np.newaxis]

    Cov = np.dot(x, x.T) / float(ntrl - 1)
    # submatrices of joint covariance

    chc = np.linalg.cholesky(Cov)

    # entropies in nats
    # normalizations cancel for cmi
    hx = np.sum(np.log(np.diagonal(chc))) + 0.5 * nvarx * (np.log(2 * np.pi) + 1.0)

    if biascorrect:
        psiterms = (
            sp.special.psi((ntrl - np.arange(1, nvarx + 1).astype(float)) / 2.0) / 2.0
        )
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms.sum()

    if base is not None:
        hx /= np.log(base)

    return hx


def bootci(nboot, info_func, xsamp_range, alpha):
    """Compute the bootstrap confidence interval of a statistic.

    Parameters
    ----------
    nboot : int
        Number of bootstrap samples to generate.
    info_func : function
        Function to apply to the bootstrapped samples.
    xsamp_range : numpy.ndarray
        Range of values to generate bootstrap samples from.
    alpha : float
        The significance level of the confidence interval.

    Returns
    -------
    tuple
        The lower and upper bounds of the confidence interval.
    """

    stats = list()
    for i in range(nboot):
        xsamp = resample(xsamp_range, n_samples=len(xsamp_range))
        info = info_func(xsamp)
        stats.append(info)
    # confidence intervals
    p = ((alpha) / 2.0) * 100
    lower = np.percentile(stats, p)
    p = (1 - (alpha) / 2.0) * 100
    upper = np.percentile(stats, p)
    return lower, upper


def ncr(n, r):
    """Calculate the number of possible combinations (n choose r).

    Parameters
    ----------
    n : int
        The total number of items.
    r : int
        The number of items to be selected.

    Returns
    -------
    int
        The number of possible combinations.

    """
    if n < r:
        return 0
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


class combinations_manager:
    """A class to manage combinations of N choose K."""

    def __init__(self, N, K):
        """
        Parameters
        ----------
        N : int
            The total number of items.
        K : int
            The number of items to be selected.

        """
        if K > N:
            print("error: K can't be greater than N in N choose K")
            sys.exit()
        self.N = N
        self.K = K
        self.lim = K
        self.inc = 1
        if K > N / 2:
            WV = N - K
        else:
            WV = K

        self.BC = ncr(N, WV) - 1
        self.CNT = 0
        self.WV = []

    def nextchoose(self):
        """Generate the next combination.

        Returns
        -------
        list
            The next combination.

        """
        if self.CNT == 0 or self.K == self.N:
            self.WV = np.arange(1, self.K + 1)
            self.B = self.WV
            self.CNT += 1
            return self.B

        if self.CNT == self.BC:
            self.B = np.arange(self.N - self.K + 1, self.N + 1)
            self.CNT = 0
            self.inc = 1
            self.lim = self.K
            return self.B

        for jj in range(self.inc):
            self.WV[self.K + jj - self.inc] = self.lim + jj + 1

        if self.lim < (self.N - self.inc):
            self.inc = 0

        self.inc += 1
        self.lim = self.WV[self.K - self.inc]
        self.CNT += 1
        self.B = self.WV
        return self.B

    def combination2number(self, comb):
        """Convert a combination to a number.

        Parameters
        ----------
        comb : list
            The combination to convert.

        Returns
        -------
        num : int
            The number corresponding to the combination.

        """
        num = 0
        k = len(comb)
        for i in range(1, k + 1):
            c = comb[i - 1] - 1
            num += ncr(c, i)
        return num

    def number2combination(self, num):
        """Convert a number to a combination.

        Parameters
        ----------
        num : int
            The number to convert.

        Returns
        -------
        comb : list
            The combination corresponding to the number.

        """
        comb = []
        k = self.K
        num_red = num
        while k > 0:
            m = k - 1
            while True:
                mCk = ncr(m, k)
                if mCk > num_red:
                    break
                if comb.count(m) > 0:
                    break
                m += 1
            comb.append(m)
            num_red -= ncr(m - 1, k)
            k -= 1
        comb.reverse()
        comb = np.array(comb)
        return comb


def get_ent(X, estimator, biascorrect=True, base=None):
    """Compute the entropy of a multivariate variable X.

    Parameters
    ----------
    X : numpy.ndarray
        Variable with one or more dimensions. Each row represents
        a different dimension, and each column represents a different sample.
    estimator : str
        The estimator to use for entropy estimation.
        Options are "lin_est", "gcmi", and "cat_ent".
    biascorrect : bool (default=True)
        Specifies whether bias correction should be applied to
        computation of the entropy. Is applied only for the "lin_est"
        and the "gcmi" estimator.
    base : float, int or None
        The logarithmic base to use for calculating the entropy.
        If None, the natural logarithm is used.
        Default is None.

    Returns
    -------
    entropy : float
        The entropy of the multivariate variable X.
    """

    if estimator == "lin_est":
        entropy = lin_ent(X, biascorrect=biascorrect, base=base)
    elif estimator == "gcmi":
        entropy = ent_g(X, biascorrect=biascorrect, base=base)
    elif estimator == "cat_ent":
        entropy = get_entropy_scipy(X, base=base)

    else:
        print(
            "Please use estimator out of the following - 'lin_est', 'gcmi' or 'cat_ent'"
        )
        sys.exit()
    return entropy


def generate_components(input_set):
    subsets = list(itertools.combinations(input_set, len(input_set) - 1))
    return subsets


def check_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return start1 <= end2 and start2 <= end1


# utils for dO-info with gaussian copula estimation


def cmi_ggg(x, y, z, biascorrect=True, demeaned=False):
    """Conditional Mutual information (CMI) between two Gaussian variables
    conditioned on a third

    Parameters
    ----------
    x : numpy.ndarray
        First variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    y : numpy.ndarray
        Second variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    z : numpy.ndarray
        Conditioning variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    biascorrect : bool (default=True)
        Specifies whether bias correction should be applied to
        computation of the entropy.
    demeaned : bool (default=False)
        Specifies whether input data already has zero mean or whether it
        should be subtracted from the data prior to computation.
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarz = z.shape[0]
    Nvaryz = Nvary + Nvarz
    Nvarxy = Nvarx + Nvary
    Nvarxz = Nvarx + Nvarz
    Nvarxyz = Nvarx + Nvaryz

    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xyz = np.vstack((x, y, z))
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1)[:, np.newaxis]
    Cxyz = np.dot(xyz, xyz.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cz = Cxyz[Nvarxy:, Nvarxy:]
    Cyz = Cxyz[Nvarx:, Nvarx:]
    Cxz = np.zeros((Nvarxz, Nvarxz))
    Cxz[:Nvarx, :Nvarx] = Cxyz[:Nvarx, :Nvarx]
    Cxz[:Nvarx, Nvarx:] = Cxyz[:Nvarx, Nvarxy:]
    Cxz[Nvarx:, :Nvarx] = Cxyz[Nvarxy:, :Nvarx]
    Cxz[Nvarx:, Nvarx:] = Cxyz[Nvarxy:, Nvarxy:]

    chCz = np.linalg.cholesky(Cz)
    chCxz = np.linalg.cholesky(Cxz)
    chCyz = np.linalg.cholesky(Cyz)
    chCxyz = np.linalg.cholesky(Cxyz)

    # entropies in nats
    # normalizations cancel for cmi
    HZ = np.sum(np.log(np.diagonal(chCz)))  # + 0.5*Nvarz*(np.log(2*np.pi)+1.0)
    HXZ = np.sum(np.log(np.diagonal(chCxz)))  # + 0.5*Nvarxz*(np.log(2*np.pi)+1.0)
    HYZ = np.sum(np.log(np.diagonal(chCyz)))  # + 0.5*Nvaryz*(np.log(2*np.pi)+1.0)
    HXYZ = np.sum(np.log(np.diagonal(chCxyz)))  # + 0.5*Nvarxyz*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = (
            sp.special.psi((Ntrl - np.arange(1, Nvarxyz + 1)).astype(np.float64) / 2.0)
            / 2.0
        )
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HZ = HZ - Nvarz * dterm - psiterms[:Nvarz].sum()
        HXZ = HXZ - Nvarxz * dterm - psiterms[:Nvarxz].sum()
        HYZ = HYZ - Nvaryz * dterm - psiterms[:Nvaryz].sum()
        HXYZ = HXYZ - Nvarxyz * dterm - psiterms[:Nvarxyz].sum()

    # MI in bits
    MI = (HXZ + HYZ - HXYZ - HZ) / ln2
    return MI


def gccmi_ccc_nocopnorm(x, y, z):
    """Conditional Mutual information (CMI) between two Gaussian variables
    conditioned on a third, without copula normalization.

    Parameters
    ----------
    x : numpy.ndarray
        First variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    y : numpy.ndarray
        Second variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    z : numpy.ndarray
        Conditioning variable samples, rows correspond to dimensions/variables.
        (Samples first axis)

    Returns
    -------
    CMI : float
        The conditional mutual information between x and y conditioned on z.

    Notes
    -----
    This function does not perform copula normalization.

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")

    Ntrl = x.shape[1]

    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")
    CMI = cmi_ggg(x, y, z, True, True)
    return CMI


def get_cmi(x, y, z, estimator, biascorrect=True, base=None):
    """Conditional Mutual Information between two variables
    conditioned on a third.

    Parameters
    ----------
    x : numpy.ndarray
        First variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    y : numpy.ndarray
        Second variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    z : numpy.ndarray
        Conditioning variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    estimator : str
        The estimator to use for entropy estimation.
        Options are "lin_est", "gcmi", and "cat_ent".
    biascorrect : bool (default=True)
        Specifies whether bias correction should be applied to
        computation of the entropy. Is applied only for the "lin_est"
        and the "gcmi" estimator.
    base : float, int or None
        The logarithmic base to use for calculating the entropy.
        If None, the natural logarithm is used.
        Default is None.

    Returns
    -------
    MI : float
        The conditional mutual information between x and y conditioned on z.
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")

    # joint variable
    xyz = np.vstack((x, y, z))
    xz = np.vstack((x, z))
    yz = np.vstack((y, z))

    xyz = xyz - xyz.mean(axis=1)[:, np.newaxis]
    xz = xz - xz.mean(axis=1)[:, np.newaxis]
    yz = yz - yz.mean(axis=1)[:, np.newaxis]

    # entropies in nats
    HZ = get_ent(z, estimator, biascorrect=biascorrect, base=base)
    HXZ = get_ent(xz, estimator, biascorrect=biascorrect, base=base)
    HYZ = get_ent(yz, estimator, biascorrect=biascorrect, base=base)
    HXYZ = get_ent(xyz, estimator, biascorrect=biascorrect, base=base)

    # MI in bits
    MI = HXZ + HYZ - HXYZ - HZ
    return MI


def lin_CE(Yb, Z):
    """Linear conditional entropy

    Parameters
    ----------
    Yb : numpy.ndarray
        Variable samples, rows correspond to dimensions/variables.
        (Samples first axis)
    Z : numpy.ndarray
        Conditioning variable samples, rows correspond to dimensions/variables.
        (Samples first axis)

    Returns
    -------
    ce : float
        The conditional entropy between Yb and Z.
    """

    # Yb (output), Z (input) are of shape (num timepoints, num variables)
    Am = np.linalg.lstsq(Z, Yb, rcond=None)[0]
    Yp = Z @ Am
    Up = Yb - Yp
    S = np.cov(Up.T)
    if S.ndim == 0:
        S = np.var(Up.T)
        detS = S
    else:
        detS = np.linalg.det(S)
    N = Yb.shape[1]
    ce = 0.5 * np.log(detS) + 0.5 * N * np.log(2 * np.pi * np.exp(1))
    return ce


def lin_cmi_ccc(Y, X0, Y0):
    H_Y_Y0 = lin_CE(Y, Y0)
    X0Y0 = np.concatenate((X0, Y0), axis=1)
    H_Y_X0Y0 = lin_CE(Y, X0Y0)
    cmi = H_Y_Y0 - H_Y_X0Y0
    # print(cmi, H_Y_Y0, H_Y_X0Y0)
    return cmi


# utils for categorical O-info (exact calculation) and gaussian
# based estimantion of the O-info


def get_entropy_scipy(X, base=None, axis=1):
    """
    Compute the entropy of a multivariate variable X.

    Parameters
    ----------
    X : numpy.ndarray
        Categorical variable with one or more dimensions.
        Each row represents a different dimension, and each column
        represents a different sample.
    base : float, int or None
        The logarithmic base to use for calculating the entropy.
        If None, the natural logarithm is used.
        Default is None.
    axis : int
        The axis along which the entropy is calculated.
        Default is 1.

    Returns
    -------
    ent : float
        The entropy of the multivariate categorical variable X.
        The entropy is estimated using the function scipy.stats.entropy()
    """

    if len(X.shape) == 1:
        ent = sp.stats.entropy(np.unique(X, return_counts=True)[1], base=base)
    else:
        ent = sp.stats.entropy(
            np.unique(X, return_counts=True, axis=axis)[1], base=base
        )

    return ent
