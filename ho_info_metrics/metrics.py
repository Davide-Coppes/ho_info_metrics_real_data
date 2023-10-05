import numpy as np
import itertools

from ho_info_metrics.utils import get_ent, get_cmi, copnorm


def tc_boot(
    data, estimator="lin_est", indvar=None, indsample=None, biascorrect=True, base=None
):
    """Calculates the total correlation (TC) of a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    tc : float
        The total correlation (TC) value.

    Notes
    -----
    The total correlation (TC) is a measure of the dependence among the variables
    in the dataset, taking into account both the individual entropies and the
    joint entropies of all subsets of variables.
    It is computed by subtracting the joint entropy of all variables from the
    sum of the individual entropies of each variable excluding itself.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    # here we perform the copnorm only once, to speed up the code.
    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    M, N = X.shape
    tc = -get_ent(X, estimator, biascorrect=biascorrect, base=base)
    for j in range(M):
        tc = tc + get_ent(X[j, :], estimator, biascorrect=biascorrect, base=base)
    return tc


def dtc_boot(
    data, estimator="lin_est", indvar=None, indsample=None, biascorrect=True, base=None
):
    """Calculates the dual total correlation (DTC) of a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    dtc : float
        The dual total correlation (DTC) value.

    Notes
    -----
    The dual total correlation (DTC) is a measure of the dependence among the variables
    in the dataset, taking into account both the individual entropies and the
    joint entropies of all subsets of variables.
    It is computed by subtracting the joint entropy of all variables excluding
    itself from the sum of the individual entropies of each variable excluding itself.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    M, N = X.shape
    dtc = (-M + 1) * get_ent(X, estimator, biascorrect=biascorrect, base=base)
    for j in range(M):
        X1 = np.delete(X, j, axis=0)
        dtc = dtc + get_ent(X1, estimator, biascorrect=biascorrect, base=base)
    return dtc


def o_information_boot(
    data, estimator="lin_est", indvar=None, indsample=None, biascorrect=True, base=None
):
    """Calculates the O-information of a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    o : float
        The O-information value.

    Notes
    -----
    The O-information measures the amount of information shared among all subsets
    of variables.
    It is computed by summing the entropies of each variable and each subset
    minus the joint entropy of the subsets.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    M, N = X.shape
    o = (M - 2) * get_ent(X, estimator, biascorrect=biascorrect, base=base)

    for j in range(M):
        X1 = np.delete(X, j, axis=0)
        o = (
            o
            + get_ent(X[j, :], estimator, biascorrect=biascorrect, base=base)
            - get_ent(X1, estimator, biascorrect=biascorrect, base=base)
        )
    return o


def o_information_gradient_boot(
    data,
    indices,
    estimator="lin_est",
    indvar=None,
    indsample=None,
    biascorrect=True,
    base=None,
):
    """Calculates the O-information gradient of a subset of variables in a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    indices : list
        A list of indices specifying the variables to consider.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    og : float
        The O-information gradient value.

    Notes
    -----
    The O-information gradient of a subset of variables measures the change
    in O-information when that subset is included.
    It is computed by subtracting the O-information of the subset from the
    O-information of the entire dataset.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    og = o_information_boot(
        X, estimator, indvar=None, indsample=None, biascorrect=biascorrect, base=base
    )
    for k in range(len(indices)):
        for combo in itertools.combinations(indices, k + 1):
            X1 = np.delete(X, combo, axis=0)
            og += ((-1) ** (k + 1)) * o_information_boot(
                X1,
                estimator,
                indvar=None,
                indsample=None,
                biascorrect=biascorrect,
                base=base,
            )

    return og


def o_information_gradient_order_1(
    data, estimator="lin_est", indvar=None, indsample=None, biascorrect=True, base=None
):
    """Computes the O-information gradient of all variables in X.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    og_order_1 : numpy.ndarray
        The O-information gradient values for each variable.

    Notes
    -----
    The O-information gradient of a variable measures the change in O-information
    when that variable is included.
    This function computes the O-information gradient for all variables in X.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    og_order_1 = []

    o_total = o_information_boot(
        X,
        estimator=estimator,
        indvar=None,
        indsample=None,
        biascorrect=biascorrect,
        base=base,
    )
    for i in range(len(X[:, 0])):
        og_order_1.append(
            o_total
            - o_information_boot(
                np.delete(X, i, axis=0),
                estimator=estimator,
                indvar=None,
                indsample=None,
                biascorrect=biascorrect,
                base=base,
            )
        )

    return og_order_1


def o_information_gradient_order_2(
    data, estimator="lin_est", indvar=None, indsample=None, biascorrect=True, base=None
):
    """Computes the O-information gradient of all pairs of variables in X.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    og_order_2 : numpy.ndarray
        The O-information gradient values for each pair of variables.

    Notes
    -----
    The O-information gradient of a pair of variables measures the change
    in O-information when those variables are included together.
    This function computes the O-information gradient for all pairs
    of variables in X.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    n_r = len(X)
    og_order_2 = np.zeros((n_r, n_r))

    o_total = o_information_boot(
        X,
        estimator=estimator,
        indvar=None,
        indsample=None,
        biascorrect=biascorrect,
        base=base,
    )

    og_order_1 = []

    for i in range(n_r):
        og_order_1.append(
            o_information_boot(
                np.delete(X, i, axis=0),
                estimator=estimator,
                indvar=None,
                indsample=None,
                biascorrect=biascorrect,
                base=base,
            )
        )

    for i in range(n_r):
        for j in range(i):
            og_order_2[i, j] = (
                o_total
                - og_order_1[i]
                - og_order_1[j]
                + o_information_boot(
                    np.delete(X, [i, j], axis=0),
                    estimator=estimator,
                    indvar=None,
                    indsample=None,
                    biascorrect=biascorrect,
                    base=base,
                )
            )

    return og_order_2 + og_order_2.T


def mi(x, y, estimator="lin_est", indsample=None, biascorrect=True, base=None):
    """Computes the mutual information between two variables.

    Parameters
    ----------
    x : numpy.ndarray
        The input data array for the first variable.
    y : numpy.ndarray
        The input data array for the second variable.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    mutual_info : float
        The mutual information value between the two variables.

    Notes
    -----
    Mutual information measures the amount of information shared between two variables.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    if indsample is not None:
        x = x[:, indsample]
        y = y[:, indsample]

    if estimator == "gcmi":

        x = copnorm(x)
        y = copnorm(y)
        estimator = "lin_est"

    xy = np.vstack((x, y))

    mutual_info = (
        get_ent(y, estimator, biascorrect=biascorrect, base=base)
        + get_ent(x, estimator, biascorrect=biascorrect, base=base)
        - get_ent(xy, estimator, biascorrect=biascorrect, base=base)
    )

    return mutual_info


def mi_lagged(
    x, y, m=1, estimator="lin_est", indsample=None, biascorrect=True, base=None
):
    """Computes the time-delayed mutual information between two variables.

    Parameters
    ----------
    x : numpy.ndarray
        The input data array for the first variable.
    y : numpy.ndarray
        The input data array for the second variable.
    m : int, optional
        The time delay between the variables. Default is 1.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    t_d_mi : float
        The time-delayed mutual information value between the two variables.

    Notes
    -----
    Time-delayed mutual information measures the amount of information shared
    between two variables at a given time delay.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    t_d_mi = mi(
        x[:, :-m],
        y[:, m:],
        estimator,
        indsample=indsample,
        biascorrect=biascorrect,
        base=base,
    )

    return t_d_mi


def redundancy_matrix(
    data,
    m=1,
    estimator="lin_est",
    indvar=None,
    indsample=None,
    biascorrect=True,
    base=None,
):
    """Compute the redundancy matrix for all pairs of variables in the system.

    Parameters
    ----------
    data : numpy.ndarray
        The input data matrix of shape (nvar, ntrl),
        where nvar is the number of variables and ntrl is the number of trials.
    m : int, optional
        The time delay between the variables. Default is 1.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    red_m : numpy.ndarray
        The redundancy matrix of shape (nvar, nvar), representing
        the redundancy atom of the Integrated information decomposition.

    Notes
    -----
    The redundancy atom represents the minimum of the individual time-delayed
    mutual informations of a pair of variables.
    It is computed by subtracting the maximum individual time-delayed mutual
    information from the total time-delayed mutual information of the pair.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    nvar, ntrl = X.shape

    red_m = np.zeros((nvar, nvar))
    for i in range(nvar):
        for j in range(i):
            x = X[i, :]
            y = X[j, :]
            red_m[i, j] = min(
                mi_lagged(
                    x, y, m=m, estimator=estimator, biascorrect=biascorrect, base=base
                ),
                mi_lagged(
                    x, x, m=m, estimator=estimator, biascorrect=biascorrect, base=base
                ),
                mi_lagged(
                    y, x, m=m, estimator=estimator, biascorrect=biascorrect, base=base
                ),
                mi_lagged(
                    y, y, m=m, estimator=estimator, biascorrect=biascorrect, base=base
                ),
            )

    return red_m + red_m.T


def synergy_matrix(
    data,
    m=1,
    estimator="lin_est",
    indvar=None,
    indsample=None,
    biascorrect=True,
    base=None,
):
    """Computes the synergy matrix for all pairs of variables in the system.

    Parameters
    ----------
    data : numpy.ndarray
        The input data matrix of shape (nvar, ntrl),
        where nvar is the number of variables and ntrl is the number of trials.
    m : int, optional
        The time delay between the variables. Default is 1.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    syn_m : numpy.ndarray
        The synergy matrix of shape (nvar, nvar), representing
        the synergy atom of the Integrated information decomposition.

    Notes
    -----
    The synergy atom represents the maximum of the individual time-delayed
    mutual informations of a pair of variables.
    It is computed by subtracting the minimum individual time-delayed mutual
    information from the total time-delayed mutual information of the pair.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.
    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    nvar, ntrl = data.shape

    syn_m = np.zeros((nvar, nvar))
    for i in range(nvar):
        for j in range(i):
            x = X[i, :]
            y = X[j, :]
            xy = np.vstack((x, y))
            mi_1 = mi_lagged(
                x, xy, m=m, estimator=estimator, biascorrect=biascorrect, base=base
            )
            mi_2 = mi_lagged(
                xy, y, m=m, estimator=estimator, biascorrect=biascorrect, base=base
            )
            mi_3 = mi_lagged(
                y, xy, m=m, estimator=estimator, biascorrect=biascorrect, base=base
            )
            mi_4 = mi_lagged(
                xy, y, m=m, estimator=estimator, biascorrect=biascorrect, base=base
            )
            syn_m[i, j] = mi_lagged(
                xy, xy, m=m, estimator=estimator, biascorrect=biascorrect, base=base
            ) - max(mi_1, mi_2, mi_3, mi_4)
    return syn_m + syn_m.T


def s_information_boot(
    data, estimator="lin_est", indvar=None, indsample=None, biascorrect=True, base=None
):
    """Computes the S-information of a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.
        If None, the natural logarithm is used. Default is None.

    Returns
    -------
    s : float
        The S-information value.

    Notes
    -----
    The S-information measures the amount of information shared among all subsets
    of variables.
    It is computed by summing the entropies of each variable and each subset
    minus the joint entropy of the subsets.
    If 'indvar' is provided, only the specified variables are used for computation.
    If 'indsample' is provided, only the specified samples are used for computation.

    """

    if indvar is not None:
        data = data[indvar, :]

    if indsample is not None:
        data = data[:, indsample]

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    if indsample is not None:
        X = X[:, indsample]
    M, N = X.shape
    s = -M * get_ent(X, estimator, biascorrect=biascorrect, base=base)

    for j in range(M):
        X1 = np.delete(X, j, axis=0)
        s = (
            s
            + get_ent(X[j, :], estimator, biascorrect=biascorrect, base=base)
            + get_ent(X1, estimator, biascorrect=biascorrect, base=base)
        )  # check this expression
    return s


def redundancy_mmi(
    sources,
    target,
    estimator="lin_est",
    indvar_sources=None,
    indvar_target=None,
    indsample=None,
    biascorrect=True,
    base=None,
):
    """Computes the redundancy between multiple sources and a target variable using minimum mutual information approach.

    Parameters
    ----------
    sources : numpy.ndarray
        N x M array representing the sources, where N is the number of variables and M is the number of samples.
    target : numpy.ndarray
        1 x M array representing the target variable.
    estimator : str, optional
        Estimator to calculate mutual information. Defaults to 'lin_est'.
    indvar_sources : numpy.ndarray or None, optional
        An array of indices specifying the sources to consider.
        If None, all sources are used. Default is None.
    indvar_target : numpy.ndarray or None, optional
        An array of indices specifying the target to consider.
        If None, all targets are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.

    Returns
    -------
    redundancy : float
        The computed redundancy value.

    """

    if indvar_sources is not None:
        sources = sources[indvar_sources, :]

    if indvar_target is not None:
        target = target[indvar_target, :]

    if indsample is not None:
        sources = sources[:, indsample]
        target = target[:, indsample]

    if estimator == "gcmi":

        X = copnorm(sources.copy())
        Y = copnorm(target.copy())
        estimator = "lin_est"

    else:
        X = sources.copy()
        Y = target.copy()

    n_var, n_tr = X.shape
    list_mi = []
    for i in range(n_var - 1):
        list_mi.append(
            mi(X[i], Y, estimator=estimator, biascorrect=biascorrect, base=base)
        )

    return min(list_mi)


def synergy_mmi(
    sources,
    target,
    estimator="lin_est",
    indvar_sources=None,
    indvar_target=None,
    indsample=None,
    biascorrect=True,
    base=None,
):
    """Computes the synergy between multiple sources and a target variable using minimum mutual information approach.

    Parameters
    ----------
    sources : numpy.ndarray
        N x M array representing the sources, where N is the number of variables and M is the number of samples.
    target : numpy.ndarray
        1 x M array representing the target variable.
    estimator : str, optional
        Estimator to calculate mutual information. Defaults to 'lin_est'.
    indvar_sources : numpy.ndarray or None, optional
        An array of indices specifying the sources to consider.
        If None, all sources are used. Default is None.
    indvar_target : numpy.ndarray or None, optional
        An array of indices specifying the target to consider.
        If None, all targets are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    biascorrect : bool, optional
        Whether to apply the bias correction to the entropy estimators.
        Default is True.
    base : float or None, optional
        The base of the logarithm used to compute the entropy.

    Returns
    -------
    float
        The computed synergy value.

    """

    if indvar_sources is not None:
        sources = sources[indvar_sources, :]

    if indvar_target is not None:
        target = target[indvar_target, :]

    if indsample is not None:
        sources = sources[:, indsample]
        target = target[:, indsample]

    if estimator == "gcmi":
        X = copnorm(sources.copy())
        Y = copnorm(target.copy())
        estimator = "lin_est"

    else:
        X = sources.copy()
        Y = target.copy()

    n_var, n_tr = X.shape
    list_mi = []
    for i in range(n_var - 1):
        X1 = np.delete(X, i, axis=0)
        list_mi.append(
            mi(X1, Y, estimator=estimator, biascorrect=biascorrect, base=base)
        )

    return mi(X, Y, estimator=estimator) - max(list_mi)


def o_information_lagged_boot(
    Y, X, m=1, indstart=None, chunklength=None, estimator="lin_est", indvar=None
):
    """Calculates the dynamical o_information of a dataset.

    Parameters
    ----------
    Y : numpy.ndarray
        The input data array for the target variable.
    X : numpy.ndarray
        The input data array for the driver variables.
    m : int, optional
        The time delay between the variables. Default is 1.
    indstart : numpy.ndarray or None, optional
        An array of indices specifying the starting indices for each chunk.
        If None, all indices are used. Default is None.
    chunklength : numpy.ndarray or None, optional
        An array of indices specifying the length of each chunk.
        If None, all indices are used. Default is None.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi', 'cat_ent'. Default is 'lin_est'.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.

    Returns
    -------
    o : float
        The dynamical o_information value.

    """
    if chunklength is None:
        chunklength = len(X[:, 0])

    if indstart is None:
        indstart = [0]

    if chunklength == 0:
        indsample = np.arange(len(Y))
    else:
        nchunks = int(np.floor(len(Y) / chunklength))
        indstart = indstart[0:nchunks]
        indsample = np.zeros(nchunks * chunklength)
        for istart in range(nchunks):
            indsample[(istart) * chunklength : (istart + 1) * chunklength] = np.arange(
                indstart[istart], indstart[istart] + chunklength
            )

    indsample = indsample.astype("int32")

    Y = Y[indsample]
    X = X[indsample, :]

    if indvar is not None:
        X = X[:, indvar]

    N, M = X.shape

    n = N - m
    X0 = np.zeros((n, m, M))
    Y0 = np.zeros((n, m))
    y = np.array([Y[m:]])
    for i in range(n):
        for j in range(m):
            Y0[i, j] = Y[m - j + i - 1]
            for k in range(M):
                X0[i, j, k] = X[m - j + i - 1, k]

    X0_reshaped = np.reshape(np.ravel(X0, order="F"), (n, m * M), order="F").T
    Y0 = Y0.T

    o = -(M - 1) * get_cmi(y, X0_reshaped, Y0, estimator)

    for k in range(M):
        X = np.delete(X0, k, axis=2)
        X_reshaped = np.reshape(np.ravel(X, order="F"), (n, m * (M - 1)), order="F").T
        o = o + get_cmi(y, X_reshaped, Y0, estimator)

    return o


def o_information_lagged_all(
    data,
    estimator="lin_est",
    m=1,
    indvar=None,
    indsample=None,
    indstart=None,
    chunklength=None,
):
    """Calculates the total dynamical o_information of a dataset.

    Parameters
    ----------
    data : numpy.ndarray
        The input data array of shape (M, N), where M is the number of variables
        and N is the number of samples.
    estimator : str, optional
        The estimator used to compute the entropy.
        Options: 'lin_est', 'gcmi'. Default is 'lin_est'.
    m : int, optional
        The time delay between the variables. Default is 1.
    indvar : numpy.ndarray or None, optional
        An array of indices specifying the variables to consider.
        If None, all variables are used. Default is None.
    indsample : numpy.ndarray or None, optional
        An array of indices specifying the samples to consider.
        If None, all samples are used. Default is None.
    indstart : numpy.ndarray or None, optional
        An array of indices specifying the starting indices for each chunk.
        If None, all indices are used. Default is None.
    chunklength : numpy.ndarray or None, optional
        An array of indices specifying the length of each chunk.
        If None, all indices are used. Default is None.

    Returns
    -------
    dyn_o : float
        The total dynamical o_information value.

    """

    if estimator == "gcmi":

        X = copnorm(data.copy())
        estimator = "lin_est"

    else:
        X = data.copy()

    X = X.T

    if indvar is not None:
        X = X[indvar, :]

    if indsample is not None:
        X = X[:, indsample]

    dyn_o = 0

    for i in range(len(X[0, :])):

        dyn_o += o_information_lagged_boot(
            X[:, i],
            np.delete(X, i, axis=1),
            m=m,
            indstart=indstart,
            chunklength=chunklength,
            estimator=estimator,
        )

    return dyn_o
