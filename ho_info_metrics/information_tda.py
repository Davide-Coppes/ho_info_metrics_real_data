import numpy as np
import scipy
from scipy.stats import f
from itertools import combinations
from frites import core
import gudhi as gd


def discr_local_entropy(pk, base):
    """
    To calculate local entropy in case of a 1-d discretized input pk.
    """
    pk = np.asarray(pk)
    pk = 1.0 * pk / np.sum(pk, keepdims=True)
    if base is None:
        S = scipy.special.entr(pk)
    else:
        S = (scipy.special.entr(pk)) / np.log(base)
    return S / pk


def discr_local_entropy_nd(X):
    """
    To calculate local entropy in case of a n-d discretized input X.
    """
    X_unq, X_counts = np.unique(X.T, axis=0, return_counts=True)
    X_probs = {
        tuple(x for x in X_unq[i, :]): discr_local_entropy(X_counts, base=2)[i]
        for i in range(X_unq.shape[0])
    }
    local_time_series = (1 - X.shape[0]) * np.array(
        [X_probs[tuple(x for x in X[:, t])] for t in range(X.shape[1])]
    )
    return local_time_series


def discr_loc_tot_corr(X):
    """
    To calculate local total correlation in case of a discretized input X.
    """
    # Joint entropy
    X_unq, X_counts = np.unique(X.T, axis=0, return_counts=True)
    X_probs = {
        tuple(x for x in X_unq[i, :]): discr_local_entropy(X_counts, base=2)[i]
        for i in range(X_unq.shape[0])
    }
    local_time_series = -np.array(
        [X_probs[tuple(x for x in X[:, t])] for t in range(X.shape[1])]
    )

    # Sum of single entropies
    for i in range(X.shape[0]):
        Xi_unq, Xi_counts = np.unique(X[i], axis=0, return_counts=True)
        Xi_probs = {
            Xi_unq[x]: discr_local_entropy(Xi_counts, base=2)[x]
            for x in range(Xi_unq.shape[0])
        }
        local_time_series_i = np.array([Xi_probs[X[i][t]] for t in range(X.shape[1])])
        local_time_series += local_time_series_i

    return local_time_series


def discr_loc_dual_tot_corr(X):
    """
    To calculate local dual total correlation in case of a discretized input X.
    """
    # Joint entropy
    X_unq, X_counts = np.unique(X.T, axis=0, return_counts=True)
    X_probs = {
        tuple(x for x in X_unq[i, :]): discr_local_entropy(X_counts, base=2)[i]
        for i in range(X_unq.shape[0])
    }
    local_time_series = (1 - X.shape[0]) * np.array(
        [X_probs[tuple(x for x in X[:, t])] for t in range(X.shape[1])]
    )

    # Residual entropies
    for i in range(X.shape[0]):
        X_inv = X[[x for x in range(X.shape[0]) if x != i]]
        Xinv_unq, Xinv_counts = np.unique(X_inv.T, axis=0, return_counts=True)
        Xinv_probs = {
            tuple(x for x in Xinv_unq[i, :]): discr_local_entropy(Xinv_counts, base=2)[
                i
            ]
            for i in range(Xinv_unq.shape[0])
        }
        local_time_series_i = np.array(
            [Xinv_probs[tuple(x for x in X_inv[:, t])] for t in range(X.shape[1])]
        )
        local_time_series += local_time_series_i

    return local_time_series


# The following 8 functions are mostly contribution of Thomas F. Varley


def mean(X):
    """
    Utility function to calculate the mean value of 1-d input array X.
    """
    total = 0.0
    N = X.shape[0]

    for i in range(X.shape[0]):
        total += X[i]

    return total / N


def std(X):
    """
    Utility function to calculate the standard deviation of 1-d input array X.
    """
    avg = mean(X)
    total = 0.0
    N = X.shape[0]

    for i in range(X.shape[0]):
        total += (X[i] - avg) ** 2

    total /= N

    return np.sqrt(total)


def gauss_local_entropy_1d(x, mu, sigma):
    """
    To calculate the local entropy of 1-d input array x.
    """
    return -1.0 * np.log(
        1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2.0)
    )


def gaussian_nd(x, mu, cov, inv, det):
    """
    To calculate the probability denisty of pulled from an n-dimensional Gaussian.
    """
    N = x.shape[0]
    Nf = x.shape[0]
    norm = 1.0 / np.sqrt(((2.0 * np.pi) ** (Nf)) * det)
    err = np.zeros(N)

    for i in range(N):
        err[i] = x[i] - mu[i]

    mul = -0.5 * np.matmul(np.matmul(err.T, inv), err)

    return norm * np.exp(mul)


def gauss_local_entropy_nd(X, mu, cov, inv, det):
    """
    To calculate the local entropy of n-d input array x.
    """
    return -1.0 * np.log(gaussian_nd(X, mu, cov, inv, det))


def gauss_local_entropy_series_nd(X):
    """
    To create the n-d series of gauss_local_entropy_nd values.
    """
    N0 = X.shape[0]
    N1 = X.shape[1]

    ents = np.zeros(N1)

    np.zeros(N0)
    mu = np.zeros(N0)
    sigma = np.zeros(N0)

    for i in range(N0):
        mu[i] = mean(X[i])

    if N0 == 1:
        sigma[i] = std(X[0])
        for i in range(N1):
            ents[i] = gauss_local_entropy_1d(X[0][i], mu[0], sigma[0])
    else:
        cov = np.cov(X, ddof=0.0).reshape((X.shape[0], X.shape[0]))
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        for i in range(N1):
            ents[i] = gauss_local_entropy_nd(X[:, i], mu, cov, inv, det)

    return np.array(ents)


def gauss_loc_tot_corr(X):
    """
    To calculate local total correlation (tc) in case of gaussian-assumption or
    gaussian copula pre-processed input.
    """
    N0 = X.shape[0]
    N1 = X.shape[1]

    mu = np.mean(X, axis=1)
    sigma = np.var(X, axis=1)

    joint_ents = gauss_local_entropy_series_nd(X)
    sum_marg_ents = np.zeros(N1)

    cov = np.cov(X, ddof=0.0).reshape((X.shape[0], X.shape[0]))
    np.linalg.inv(cov)
    np.linalg.det(cov)

    for i in range(N1):
        for j in range(N0):
            sum_marg_ents[i] += gauss_local_entropy_1d(X[j, i], mu[j], sigma[j])

    return np.subtract(sum_marg_ents, joint_ents)


def gauss_loc_dual_tot_corr(X):
    """
    To calculate local dual total correlation (dtc) in case of gaussian-assumption or
    gaussian copula pre-processed input.
    """
    N0 = X.shape[0]
    N1 = X.shape[1]

    joint_ents = gauss_local_entropy_series_nd(X)
    sum_resid_ents = np.zeros(N1)

    for i in range(N0):
        X_resid = X[[x for x in range(N0) if x != i]]
        joint_ents_resid = gauss_local_entropy_series_nd(X_resid)

        for j in range(N1):
            sum_resid_ents[j] += joint_ents[j] - joint_ents_resid[j]

    local_dtc = np.zeros(N1)
    for i in range(N1):
        local_dtc[i] += joint_ents[i] - sum_resid_ents[i]

    return np.array(local_dtc)


def compute_discr_dtc_bounded(X):
    """
    To calculate dtc normalized to entropy values in case of discretized input.
    """
    dtc = discr_loc_dual_tot_corr(X)
    h = discr_local_entropy_nd(X)

    return dtc / h


def compute_gauss_dtc_bounded(X):
    """
    To calculate dtc normalized to entropy values in case of gaussian-assumption
    pre-processed input.
    """
    dtc = gauss_loc_dual_tot_corr(X)
    h = gauss_local_entropy_series_nd(X)

    return dtc / h


def compute_cop_dtc_bounded(X):
    """
    To calculate dtc normalized to entropy values in case
    of gaussian copula pre-processed input.
    """
    dtc = gauss_loc_dual_tot_corr(core.copnorm_nd(X))
    h = gauss_local_entropy_series_nd(core.copnorm_nd(X))

    return dtc / h


def hyper_coherence(dataset2, dataset3, X2, X3):
    """
    To calculate hyper coherence values over time.
    Input: dataset2 = list arranging time series in tuples containing
           couples in the form [([],[]),([],[]),...]
           dataset3 = list arranging time series in tuples containing
           triples in the form [([],[],[]),([],[],[]),...]

           X2 = list of dtc values at each time step for all pairs
           X3 = list of dtc values at each time step for all triangles
    """
    # Selecting violating triangles
    violations_t = []
    for p in range(X3.shape[0]):
        triples = list(combinations(dataset3[p], 2))
        edges_indexes = [dataset2.index(triples[edge]) for edge in range(len(triples))]
        [
            violations_t.append([p, t])
            for t in range(X3.shape[1])
            if (
                X3[p][t] > X2[edges_indexes[0]][t]
                or X3[p][t] > X2[edges_indexes[1]][t]
                or X3[p][t] > X2[edges_indexes[2]][t]
            )
        ]

    # Arranging violations into a dictionary: {triangles: list of violating time points}
    v = np.array(violations_t)
    unq, index = np.unique(np.array(v)[:, 0], return_counts=False, return_index=True)
    violations_dict = {
        unq[u]: v[:, 1][index[u] : index[u + 1]].tolist() for u in range(len(unq) - 1)
    }

    # Extracting fraction of violating triangles over time
    sum_triangles = 0
    hyp_coherence_list = []

    for t in range(X3.shape[1]):
        for p in range(len(violations_dict)):
            if t in violations_dict[p]:
                sum_triangles += 1
        hyp_coherence = sum_triangles / len(dataset3)
        sum_triangles = 0
        hyp_coherence_list.append(hyp_coherence)

    return hyp_coherence_list, violations_dict


def ICC(matrix, alpha=0.05, r0=0):
    """
    To calculate Intraclass correlation coefficient (ICC).
    Translated into python by Andrea Santoro from
    the matlab code of Arash Salarian, 2008.
    """
    M = np.array(matrix)
    n, k = np.shape(M)
    SStotal = np.var(M.flatten(), ddof=1) * (n * k - 1)
    MSR = np.var(np.mean(M, 1), ddof=1) * k
    MSW = np.sum(np.var(M, 1, ddof=1)) / n
    MSC = np.var(np.mean(M, 0), ddof=1) * n
    (SStotal - MSR * (n - 1) - MSC * (k - 1)) / ((n - 1) * (k - 1))

    r = (MSR - MSW) / (MSR + (k - 1) * MSW)

    F = (MSR / MSW) * (1 - r0) / (1 + (k - 1) * r0)
    df1 = n - 1
    df2 = n * (k - 1)

    p = 1 - f.cdf(F, df1, df2)

    FL = (MSR / MSW) * (f.isf(1 - alpha / 2, n * (k - 1), n - 1))
    FU = (MSR / MSW) / (f.isf(1 - alpha / 2, n - 1, n * (k - 1)))

    LB = (FL - 1) / (FL + (k - 1))
    UB = (FU - 1) / (FU + (k - 1))

    return (r, LB, UB, F, df1, df2, p)


def hyper_complexity(n_nodes, X2, X3, violations_dict):
    """
    To calculate hyper complexity values over time.
    Input: n_nodes = number of nodes
           X2 = list of dtc values at each time step for all pairs
           X3 = list of dtc values at each time step for all triangles
           violations_dict = dictionary containing violating time points per triangle
    """
    nodes = list(range(n_nodes))
    edges = list(combinations(nodes, 2))
    triangles = list(combinations(nodes, 3))

    st_list = []  # simplex tree list (over time)

    for t in range(X2.shape[1]):
        st = gd.SimplexTree()
        for edge in range(len(edges)):
            st.insert(list(edges[edge]), filtration=X2[edge][t])
        for p in range(len(violations_dict)):
            if t not in violations_dict[p]:
                st.insert(list(triangles[p]), filtration=X3[p][t])

        st.make_filtration_non_decreasing()
        st_list.append(st)

    wass_distance = []
    for t in range(X2.shape[1]):
        st_list[t].persistence()
        I0 = st_list[t].persistence_intervals_in_dimension(1)
        pi = 0.0
        for ind in range(I0.shape[0]):
            if I0[ind][1] < float("inf"):
                pi += I0[ind][1] - I0[ind][0]
        wass_distance.append(pi)

    return wass_distance
