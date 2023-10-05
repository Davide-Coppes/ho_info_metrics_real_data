from tqdm.auto import tqdm
import numpy as np

from .utils import (
    copnorm,
    combinations_manager,
    ncr,
    bootci,
    generate_components,
    check_overlap,
)
from .metrics import (
    o_information_boot,
    s_information_boot,
    tc_boot,
    dtc_boot,
    o_information_lagged_boot,
)


def exhaustive_loop_zerolag(ts, config):
    """Function that implements points 1-3 of the pipeline described in https://arxiv.org/abs/2205.01035
    for the Oinfo, Sinfo, TC and DTC metrics.

    Parameters
    ----------
    ts : array
        Array of shape (n_variables, n_timepoints) containing the time series
    config : dict
        Dictionary containing the configuration parameters.
        The dictionary must contain the following keys:
            - estimator: string, one of 'lin_est', 'gcmi', 'cat_ent'
            - metric: string, one of 'O-info', 'S-info', 'TC', 'DTC'
            - maxsize: int, maximum size of the multiplet
            - n_best: int, number of most informative multiplets retained
            - nboot: int, number of bootstrap samples
            - alphaval: float, alpha value for the confidence interval
            - biascorrect: bool, whether to use bias correction for the metrics
            - base: int or float, base for the logarithm to compute entropies
    -------
    resDict : dict
        Dictionary containing the results of the computation.
        For every multiplet size, the dictionary contains the following keys:
            - 'sorted': array of shape (n_best,), containing the sorted values of the metric
            - 'index': array of shape (n_best,), containing the indices of the multiplets
            - 'bootsig': array of shape (n_best, 1), containing the significance of the multiplets
            - 'ci': array of shape (n_best, 2), containing the confidence intervals of the multiplets
    """
    if "biascorrect" not in config:
        biascorrect = True
    else:
        biascorrect = config["biascorrect"]
    if "base" not in config:
        base = None
    else:
        base = config["base"]
    estimator = config["estimator"]
    metric = config["metric"]
    if metric not in ["O-info", "S-info", "TC", "DTC"]:
        print("ERROR: please use one of the following metrics: O-info, S-info, TC, DTC")
        return 0
    Xfull = copnorm(ts)
    nvartot, N = Xfull.shape
    print(
        "Timeseries details - Number of variables: ",
        str(nvartot),
        ", Number of timepoints: ",
        str(N),
    )
    print("Computing " + metric + " using " + estimator + " estimator")
    X = Xfull
    maxsize = config["maxsize"]  # max number of variables in the multiplet
    n_best = config["n_best"]  # number of most informative multiplets retained
    nboot = config["nboot"]  # number of bootstrap samples
    alphaval = config["alphaval"]

    resDict = {}

    bar_length = maxsize + 1 - 3
    with tqdm(total=bar_length) as pbar:
        pbar.set_description("Outer loops")
        for isize in tqdm(range(3, maxsize + 1), disable=True):
            tot = {}
            H = combinations_manager(nvartot, isize)
            ncomb = ncr(nvartot, isize)
            if metric == "O-info":
                O_pos = np.zeros(n_best)
                O_neg = np.zeros(n_best)
                ind_pos = np.zeros(n_best)
                ind_neg = np.zeros(n_best)
            else:
                V = np.zeros(n_best)
                ind_V = np.zeros(n_best)

            for icomb in tqdm(range(ncomb), desc="Inner loop", leave=False):
                comb = H.nextchoose()
                if metric == "O-info":
                    Osize = o_information_boot(
                        X,
                        estimator=estimator,
                        indsample=range(N),
                        indvar=comb - 1,
                        biascorrect=biascorrect,
                        base=base,
                    )
                    valpos, ipos = np.min(O_pos), np.argmin(O_pos)
                    valneg, ineg = np.max(O_neg), np.argmax(O_neg)
                    if Osize > 0 and Osize > valpos:
                        O_pos[ipos] = Osize
                        ind_pos[ipos] = H.combination2number(comb)
                    if Osize < 0 and Osize < valneg:
                        O_neg[ineg] = Osize
                        ind_neg[ineg] = H.combination2number(comb)
                    Osort_pos, ind_pos_sort = (
                        np.sort(O_pos)[::-1],
                        np.argsort(O_pos)[::-1],
                    )
                    Osort_neg, ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
                else:
                    if metric == "S-info":
                        Vsize = s_information_boot(
                            X,
                            estimator=estimator,
                            indsample=range(N),
                            indvar=comb - 1,
                            biascorrect=biascorrect,
                            base=base,
                        )
                    elif metric == "TC":
                        Vsize = tc_boot(
                            X,
                            range(N),
                            comb - 1,
                            estimator,
                            biascorrect=biascorrect,
                            base=base,
                        )
                    elif metric == "DTC":
                        Vsize = dtc_boot(
                            X,
                            range(N),
                            comb - 1,
                            estimator,
                            biascorrect=biascorrect,
                            base=base,
                        )
                    iV = np.argmin(V)
                    V[iV] = Vsize
                    ind_V[iV] = H.combination2number(comb)
                    Vsort, ind_V_sort = np.sort(V)[::-1], np.argsort(V)[::-1]

            if metric == "O-info":
                if Osort_pos.size != 0:
                    n_sel = min(n_best, len(Osort_pos))
                    boot_sig = np.zeros((n_sel, 1))
                    ci_array = np.zeros((n_sel, 2))
                    for isel in range(n_sel):
                        indvar = H.number2combination(ind_pos[ind_pos_sort[isel]])

                        def f(xsamp):
                            return o_information_boot(
                                X,
                                estimator=estimator,
                                indsample=xsamp,
                                indvar=indvar - 1,
                                biascorrect=biascorrect,
                                base=base,
                            )

                        ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
                        boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                        ci_array[isel] = [ci_lower, ci_upper]
                    tot["sorted_red"] = Osort_pos[0:n_sel]
                    tot["index_red"] = ind_pos[ind_pos_sort[0:n_sel]].flatten()
                    tot["bootsig_red"] = boot_sig
                    tot["ci_red"] = ci_array

                if Osort_neg.size != 0:
                    n_sel = min(n_best, len(Osort_neg))
                    boot_sig = np.zeros((n_sel, 1))
                    ci_array = np.zeros((n_sel, 2))
                    for isel in range(n_sel):
                        indvar = H.number2combination(ind_neg[ind_neg_sort[isel]])

                        def f(xsamp):
                            return o_information_boot(
                                X,
                                estimator=estimator,
                                indsample=xsamp,
                                indvar=indvar - 1,
                                biascorrect=biascorrect,
                                base=base,
                            )

                        ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
                        boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                        ci_array[isel] = [ci_lower, ci_upper]
                    tot["sorted_syn"] = Osort_neg[0:n_sel]
                    tot["index_syn"] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                    tot["bootsig_syn"] = boot_sig
                    tot["ci_syn"] = ci_array
            else:
                if Vsort.size != 0:
                    n_sel = min(n_best, len(Vsort))
                    boot_sig = np.zeros((n_sel, 1))
                    ci_array = np.zeros((n_sel, 2))
                    for isel in range(n_sel):
                        indvar = H.number2combination(ind_V[ind_V_sort[isel]])
                        if metric == "S-info":

                            def f(xsamp):
                                return s_information_boot(
                                    X,
                                    estimator=estimator,
                                    indsample=xsamp,
                                    indvar=indvar - 1,
                                    biascorrect=biascorrect,
                                    base=base,
                                )

                        elif metric == "TC":

                            def f(xsamp):
                                return tc_boot(
                                    X,
                                    xsamp,
                                    indvar - 1,
                                    estimator,
                                    biascorrect=biascorrect,
                                    base=base,
                                )

                        elif metric == "DTC":

                            def f(xsamp):
                                return dtc_boot(
                                    X,
                                    xsamp,
                                    indvar - 1,
                                    estimator,
                                    biascorrect=biascorrect,
                                    base=base,
                                )

                        ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
                        boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                        ci_array[isel] = [ci_lower, ci_upper]
                    tot["sorted"] = Vsort[0:n_sel]
                    tot["index"] = ind_V[ind_V_sort[0:n_sel]].flatten()
                    tot["bootsig"] = boot_sig
                    tot["ci"] = ci_array
            resDict[isize] = tot
            pbar.update(1)
    return resDict


def validated_hypergraph(ts, config):
    """Function that implements points 1-5 of the pipeline described in https://arxiv.org/abs/2205.01035

    Parameters
    ----------
    ts : array
        Array of shape (n_variables, n_timepoints) containing the time series
    config : dict
        Dictionary containing the configuration parameters.
        The dictionary must contain the following keys:
            - estimator: string, one of 'lin_est', 'gcmi', 'cat_ent'
            - metric: string, one of 'O-info', 'S-info', 'TC', 'DTC'
            - maxsize: int, maximum size of the multiplet
            - n_best: int, number of most informative multiplets retained
            - nboot: int, number of bootstrap samples
            - alphaval: float, alpha value for the confidence interval
            - biascorrect: bool, whether to use bias correction for the metrics
            - base: int or float, base for the logarithm to compute entropies
    Returns
    -------
    nvartot : int
        Number of variables in the time series
    maxsize : int
        Maximum size of the multiplet
    resDict : dict
        Dictionary containing the results of the computation.
        For every multiplet size, the dictionary contains the following keys:
            - 'sorted': array of shape (n_best,), containing the sorted values of the metric
            - 'index': array of shape (n_best,), containing the indices of the multiplets
            - 'bootsig': array of shape (n_best, 1), containing the significance of the multiplets
            - 'ci': array of shape (n_best, 2), containing the confidence intervals of the multiplets
    """

    metric = config["metric"]
    maxsize = config["maxsize"]  # max number of variables in the multiplet
    Xfull = copnorm(ts)
    nvartot, N = Xfull.shape
    resDict = exhaustive_loop_zerolag(ts, config)
    for isize in range(4, maxsize + 1):
        H = combinations_manager(nvartot, isize)
        G = combinations_manager(nvartot, isize - 1)
        # redundancy
        if metric == "O-info":
            # redundancy
            if "bootsig_red" in resDict[isize]:
                significancy_array = resDict[isize]["bootsig_red"]
                index_array = resDict[isize]["index_red"]
                ci_array = resDict[isize]["ci_red"]
                ci_array_inf = resDict[isize - 1]["ci_red"]
                for multiplet in index_array:
                    input_set = H.number2combination(multiplet)
                    index_sup = int(
                        np.where(resDict[isize]["index_red"] == multiplet)[0]
                    )
                    subsets = generate_components(input_set)
                    for subset in subsets:
                        subset_number = G.combination2number(subset)
                        ci_array_inf = resDict[isize - 1]["ci_red"]
                        if (
                            len(
                                np.where(
                                    resDict[isize - 1]["index_red"] == subset_number
                                )[0]
                            )
                            == 1
                        ):
                            index_inf = int(
                                np.where(
                                    resDict[isize - 1]["index_red"] == subset_number
                                )[0]
                            )
                            interval1 = ci_array_inf[index_inf]
                            interval2 = ci_array[index_sup]
                            if check_overlap(interval1, interval2):
                                significancy_array[index_sup] = [0.0]

                resDict[isize]["bootsig_red"] = significancy_array

            # synergy
            if "bootsig_syn" in resDict[isize]:
                significancy_array = resDict[isize]["bootsig_syn"]
                index_array = resDict[isize]["index_syn"]
                ci_array = resDict[isize]["ci_syn"]
                ci_array_inf = resDict[isize - 1]["ci_syn"]
                for multiplet in index_array:
                    input_set = H.number2combination(multiplet)
                    index_sup = int(
                        np.where(resDict[isize]["index_syn"] == multiplet)[0]
                    )
                    subsets = generate_components(input_set)
                    for subset in subsets:
                        subset_number = G.combination2number(subset)
                        ci_array_inf = resDict[isize - 1]["ci_syn"]
                        if (
                            len(
                                np.where(
                                    resDict[isize - 1]["index_syn"] == subset_number
                                )[0]
                            )
                            == 1
                        ):
                            index_inf = int(
                                np.where(
                                    resDict[isize - 1]["index_syn"] == subset_number
                                )[0]
                            )
                            interval1 = ci_array_inf[index_inf]
                            interval2 = ci_array[index_sup]
                            if check_overlap(interval1, interval2):
                                significancy_array[index_sup] = [0.0]

                resDict[isize]["bootsig_syn"] = significancy_array

        else:
            if "bootsig" in resDict[isize]:
                significancy_array = resDict[isize]["bootsig"]
                index_array = resDict[isize]["index"]
                ci_array = resDict[isize]["ci"]
                ci_array_inf = resDict[isize - 1]["ci"]
                for multiplet in index_array:
                    input_set = H.number2combination(multiplet)
                    index_sup = int(np.where(resDict[isize]["index"] == multiplet)[0])
                    subsets = generate_components(input_set)
                    for subset in subsets:
                        subset_number = G.combination2number(subset)
                        ci_array_inf = resDict[isize - 1]["ci"]
                        if (
                            len(
                                np.where(resDict[isize - 1]["index"] == subset_number)[
                                    0
                                ]
                            )
                            == 1
                        ):
                            index_inf = int(
                                np.where(resDict[isize - 1]["index"] == subset_number)[
                                    0
                                ]
                            )
                            interval1 = ci_array_inf[index_inf]
                            interval2 = ci_array[index_sup]
                            if check_overlap(interval1, interval2):
                                significancy_array[index_sup] = [0.0]

                resDict[isize]["bootsig"] = significancy_array
    return nvartot, maxsize, resDict


def exhaustive_loop_lagged(ts, config):
    """Function that implements points 1-3 of the pipeline described in https://arxiv.org/abs/2205.01035
    for the dOinfo metric.

    Parameters
    ----------
    ts : array
        Array of shape (n_variables, n_timepoints) containing the time series
    config : dict
        Dictionary containing the configuration parameters.
        The dictionary must contain the following keys:
            - estimator: string, one of 'lin_est', 'gcmi', 'cat_ent'
            - modelorder: int, model order for the dOinfo computation
            - maxsize: int, maximum size of the multiplet
            - n_best: int, number of most informative multiplets retained
            - nboot: int, number of bootstrap samples
            - alphaval: float, alpha value for the confidence interval
    Returns
    -------
    Odict : dict
        Dictionary containing the results of the computation.
        For every multiplet size and for every target variable, the dictionary contains the following keys:
            - 'sorted_red': array of shape (n_best,), containing the sorted values of the metric
            - 'index_red': array of shape (n_best,), containing the indices of the multiplets
            - 'bootsig_red': array of shape (n_best, 1), containing the significance of the multiplets
            - 'sorted_syn': array of shape (n_best,), containing the sorted values of the metric
            - 'index_syn': array of shape (n_best,), containing the indices of the multiplets
            - 'bootsig_syn': array of shape (n_best, 1), containing the significance of the multiplets
    """

    estimator = config["estimator"]
    Xfull = copnorm(ts)
    nvartot, N = Xfull.shape
    print(
        "Timeseries details - Number of variables: ",
        str(nvartot),
        ", Number of timepoints: ",
        str(N),
    )
    print("Computing dOinfo using " + estimator + " estimator")
    X = Xfull.T
    modelorder = config["modelorder"]
    maxsize = config["maxsize"]
    n_best = config["n_best"]
    nboot = config["nboot"]
    chunklength = round(N / 5)
    # can play around with this
    alphaval = 0.05

    Odict = {}
    bar_length = nvartot * (maxsize + 1 - 2)
    with tqdm(total=bar_length) as pbar:
        pbar.set_description("Outer loops")
        for itarget in tqdm(range(nvartot), disable=True):
            Otarget = {}
            t = X[:, itarget]
            for isize in tqdm(range(2, maxsize + 1), disable=True):
                Otot = {}
                var_arr = np.setdiff1d(np.arange(1, nvartot + 1), itarget + 1)
                H = combinations_manager(len(var_arr), isize)
                ncomb = ncr(len(var_arr), isize)
                O_pos = np.zeros(n_best)
                O_neg = np.zeros(n_best)
                ind_pos = np.zeros(n_best)
                ind_neg = np.zeros(n_best)
                Osize = np.zeros(ncomb)
                for icomb in tqdm(
                    range(ncomb), desc="Inner loop, computing dO-info", leave=False
                ):
                    comb = H.nextchoose()
                    Osize = o_information_lagged_boot(
                        t,
                        X,
                        modelorder,
                        np.arange(N),
                        0,
                        var_arr[comb - 1] - 1,
                        estimator,
                    )
                    valpos, ipos = np.min(O_pos), np.argmin(O_pos)
                    valneg, ineg = np.max(O_neg), np.argmax(O_neg)
                    if Osize > 0 and Osize > valpos:
                        O_pos[ipos] = Osize
                        ind_pos[ipos] = H.combination2number(comb)
                    if Osize < 0 and Osize < valneg:
                        O_neg[ineg] = Osize
                        ind_neg[ineg] = H.combination2number(comb)
                Osort_pos, ind_pos_sort = np.sort(O_pos)[::-1], np.argsort(O_pos)[::-1]
                Osort_neg, ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
                if Osort_pos.size != 0:
                    n_sel = min(n_best, len(Osort_pos))
                    boot_sig = np.zeros((n_sel, 1))
                    for isel in range(n_sel):
                        if Osort_pos[isel] != 0.0:
                            indvar = H.number2combination(ind_pos[ind_pos_sort[isel]])

                            def f(xsamp):
                                return o_information_lagged_boot(
                                    t,
                                    X,
                                    modelorder,
                                    xsamp,
                                    chunklength,
                                    var_arr[indvar - 1] - 1,
                                    estimator,
                                )

                            ci_lower, ci_upper = bootci(
                                nboot, f, np.arange(N - chunklength + 1), alphaval
                            )
                            boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                    Otot["sorted_red"] = Osort_pos[0:n_sel]
                    Otot["index_red"] = ind_pos[ind_pos_sort[0:n_sel]].flatten()
                    Otot["bootsig_red"] = boot_sig
                if Osort_neg.size != 0:
                    n_sel = min(n_best, len(Osort_neg))
                    boot_sig = np.zeros((n_sel, 1))
                    for isel in range(n_sel):
                        if Osort_neg[isel] != 0.0:
                            indvar = H.number2combination(ind_neg[ind_neg_sort[isel]])

                            def f(xsamp):
                                return o_information_lagged_boot(
                                    t,
                                    X,
                                    modelorder,
                                    xsamp,
                                    chunklength,
                                    var_arr[indvar - 1] - 1,
                                    estimator,
                                )

                            ci_lower, ci_upper = bootci(
                                nboot, f, np.arange(N - chunklength + 1), alphaval
                            )
                            boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                    Otot["sorted_syn"] = Osort_neg[0:n_sel]
                    Otot["index_syn"] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                    Otot["bootsig_syn"] = boot_sig
                Otot["var_arr"] = var_arr
                Otarget[isize] = Otot
                pbar.update(1)
            Odict[itarget + 1] = Otarget
    return Odict
