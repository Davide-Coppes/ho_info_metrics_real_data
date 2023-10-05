import xgi
from .utils import combinations_manager


def draw_hypergraph(resDict, nvar, maxsize, ax=None):
    """Draws the hypergraph of the given result dictionary.
    For DTC, TC and S-info.

    Parameters
    ----------
    resDict : dict
        Dictionary containing the results of the computation.
        See processing.py for more details.
    nvar : int
        Number of variables.
    maxsize : int
        Maximum size of the hyperedges.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the hypergraph. If not provided, a new figure is created.

    Returns
    -------
    H : xgi.Hypergraph
        The hypergraph of the given result dictionary.
    """
    H = xgi.Hypergraph()
    hyperedges = []
    for isize in range(3, maxsize + 1):
        G = combinations_manager(nvar, isize)
        if "sorted" in resDict[isize]:
            for pos in range(len(resDict[isize]["sorted"])):
                if resDict[isize]["bootsig"][pos] != 0:
                    multiplet = resDict[isize]["index"][pos]
                    hyperedges.append(G.number2combination(multiplet))
    H.add_edges_from(hyperedges)

    xgi.draw(H, ax=ax)
    return H


def draw_red_hypergraph(resDict, nvar, maxsize, ax=None):
    """Draws the redundancy hypergraph of the given result dictionary for the O-info.

    Parameters
    ----------
    resDict : dict
        Dictionary containing the results of the computation.
        See processing.py for more details.
    nvar : int
        Number of variables.
    maxsize : int
        Maximum size of the hyperedges.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the hypergraph. If not provided, a new figure is created.

    Returns
    -------
    H : xgi.Hypergraph
        The redundancy hypergraph of the given result dictionary.
    """

    H = xgi.Hypergraph()
    hyperedges = []
    for isize in range(3, maxsize + 1):
        G = combinations_manager(nvar, isize)
        if "sorted_red" in resDict[isize]:
            for pos in range(len(resDict[isize]["sorted_red"])):
                if resDict[isize]["bootsig_red"][pos] != 0:
                    multiplet = resDict[isize]["index_red"][pos]
                    hyperedges.append(G.number2combination(multiplet))
    H.add_edges_from(hyperedges)

    xgi.draw(H, ax=ax)
    return H


def draw_syn_hypergraph(resDict, nvar, maxsize, ax=None):
    """Draws the synergy hypergraph of the given result dictionary for the O-info.

    Parameters
    ----------
    resDict : dict
        Dictionary containing the results of the computation.
        See processing.py for more details.
    nvar : int
        Number of variables.
    maxsize : int
        Maximum size of the hyperedges.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the hypergraph. If not provided, a new figure is created.

    Returns
    -------
    H : xgi.Hypergraph
        The synergy hypergraph of the given result dictionary.
    """

    H = xgi.Hypergraph()
    hyperedges = []
    for isize in range(3, maxsize + 1):
        G = combinations_manager(nvar, isize)
        if "sorted_syn" in resDict[isize]:
            for pos in range(len(resDict[isize]["sorted_syn"])):
                if resDict[isize]["bootsig_syn"][pos] != 0:
                    multiplet = resDict[isize]["index_syn"][pos]
                    hyperedges.append(G.number2combination(multiplet))
    H.add_edges_from(hyperedges)

    xgi.draw(H, ax=ax)
    return H
