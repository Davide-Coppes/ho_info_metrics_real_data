import numpy as np
import xgi
from multiprocessing import Pool
import random
from scipy.spatial.distance import jensenshannon
import itertools

from ho_info_metrics.metrics import o_information_lagged_all


def get_edges_neighbors(edges_list, i):
    """Return the list of nodes that are connected to the node i with an edge

    Parameters
    ----------
    edges_list : list
        List of edges
    i : int
        Node index

    Returns
    -------
    neighbors : list
        List of nodes connected to the node i
    """
    neighbors = []
    for edge in edges_list:
        if i in edge:
            neighbors.append(edge[0] if edge[1] == i else edge[1])
    return neighbors


def get_triangles_neighbors(triangles_list, i):
    """Return the list of triangles that are connected to the node i

    Parameters
    ----------
    triangles_list : list
        List of triangles
    i : int
        Node index

    Returns
    -------
    neighbors : list
        List of triangles connected to the node i
    """

    neighbors = []
    for triangle in triangles_list:
        if i in triangle:
            neighbors.append(triangle)
    return neighbors


def ising_dynamic(N, edges_list, triangles_list, J1, J2, n_steps, T=None):
    """Simulate the Ising model on a simplicial complex.

    Parameters
    ----------
    N : int
        The number of spins.
    edges_list : list
        The list of edges.
    triangles_list : list
        The list of triangles.
    J1 : float
        The coupling constant of the edges.
    J2 : float
        The coupling constant of the triangles.
    T : float or None
        The temperature. If None, T = 1/J1, set the system at the critical temperature.
    n_steps : int
        The number of steps.

    Returns
    -------
    ts : array
        The time series of the spins.
    """
    if T is None:
        T = 1 / J1

    energies = []
    magnetizations = []

    # initialize the spins
    spins = np.random.choice([-1, 1], size=N)
    ts = np.zeros((N, n_steps))
    energy = J1 * np.sum([spins[i] * spins[j] for i, j in edges_list]) + J2 * np.sum(
        [spins[i] * spins[j] * spins[k] for i, j, k in triangles_list]
    )
    for step in range(n_steps):
        i = np.random.randint(N)
        edges_neighbors = get_edges_neighbors(edges_list, i)
        triangles_neighbors = get_triangles_neighbors(triangles_list, i)
        # compute the energy change
        delta_E = 0
        for j in edges_neighbors:
            delta_E += J1 * spins[i] * spins[j]
        for triangle in triangles_neighbors:
            delta_E += J2 * spins[triangle[0]] * spins[triangle[1]] * spins[triangle[2]]
            # flip the spin with probability
        if np.random.rand() < np.exp(delta_E / T) / (
            1 + np.exp(delta_E / T)
        ):  # modified, Glauber dynamics
            energy -= delta_E
            spins[i] *= -1
        ts[:, step] = spins
        energies.append(energy)
        magnetizations.append(np.sum(spins) / N)
    return ts, energies, magnetizations


def process_combination(comb):
    J1, J2 = comb

    # Simplicial Complex parameters

    N = 200
    ps = [0.01, 0.0005]

    S = xgi.random_simplicial_complex(N, ps, seed=1)

    # 2-simplices
    two_simplices = S.edges.filterby("size", 3).members()
    two_simplices = [sorted(list(i)) for i in two_simplices]

    # random triplets
    skeleton = xgi.convert_to_graph(S)
    all_triplets = [
        sorted(list(i))
        for i in itertools.combinations(skeleton.nodes, 3)
        if i not in two_simplices
    ]
    num_to_select = len(two_simplices)  # set the number to select here.
    random_triplets = random.sample(all_triplets, num_to_select)

    edges = [list(i) for i in S.edges.filterby("order", 1).members()]
    triangles = [list(i) for i in S.edges.filterby("order", 2).members()]

    ts, _, _ = ising_dynamic(N, edges, triangles, J1, J2, 100000, T=6)
    ts = ts[:, 60000:]

    simplices = []
    for triplet in two_simplices:
        simplices.append(o_information_lagged_all(ts[triplet, :], estimator="cat_ent"))

    other_triplets = []
    for tri in random_triplets:
        other_triplets.append(o_information_lagged_all(ts[tri, :], estimator="cat_ent"))

    bins = np.linspace(
        min(simplices + other_triplets), max(simplices + other_triplets), 100
    )
    p_simplices, _ = np.histogram(simplices, bins=bins, density=True)
    p_random, _ = np.histogram(other_triplets, bins=bins, density=True)

    print(
        f">>> Processing J1={J1}, J2={J2} \t js={jensenshannon(p_simplices, p_random)} \t T={6}"
    )

    return jensenshannon(p_simplices, p_random)


if __name__ == "__main__":
    shape = 40
    J1s = np.around(np.linspace(0, 8, shape), 2)
    J2s = np.around(np.linspace(0, 8, shape), 2)

    res = np.zeros((shape, shape))

    combinations = list(itertools.product(J1s, J2s))

    with Pool() as pool:
        results = list(pool.imap(process_combination, combinations))

    for i, js in enumerate(results):
        J1, J2 = combinations[i]
        res[np.argwhere(J1s == J1), np.argwhere(J2s == J2)] = js

    np.save("do_info_js_J1_J2_ising_big.npy", res)
