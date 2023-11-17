import numpy as np
from itertools import combinations
import random
import itertools
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool

from ho_info_metrics.metrics import o_information_lagged_all

from higher_order_contagion import *


def process_combination(comb):
    lambda1, lambda2 = comb

    # Simplicial Complex parameters
    N = 300  # number of nodes
    k1_init = 20  # average degree from wich we construct the network
    k2_init = 7  # average hyper-degree (mean number of triangles per node)
    p1, p2 = get_p1_and_p2_correction(k1_init, k2_init, N)

    G, node_neighbors_dict, triangles_list = generate_my_simplicial_complex_d2(
        N, p1, p2
    )

    # Real average degree and hyper-degree of the simplicial complex
    k1 = (
        1.0
        * sum([len(v) for v in node_neighbors_dict.values()])
        / len(node_neighbors_dict)
    )
    k2 = 3.0 * len(triangles_list) / len(node_neighbors_dict)

    print(k1, k2)

    I0_percentage = 30.0  # percentage of initial infected nodes

    mySimplagionModel = SimplagionModel(
        node_neighbors_dict, triangles_list, I0_percentage
    )

    initial_infected = mySimplagionModel.initial_setup(print_status=False)
    # create and save the initial infected nodes list

    t_max = 10000  # number of time steps
    mu = 0.8  # recovery rate

    beta1 = lambda1 * mu / k1  # simple-infection rate
    beta2 = lambda2 * mu / k2  # simplicial-infection rate
    mySimplagionModel.initial_setup(fixed_nodes_to_infect=initial_infected)
    one_result = mySimplagionModel.run(t_max, beta1, beta2, mu, print_status=False)

    two_simplices = []
    for triplet in triangles_list:
        two_simplices.append(
            o_information_lagged_all(one_result[triplet, :], estimator="cat_ent")
        )

    other_triplets = []
    random_triplets = list(combinations(G.nodes(), 3))
    random.shuffle(random_triplets)
    counter = 1
    for tri in random_triplets:
        if counter > len(triangles_list):
            break
        if tri in triangles_list:
            continue
        else:
            other_triplets.append(
                o_information_lagged_all(one_result[tri, :], estimator="cat_ent")
            )
            counter += 1

    bins = np.linspace(
        min(two_simplices + other_triplets), max(two_simplices + other_triplets), 100
    )
    p_simplices, _ = np.histogram(two_simplices, bins=bins, density=True)
    p_random, _ = np.histogram(other_triplets, bins=bins, density=True)

    js = jensenshannon(p_simplices, p_random)

    print(f">>> Processing lambda1={lambda1}, lambda2={lambda2} \t js={js}")

    return js


if __name__ == "__main__":
    shape = 40

    lambda1s = np.around(np.linspace(0.1, 10, shape), 2)
    lambda2s = np.around(np.linspace(0.1, 10, shape), 2)

    res = np.zeros((shape, shape))

    combinations = list(itertools.product(lambda1s, lambda2s))

    with Pool() as pool:
        results = list(pool.imap(process_combination, combinations))

    for i, js in enumerate(results):
        lambda1, lambda2 = combinations[i]
        res[np.argwhere(lambda1s == lambda1), np.argwhere(lambda2s == lambda2)] = js

    np.save("do_info_simplagion_js_l1_l2_" + str(shape) + ".npy", res)
