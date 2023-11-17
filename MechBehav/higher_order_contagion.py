import networkx as nx
from itertools import combinations
import string
import numpy as np
import random
import json
import pickle
import copy


def generate_my_simplicial_complex_d2(N, p1, p2):

    """Our model"""

    # I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, p1, seed=0)

    if not nx.is_connected(G):
        giant = list(nx.connected_components(G))[0]
        G = nx.subgraph(G, giant)
        print("not connected, but GC has order %i and size %i" % (len(giant), G.size()))

    triangles_list = []
    G_copy = G.copy()

    # Now I run over all the possible combinations of three elements:
    for tri in combinations(list(G.nodes()), 3):
        # And I create the triangle with probability p2
        if random.random() <= p2:
            # I close the triangle.
            triangles_list.append(tri)

            # Now I also need to add the new links to the graph created by the triangle
            G_copy.add_edge(tri[0], tri[1])
            G_copy.add_edge(tri[1], tri[2])
            G_copy.add_edge(tri[0], tri[2])

    G = G_copy

    # Creating a dictionary of neighbors
    node_neighbors_dict = {}
    for n in list(G.nodes()):
        node_neighbors_dict[n] = G[n].keys()

    # print len(triangles_list), 'triangles created. Size now is', G.size()

    # avg_n_triangles = 3.*len(triangles_list)/G.order()

    # return node_neighbors_dict, node_triangles_dict, avg_n_triangles
    # return node_neighbors_dict, triangles_list, avg_n_triangles
    return G, node_neighbors_dict, triangles_list


def get_p1_and_p2_correction(k1, k2, N):
    p2 = (2.0 * k2) / ((N - 1.0) * (N - 2.0))
    p1 = (k1 - 2.0 * k2) / ((N - 1.0) - 2.0 * k2)
    if (p1 >= 0) and (p2 >= 0):
        return p1, p2
    else:
        raise ValueError("Negative probability!")


# model constructor
class SimplagionModel:
    def __init__(self, node_neighbors_dict, triangles_list, I_percentage):

        # parameters
        self.neighbors_dict = node_neighbors_dict
        self.triangles_list = triangles_list
        self.nodes = list(node_neighbors_dict.keys())
        self.N = len(node_neighbors_dict.keys())
        self.I = int(I_percentage * self.N / 100)

        # Initial setup
        # I save the infected nodes of the first initialisation in case I want to repeat several runs with
        # the same configuration
        # self.initial_infected_nodes = self.initial_setup()   # MY CODE, my choice

    def initial_setup(self, fixed_nodes_to_infect=None, print_status=False):
        # going to use this to store the agents in each state
        self.sAgentSet = set()
        self.iAgentSet = set()

        # and here we're going to store the counts of how many agents are in each
        # state @ each time step
        self.iList = []

        # start with everyone susceptible
        for n in self.nodes:
            self.sAgentSet.add(n)

        # infect nodes
        if (
            fixed_nodes_to_infect == None
        ):  # the first time I create the model (the instance __init__)
            infected_this_setup = []
            for ite in range(self.I):  # we will infect I agents
                # select one to infect among the supsceptibles
                to_infect = random.choice(list(self.sAgentSet))
                self.infectAgent(to_infect)
                infected_this_setup.append(to_infect)
        else:  # I already have run the model and this is not the first run, I want to infect the same nodes
            infected_this_setup = []
            for to_infect in fixed_nodes_to_infect:
                self.infectAgent(to_infect)
                infected_this_setup.append(to_infect)
        if print_status:
            print(
                "Setup:",
                self.N,
                "nodes",
                self.I,
                "infected. Initial infected list:",
                infected_this_setup,
            )
        return infected_this_setup

    def infectAgent(self, agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        return 1

    def recoverAgent(self, agent):
        self.sAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        return -1

    def run(self, t_max, beta1, beta2, mu, print_status):
        self.t = 1
        self.t_max = t_max
        self.nodes_status = []  # MY CODE
        self.simplagion_ts_matrix = []  # MY CODE

        # MY CODE: nodes_status is a tuple containing the status of each node (for t=0 in this case)
        for node in self.nodes:
            if node in self.iAgentSet:
                self.nodes_status.append(1)
            else:
                self.nodes_status.append(0)
        self.simplagion_ts_matrix.append(
            self.nodes_status
        )  # save the initial state (t=0)

        while self.t <= self.t_max:
            newIlist = set()
            self.nodes_status = []  # MY CODE

            # MY CODE: if every node is S, then restart the dynamical process, changing the seed
            if len(self.iAgentSet) == 0:
                self.initial_setup()
                print(
                    'every node is "S", restarting the process with a new random seed.'
                )

            # STANDARD CONTAGION
            # we only need to loop over the agents who are currently infectious
            for iAgent in self.iAgentSet:
                # expose their network neighbors
                for agent in self.neighbors_dict[iAgent]:
                    # given that the neighbor is susceptible
                    if agent in self.sAgentSet:
                        # infect it with probability beta1
                        if random.random() <= beta1:
                            newIlist.add(agent)

            # TRIANGLE CONTAGION
            for triangle in self.triangles_list:
                n1, n2, n3 = triangle
                if n1 in self.iAgentSet:
                    if n2 in self.iAgentSet:
                        if n3 in self.sAgentSet:
                            # infect n3 with probability beta2
                            if random.random() <= beta2:
                                newIlist.add(n3)
                    else:
                        if n3 in self.iAgentSet:
                            # infect n2 with probability beta2
                            if random.random() <= beta2:
                                newIlist.add(n2)
                else:
                    if (n2 in self.iAgentSet) and (n3 in self.iAgentSet):
                        # infect n1 with probability beta2
                        if random.random() <= beta2:
                            newIlist.add(n1)

            # Update only now the nodes that have been infected
            for n_to_infect in newIlist:
                self.infectAgent(n_to_infect)

            # for recoveries
            newRlist = set()

            # MY CODE: don't stop recovery even if everyone is infected!

            for recoverAgent in self.iAgentSet:
                # if the agent has just been infected it will not recover this time
                if recoverAgent in newIlist:
                    continue
                else:
                    if random.random() <= mu:
                        newRlist.add(recoverAgent)

            # Update only now the nodes that have been infected
            for n_to_recover in newRlist:
                self.recoverAgent(n_to_recover)

            # increment the time
            self.t += 1

            # then track the number of individuals in each state
            self.iList.append(len(self.iAgentSet))

            # MY CODE: nodes_status is a tuple containing the status of each node for each timestep
            for node in self.nodes:
                if node in self.iAgentSet:
                    self.nodes_status.append(1)
                else:
                    self.nodes_status.append(0)

            # MY CODE: here we create the time-series matrix, in wich every row is the network status for each timestep
            self.simplagion_ts_matrix.append(self.nodes_status)

        # and when we're done, return all of the relevant information
        if print_status:
            print(
                "beta1 = ",
                beta1,
                " beta2 = ",
                beta2,
                " Done!",
                len(self.iAgentSet),
                "infected agents left",
            )

        # MY CODE: we return the transpose of the matrix, in wich each row is the hystory of a node.
        return np.array(self.simplagion_ts_matrix).T
