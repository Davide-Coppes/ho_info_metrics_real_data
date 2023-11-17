from collections import defaultdict
import random
import numpy as np
import xgi

def threshold(node, status, edge, threshold=1):
    """Threshold contagion process.

    Contagion may spread if greater than a specified fraction
    of hyperedge neighbors are infected.

    Parameters
    ----------
    node : hashable
        node ID
    status : dict
        keys are node IDs and values are their statuses.
    edge : iterable of hashables
        nodes in the hyperedge
    threshold : float, default: 0.5
        the critical fraction of hyperedge neighbors above
        which contagion spreads.

    Returns
    -------
    int
        0 if no transmission can occur, 1 if it can.
    """
    neighbors = set(edge).difference({node})
    try:
        c = sum([status[i] == "I" for i in neighbors]) / len(neighbors)
    except:
        c = 0

    if c < threshold:
        return 0
    elif c >= threshold:
        return 1

def discrete_SIS_with_states(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    recovery_weight=None,
    transmission_weight=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    dt=1.0,
    return_event_data=False,
    seed=None,
    **args
):
    if seed is not None:
        random.seed(seed)

    members = H.edges.members(dtype=dict)
    memberships = H.nodes.memberships()

    if rho is not None and initial_infecteds is not None:
        raise ValueError("cannot define both initial_infecteds and rho")

    if return_event_data:
        events = list()

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.num_nodes * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)

    if transmission_weight is not None:
        def edgeweight(item):
            return item[transmission_weight]
    else:
        def edgeweight(item):
            return 1

    if recovery_weight is not None:
        def nodeweight(u):
            return H.nodes[u][recovery_weight]
    else:
        def nodeweight(u):
            return 1

    status = defaultdict(lambda: "S")
    for node in initial_infecteds:
        status[node] = "I"

        if return_event_data:
            events.append(
                {
                    "time": tmin,
                    "source": None,
                    "target": node,
                    "old_state": "S",
                    "new_state": "I",
                }
            )

    if return_event_data:
        for node in set(H.nodes).difference(initial_infecteds):
            events.append(
                {
                    "time": tmin,
                    "source": None,
                    "target": node,
                    "old_state": "I",
                    "new_state": "S",
                }
            )

    I = [len(initial_infecteds)]
    S = [H.num_nodes - I[0]]
    times = [tmin]
    t = tmin
    new_status = status

    state_matrix = []  # List to track the state of each node at each time step

    # Append initial state and time
    node_states = [1 if new_status[node] == "I" else 0 for node in H.nodes]
    state_matrix.append(node_states)

    while t <= tmax and I[-1] != 0:
        S.append(S[-1])
        I.append(I[-1])

        node_states = []  # List to track the state of each node at the current time step

        for node in H.nodes:
            if status[node] == "I":
                if random.random() <= gamma * dt * nodeweight(node):
                    new_status[node] = "S"
                    S[-1] += 1
                    I[-1] += -1

                    if return_event_data:
                        events.append(
                            {
                                "time": t,
                                "source": None,
                                "target": node,
                                "old_state": "I",
                                "new_state": "S",
                            }
                        )
                else:
                    new_status[node] = "I"
            else:
                infected = False
                for edge_id in memberships[node]:
                    edge = members[edge_id]
                    if tau[len(edge)] > 0:
                        if random.random() <= tau[len(edge)] * transmission_function(
                            node, status, edge, **args
                        ) * dt * edgeweight(edge_id):
                            new_status[node] = "I"
                            S[-1] += -1
                            I[-1] += 1

                            if return_event_data:
                                events.append(
                                    {
                                        "time": t,
                                        "source": edge_id,
                                        "target": node,
                                        "old_state": "S",
                                        "new_state": "I",
                                    }
                                )
                            infected = True
                            break
                if not infected:
                    new_status[node] == "S"
            node_states.append(1 if new_status[node] == "I" else 0)  # Store 0 or 1 based on state

        status = new_status.copy()
        t += dt
        times.append(t)
        state_matrix.append(node_states)  # Store the state of all nodes at the current time step

    if return_event_data:
        return events
    else:
        return np.array(times), np.array(S), np.array(I), np.array(state_matrix)
    
def dirac_delta(xs):
    if xs[0] == xs[1]:
        return 1
    else:
        return 0

def get_pi(i, H, state):
    neighbors = H.nodes.neighbors(i)
    pi = 1/len(neighbors)*sum([2*dirac_delta([state[i], state[j]])-1 for j in neighbors])
    return pi

def get_ksi(i, H, state):
    neighbors = H.nodes.neighbors(i)
    xi = 1/len(neighbors)*sum([state[i]*dirac_delta([state[i], state[j]]) for j in neighbors])
    return xi

def get_nu(i, H, state):
    neighbors = H.nodes.neighbors(i)
    nu = 1/len(neighbors)*sum([dirac_delta([state[i], state[j]]) for j in neighbors])
    return nu

def get_nu_r(i, H, state, r):
    visited = set()
    visited.add(i)
    last_layer = set()
    last_layer.add(i)
    for _ in range(r):
        new_layer = set()
        for node in last_layer:
            new_layer.update(H.nodes.neighbors(node))
        last_layer = new_layer.difference(visited)
        visited.update(last_layer)
    if len(last_layer) == 0:
        return 0
    nu_r = 1/len(last_layer)*sum([dirac_delta([state[i], state[j]]) for j in last_layer])
    return nu_r



def get_rho_MF(l1, l2):
    if l2 == 0:
        rho1 = (l1-1)/l1
        if rho1>0:
            return rho1
        else:
            return 0
    else:
        rho1 = (l2-l1 + np.sqrt((l1-l2)**2 - 4.*l2*(1-l1)))/(2*l2)
        if rho1>0: 
            return rho1
        else:
            return 0
        
