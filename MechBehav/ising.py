import numpy as np

def delta(x):
    """Delta function in higher dimensions

    Parameters
    ----------
    x : array
        Array of values

    Returns
    -------
    delta : bool
        True if all the values in x are equal, False otherwise
    """
    delta = np.all(x == x[0])
    return delta

def get_p1_and_p2(k1,k2,N):
    p2 = (2.*k2)/((N-1.)*(N-2.))
    p1 = (k1 - 2.*k2)/((N-1.)- 2.*k2)
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')

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


def ising_preserved(N, edges_list, triangles_list, J1, J2, n_steps, T=None, flip_all=False):
    if T is None:
        T = 1 / J1

    J1 = J1/(2*len(edges_list)/N)
    J2 = J2/(3*len(triangles_list)/N)

    energies = []
    magnetizations = []

    spins = np.random.choice([-1, 1], size=N)
    ts = np.zeros((N, n_steps))

    # Calculate the initial energy using the delta function
    energy_tot = -J1 * np.sum([spins[i] * spins[j] for i, j in edges_list]) - J2 * np.sum(
        [2 * delta([spins[i], spins[j], spins[k]]) - 1 for i, j, k in triangles_list]
    )

    for step in range(n_steps):
        i = np.random.randint(N)
        to_flip = [i]
        edges_neighbors = get_edges_neighbors(edges_list, i)
        triangles_neighbors = get_triangles_neighbors(triangles_list, i)
        pot_flip = [j for j in range(N) if j != i and j not in edges_neighbors and j not in triangles_neighbors]

        while pot_flip:
            j = np.random.choice(pot_flip)
            to_flip.append(j)
            edges_neighbors += get_edges_neighbors(edges_list, j)
            triangles_neighbors += get_triangles_neighbors(triangles_list, j)
            pot_flip = [k for k in range(N) if k not in to_flip and k not in edges_neighbors and k not in triangles_neighbors]

        a = 0

        for j in to_flip:
            # Compute the energy difference using the delta function
            delta_E = 0
            edges_neighbors = get_edges_neighbors(edges_list, j)
            triangles_neighbors = get_triangles_neighbors(triangles_list, j)
            for k in edges_neighbors:
                delta_E += 2 * J1 * spins[j] * spins[k]
            for triangle in triangles_neighbors:
                delta_E += 2 * J2 * (2 * delta([spins[triangle[0]], spins[triangle[1]], spins[triangle[2]]]) - 1)

            # Compute the acceptance probability
            if delta_E < 0:
                acceptance_prob = 1
            else:
                acceptance_prob = np.exp(-delta_E / T)  
            if np.random.rand() < acceptance_prob:
                a += 1
                spins[j] *= -1

                # Update the energy

        energy_tot = -J1 * np.sum([spins[i] * spins[j] for i, j in edges_list]) - J2 * np.sum(
            [2 * delta([spins[i], spins[j], spins[k]]) - 1 for i, j, k in triangles_list]
        )
        ts[:, step] = spins
        if flip_all:
            # random spins if they are all aligned
            if np.all(spins == 1) or np.all(spins == -1):
                print("All spins aligned, flipping spins")
                spins = np.random.choice([-1, 1], size=N)
                
        energies.append(energy_tot)
        magnetizations.append(np.sum(spins) / N)

    return ts, energies, magnetizations

def ising_broken(N, edges_list, triangles_list, J1, J2, n_steps, T=None, flip_all=False):
    if T is None:
        T = 1 / J1

    J1 = J1/(2*len(edges_list)/N)
    J2 = J2/(3*len(triangles_list)/N)

    energies = []
    magnetizations = []

    spins = np.random.choice([-1, 1], size=N)
    ts = np.zeros((N, n_steps))

    # Calculate the initial energy using the delta function
    energy_tot = -J1 * np.sum([spins[i] * spins[j] for i, j in edges_list]) - J2 * np.sum(
        [spins[i]*spins[j]*spins[k] for i, j, k in triangles_list]
    )

    for step in range(n_steps):
        i = np.random.randint(N)
        to_flip = [i]
        edges_neighbors = get_edges_neighbors(edges_list, i)
        triangles_neighbors = get_triangles_neighbors(triangles_list, i)
        pot_flip = [j for j in range(N) if j != i and j not in edges_neighbors and j not in triangles_neighbors]

        while pot_flip:
            j = np.random.choice(pot_flip)
            to_flip.append(j)
            edges_neighbors += get_edges_neighbors(edges_list, j)
            triangles_neighbors += get_triangles_neighbors(triangles_list, j)
            pot_flip = [k for k in range(N) if k not in to_flip and k not in edges_neighbors and k not in triangles_neighbors]

        for j in to_flip:
            # Compute the energy difference using the delta function
            delta_E = 0
            edges_neighbors = get_edges_neighbors(edges_list, j)
            triangles_neighbors = get_triangles_neighbors(triangles_list, j)
            for k in edges_neighbors:
                delta_E += 2 * J1 * spins[j] * spins[k]
            for triangle in triangles_neighbors:
                delta_E += 2 * J2 * spins[triangle[0]] * spins[triangle[1]] * spins[triangle[2]]

            # Compute the acceptance probability
            if delta_E < 0:
                acceptance_prob = 1
            else:
                acceptance_prob = np.exp(-delta_E / T)  
            if np.random.rand() < acceptance_prob:
                spins[j] *= -1

                # Update the energy

        energy_tot = -J1 * np.sum([spins[i] * spins[j] for i, j in edges_list]) - J2 * np.sum(
            [spins[i]*spins[j]*spins[k] for i, j, k in triangles_list]
        )
        ts[:, step] = spins
        if flip_all:
            # random spins if they are all aligned
            if np.all(spins == 1) or np.all(spins == -1):
                print("All spins aligned, flipping spins")
                spins = np.random.choice([-1, 1], size=N)
                
        energies.append(energy_tot)
        magnetizations.append(np.sum(spins) / N)

    return ts, energies, magnetizations

