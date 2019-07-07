import networkx as nx
import numpy as np

from scipy.stats import unitary_group


def graph_cuts(graph: nx.Graph) -> np.ndarray:
    """For the given graph, return the cut value for all binary assignments
    of the graph. Taken from QuantumFlow
    """

    N = len(graph)
    diag_hamiltonian = np.zeros(shape=([2]*N), dtype=np.double)
    for q0, q1 in graph.edges():
        for index, _ in np.ndenumerate(diag_hamiltonian):
            if index[q0] != index[q1]:
                weight = graph[q0][q1].get('weight', 1)
                diag_hamiltonian[index] += weight

    return diag_hamiltonian


def hamiltonian(omega=1.0, ampl0=0.2):
    """A basic two-level-system Hamiltonian for QAOA

    Args:
        omega (float): energy separation of the qubit levels
        ampl0 (float): constant amplitude of the driving field
    """
    H0 = -0.5 * omega * np.array([[-1, 0], [0, 1]], dtype=np.complex128)
    H1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    return H0, H1


def generate_random_unitary(size):
    random_U = unitary_group.rvs(2**n_features)
    # Normalize to form unitary
    random_U = random_U / (np.linalg.det(random_U) ** (1/(2**n_features)))
    return random_U
