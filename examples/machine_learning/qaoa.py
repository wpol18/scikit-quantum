"""
================================================================
Solve the QAOA with gradient descent
================================================================
Builds a basic QAOA with PennyLane
"""
import pennylane as qml
from pennylane import numpy as np
from scipy.stats import unitary_group
import networkx as nx

n_features = 2
num_layers = 1
var = 0.05 * np.random.randn(num_layers, n_features*7)

dev = qml.device('default.qubit', wires=n_features)


import numpy as np

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

print(hamiltonian())


def generate_random_unitary():
    random_U = unitary_group.rvs(n_features) # random_U = unitary_group.rvs(2**n_features)
    # Normalize to form unitary
    random_U = random_U / (np.linalg.det(random_U) ** (1/(2**n_features)))
    return random_U


@qml.qnode(dev)
def _qaoa_circuit(var, betas, gammas):
    # Initialize circuit such that all solutions are equally likely.
    for i in range(n_features):
        qml.Hadamard(wires=i)

    # QAOA relies on trotterization of the adiabatic pathway.
    # Recall that the Adiabatic Theorem says that we can start with a system at
    # the global minimum of an easy optimization problem and if we evolve it to another problem,
    # we will be at the global minimum of the second problem. It only works if we evolve
    # it slow enough.
    # For the theorem to hold, there must be an energy gap between the ground state and the first excited state.

 
    # expp = np.exp(random_U) / np.exp((np.linalg.det(random_U) ** (1/(2**n_features))))

    # The hamiltonian a hermitian operator. It is not a quantum gate.
    # A hermitian must be exponentiated to be a unitary. Unitaries are quantum gates.

    H0, H1 = hamiltonian()
    graph = nx.gnp_random_graph(4, 0.5)
    H1 = graph_cuts(graph)

    J = np.array([[0,1],[0,0]])
    steps = 10
    betas = np.random.uniform(0, np.pi*2, steps)
    gammas = np.random.uniform(0, np.pi*2, steps)
    for n_step in range(steps):

        # Run Driver Hamiltonian
        for i in range(n_features):
            for j in range(n_features):

                # We to hit the quantum state with the product of the exponential of the Hamiltonian Cost
                if i!=j:
                    qml.QubitUnitary(np.e**H0, wires=[i, j])

        # Run Mixer Hamiltonian
        for i in range(n_features):
            qml.QubitUnitary(np.e**H1, wires=[i])

    return qml.expval.PauliX(0)

print(_qaoa_circuit(var, [], []))
