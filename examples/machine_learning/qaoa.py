"""
================================================================
Solve the QAOA with gradient descent
================================================================
Builds a basic QAOA with PennyLane
"""
import pennylane as qml
from pennylane import numpy as np
from scipy.stats import unitary_group

n_features = 2
num_layers = 1
var = 0.05 * np.random.randn(num_layers, n_features*7)

dev = qml.device('default.qubit', wires=n_features)


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

    # for g,b in zip(betas, gammas):
    # random_U = unitary_group.rvs(n_features) # random_U = unitary_group.rvs(2**n_features)
    # # Normalize to form unitary
    # random_U = random_U / (np.linalg.det(random_U) ** (1/(2**n_features)))

    # expp = np.exp(random_U) / np.exp((np.linalg.det(random_U) ** (1/(2**n_features))))

    # The hamiltonian a hermitian operator. It is not a quantum gate.
    # A hermitian must be exponentiated to be a unitary. Unitaries are quantum gates.

    J = np.array([[0,1],[0,0]])
    steps = 10
    betas = np.random.uniform(0, np.pi*2, steps)
    gammas = np.random.uniform(0, np.pi*2, steps)
    for n_step in range(steps):

        # Run Driver Hamiltonian
        for i in range(n_features):
            for j in range(n_features):
                # phi = -J[i, j]
                # rz_1 = [[np.e**(-j*phi/2), 0],[0, np.e**(j*phi/2)]]
                # phi = 1.0
                # rz_2 = [[np.e**(-j*phi/2), 0],[0, np.e**(j*phi/2)]]

                # if i!=j:
                #     qml.QubitUnitary(np.kron(rz_1, rz_2), wires=[i, j])

                # qml.RZ(-J[i, j], wires=i)
                # qml.RZ(1.0, wires=j)

        # Run Mixer Hamiltonian
        for i in range(n_features):
            qml.RX(-betas[n_step], wires=i)

    # We to hit the quantum state with the product of the exponential of the Hamiltonian Driver
    return qml.expval.PauliX(0)

print(_qaoa_circuit(var, [], []))
