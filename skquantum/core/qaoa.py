import pennylane as qml

from pennylane.optimize import GradientDescentOptimizer
from pennylane import numpy as np


class QAOA():
    def __init__(self, gradient_descent_n_iterations=10, qaoa_steps=3):
        self.gradient_descent_n_iterations = gradient_descent_n_iterations
        self.qaoa_steps = qaoa_steps

    def fit(self, H1):

        n_features = len(H1)

        self.dev = qml.device('projectq.simulator', wires=n_features)
        @qml.qnode(self.dev)
        def qaoa(var):
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

            # H0, H1 = hamiltonian()
            # H0 = sum([qml.PauliX(i) for i in range(n_features)])

            # The number of nodes in the graph must equal the number of qubits
            # graph = nx.gnp_random_graph(n_features, 0.5)

            # H1 = graph_cuts(graph)

            # import matplotlib.pyplot as plt
            # nx.draw(graph)
            # plt.draw()
            # plt.show()
            betas = var[:int(len(var)/2)]
            gammas = var[int(len(var)/2):]
            # betas = self.betas
            # gammas = self.gammas

            for n_step in range(len(betas)):

                # Run Driver Hamiltonian
                for i in range(n_features):
                    for j in range(n_features):

                        # We to hit the quantum state with the product of the exponential of the Hamiltonian Cost
                        if i!=j:
                            qml.QubitUnitary(np.e**(H1 * -1j * betas[n_step].val), wires=[i, j])

                # Run Mixer Hamiltonian
                for q in range(n_features): qml.RX(gammas[n_step], wires=q)

            return [qml.expval.PauliZ(n) for n in range(n_features)]

        def cost(var):
            return sum(qaoa(var))

        opt = GradientDescentOptimizer(0.1)

        steps = self.qaoa_steps # QAOA Steps
        var = np.random.uniform(0, np.pi*2, 2*steps)
        self.var_gd = [var]
        for it in range(self.gradient_descent_n_iterations):
            var = opt.step(cost, var)
            self.var_gd.append(var)

