import pennylane as qml
import itertools
import math

from pennylane.optimize import GradientDescentOptimizer
from pennylane import numpy as np


class QAOA():
    def __init__(self, gradient_descent_n_iterations=3, qaoa_steps=3):
        self.gradient_descent_n_iterations = gradient_descent_n_iterations
        self.qaoa_steps = qaoa_steps

    def fit(self, H1):

        # The number of nodes in the graph must equal the number of qubits
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

            # The hamiltonian a hermitian operator. It is not a quantum gate.
            # A hermitian must be exponentiated to be a unitary. Unitaries are indeed quantum gates.

            betas = var[:int(len(var)/2)]
            gammas = var[int(len(var)/2):]

            for n_step in range(len(betas)):

                # Run Cost Hamiltonian
                for n, wires in enumerate(list(itertools.combinations(range(int(n_features)), int(math.log(n_features, 2)) ))):
                    qml.QubitUnitary(np.e**(H1 * -1j * betas[n_step].val), wires=wires)

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

        self.solution = qaoa(self.var_gd[-1])
