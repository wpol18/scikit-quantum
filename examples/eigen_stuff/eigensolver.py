"""
=====================================================================
Solve for the lowest energy eigenvalue of a Hamiltonian with the VQE
=====================================================================
Variational Quantum Eigensolver (VQE) finds the lowest energy eigenvalue 
for some Hamiltonian, H. VQE relies on the variational principle, which 
says that the expectation value of some observable, H, will always be 
greater than or equal to H's lowest energy eigenvalue. 

We feed VQE a parameterized quantum state preparation routine (ansatz),
an initial set of parameters for our ansatz, an optimizer for our
classical optimization loop to find the best parameter choice for our
ansatz, and the Hamiltonian whose eigenvalue we wish to find.

(skquantum's VQE currently supports `scipy.optimize.minimize` optimizers
and Pennylane's built-in `GradientDescentOptimizer`. We can currently specify
our Hamiltonian using either PyQuil or ProjectQ)


Given `H_example` below, we expect the solution to be approximately -8
"""

# imports
from skquantum.core.vqe import VQE
from pennylane.optimize import GradientDescentOptimizer
import pennylane as qml
from pyquil.paulis import sZ
import numpy as np



# instantiate Hamiltonian:
H_example = 3*sZ(0) + 5*sZ(1)


# state preparation function
def ansatz_example(var, wire):
    qml.RX(var[0], wires=wire[0])
    qml.RY(var[1], wires=wire[1])

    
# initial parameter choice to feed into ansatz above
init_angles = np.random.uniform(0, 2*np.pi, 2)


# specify which optimizer and kwargs we want to use
opt = GradientDescentOptimizer(0.1)
gd_steps = 20

# instantiate VQE, fit, print solution (should be around -8):
model = VQE(optimizer=opt, optimizer_kwargs=gd_steps)
model.fit(ansatz=ansatz_example, initial_params=init_angles, H=H_example)

print(model.solution)
