# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variational Quantum Eigensolver (VQE)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~
# imports
# ~~~~~~~~~~~
import pennylane as qml
from pennylane.optimize import GradientDescentOptimizer
from pennylane import numpy as np

from pyquil.paulis import sZ, sY, sX, ID, PauliTerm, PauliSum
from projectq.ops import QubitOperator

from scipy.optimize import minimize, optimize



# ~~~~~~~~~~~~~~~~~~~~~~~~
# Making the VQE class:
# ~~~~~~~~~~~~~~~~~~~~~~~~

class VQE():
    
    def __init__(self, optimizer=minimize, optimizer_kwargs={'method': 'Nelder-Mead'}):
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        '''
        Initialize the VQE object by setting the `optimizer` you want to use
        and any `kwargs` necessary for the optimizer of choice. Supports `scipy.optimize` minimizers
        and Pennylane's built-in `GradientDescentOptimizer` 
        
        
        Parameters:
            `optimizer`: choice of optimizer for classical optimization loop.
                         Supports `scipy.optimize` minimizers
                         and Pennylane's built-in `GradientDescentOptimizer`
            `optimizer_kwargs`: specifications for `optimizer`; a dictionary for
                                `scipy.optimize` and integer for `GradientDescentOptimizer`
                                
        '''
        

    def fit(self, ansatz, initial_params, H, device=None):

        '''
        Main function: applies the VQE algorithm with the given parameters
        and arguments using the optimizer set up in `self.__init__(...)`. 
        Finds the parameters that create a state using `ansatz` with the lowest possible 
        expectation value for the operator hamiltonian `H`.
        
        Parameters:
            `ansatz`: circuit for creating an initial starting state
            `initial_params`: initial parameters to feed `ansatz` function
            `H`: hamiltonian that we want to calculate the lowest energy of
            `device`: backend to run algorithm on, be it a simulator
                      or an actual QPU
        '''    
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # get list of coefficients and wires:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        coefficients = []
        wires = []
        identity_coefficients = []
        
        # assuming provided H comes from PyQuil.... need to add generality
        if isinstance(H, (PauliTerm, PauliSum)):
            if len(H) != 1:
                for term in H.terms:
                    coef = term.coefficient
                    wire = term.get_qubits()
                    if term.pauli_string() == '': 
                        identity_coefficients.append(coef)
                    else:
                        coefficients.append(coef)
                        wires += wire
            elif len(H) == 1:
                coef = H.coefficient
                wire = H.get_qubits()
                if H.pauli_string() == '': 
                    identity_coefficients.append(coef)
                else:
                    coefficients.append(coef)
                    wires += wire
                
            
        
        # assuming H comes from ProjectQ...
        elif isinstance(H, QubitOperator):
            for term, coef in H.terms.items():
                if term == ():
                    identity_coefficients.append(coef)
                else:
                    coefficients.append(coef)
                    for wire, _ in term:
                        wires.append(wire)
                    
        # remove duplicates from wires to make new wire_labels
        wire_labels = list(dict.fromkeys(wires))
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate default device using list of wires we just found:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if device == None:
            device = qml.device('default.qubit', wires=len(wire_labels))

            
        # ~~~~~~~~~~~~~~~~~~~~~~
        # get expvals using vqe
        # ~~~~~~~~~~~~~~~~~~~~~~
        var = initial_params
        
        @qml.qnode(device)
        def vqe(var):
            # Initialize circuit with state preparation routine
            ansatz(var, wires)
            
            # create empty list for expectation values
            expvals = []
            
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # measure state in appropriate basis
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # assuming PyQuil Ham:
            if isinstance(H, (PauliTerm, PauliSum)):
                if len(H) != 1:
                    for term in H.terms:
                        wire = term.get_qubits()
                        if term.pauli_string() == '': 
                            pass # if term is Identity, ignore for measurement
                        elif term.pauli_string() == 'Z':
                            expvals.append(qml.expval.PauliZ(wire))
                        elif term.pauli_string() == 'X':
                            expvals.append(qml.expval.PauliX(wire))
                        elif term.pauli_string() == 'Y':
                            expvals.append(qml.expval.PauliY(wire))
                elif len(H) == 1:
                    wire = H.get_qubits()
                    if H.pauli_string() == '': 
                        pass # if term is Identity, ignore for measurement
                    elif H.pauli_string() == 'Z':
                        expvals.append(qml.expval.PauliZ(wire))
                    elif H.pauli_string() == 'X':
                        expvals.append(qml.expval.PauliX(wire))
                    elif H.pauli_string() == 'Y':
                        expvals.append(qml.expval.PauliY(wire))
                    
                    
                    
                        
            # assuming ProjectQ Ham:
            elif isinstance(H, QubitOperator):
                    for term, _ in H.terms.items():
                        if term == (): 
                            pass # if term is Identity, ignore for measurement
                            
                        # Else, measure in correct basis
                        else:
                            for wire, op in term:
                                if op == 'X':
                                    expvals.append(qml.expval.PauliX(wire))
                                elif op == 'Y':
                                    expvals.append(qml.expval.PauliY(wire))
                                elif op == 'Z':
                                    expvals.append(qml.expval.PauliZ(wire))
            return expvals
        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add all Hamiltonian energy contributions for one big expval
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def cost(var):
            expvals_with_coefs = [expval*coef.real for expval, coef in zip(vqe(var), coefficients)]
            identity_coefs = [coef.real for coef in identity_coefficients]
            total_sum = sum(expvals_with_coefs) + sum(identity_coefs)
            return total_sum

        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Classical optimization loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        opt = self.optimizer
        
        # can use `scipy.optimize.minimize` or Pennylane's built-in `GradientDescentOptimizer`
        if isinstance(opt, GradientDescentOptimizer) and isinstance(self.optimizer_kwargs, int):
            self.var_gd = [var]
            gradient_descent_n_iterations = self.optimizer_kwargs
            for it in range(gradient_descent_n_iterations):
                var = opt.step(cost, var)
                self.var_gd.append(var)

            self.solution = cost(self.var_gd[-1])
            
        else:
            final = opt(cost, var, **self.optimizer_kwargs)
            self.solution = final.fun
