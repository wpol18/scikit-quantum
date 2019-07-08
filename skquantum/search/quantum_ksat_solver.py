# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grover's algorithm for kSAT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import pennylane as qml
from pennylane import numpy as np


def T_gate(wire):
    qml.PhaseShift(np.pi/4, wires=wire)

def T_dag(wire):
    qml.PhaseShift(-np.pi/4, wires=wire)

def Toffoli(wires):
    # wires is a list of three qubits: [c, c, t]
    qml.Hadamard(wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    T_dag(wires[2])
    qml.CNOT(wires=[wires[0], wires[2]])
    T_gate(wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    T_dag(wires[2])
    qml.CNOT(wires=[wires[0], wires[2]])
    T_gate(wires[1])
    T_gate(wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.Hadamard(wires[2])
    T_gate(wires[0])
    T_dag(wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    
    
def n_bit_Toffoli(wires, n):
    '''
    create an n-bit Toffoli, given 'n' and wires, where
    wires = [c_0, ..., c_n, a_0, ..., a_n-1, target]
    c_i = control qubits, a_i = ancilla
    '''
    ctrl = wires[: n] # control qubits
    anc = wires[n: n + n - 1] # ancillas
    tgt = wires[-1] # target
    
    # compute
    Toffoli([ctrl[0], ctrl[1], anc[0]])
    for i in range(2, n):
        Toffoli([ctrl[i], anc[i-2], anc[i-1]])
        
    # copy
    qml.CNOT(wires=[anc[n-2], tgt])
    
    # uncompute
    for i in range(n-1, 1, -1):
        Toffoli([ctrl[i], anc[i-2], anc[i-1]])
    Toffoli([ctrl[0], ctrl[1], anc[0]])





def n_controlled_pauli_z(wires, n):
    '''
    create an n-bit Toffoli, given 'n' and wires, where
    wires = [c_0, ..., c_n, a_0, ..., a_n-1, target]
    c_i = control qubits, a_i = ancilla
    '''
    ctrl = wires[: n] # control qubits
    anc = wires[n: n + n - 1] # ancillas
    tgt = wires[-1] # target
    
    # compute
    Toffoli([ctrl[0], ctrl[1], anc[0]])
    for i in range(2, n):
        Toffoli([ctrl[i], anc[i-2], anc[i-1]])
        
    # copy
    qml.CZ(wires=[anc[n-2], tgt])
    
    # uncompute
    for i in range(n-1, 1, -1):
        Toffoli([ctrl[i], anc[i-2], anc[i-1]])
    Toffoli([ctrl[0], ctrl[1], anc[0]])


def QuantumOR(clause, qubit_wires, ancilla_wires):
    '''
    Construct an OR based on input clause, a numpy array
    with 1 and -1 values, -1 indicating a NOT
    '''
    for i in range(len(clause)):
        if clause[i].val == 1:
            qml.PauliX(wires=qubit_wires[i])
    qml.PauliX(wires=ancilla_wires[-1])
    total_wires = qubit_wires + ancilla_wires
    if len(clause) == 2:
        Toffoli(total_wires) 
    else:
        n_bit_Toffoli(total_wires, len(clause))
    for i in range(len(clause)):
        if clause[i].val == 1:
            qml.PauliX(wires=qubit_wires[i])


class QuantumKSATSolver():
    '''
    Grover's algorithm for finding the variable combination that
    satisfies some series of clauses, k variables per clause

    ex. find v1, v2, v3 = {0, 1} that satisfies
    (v1 OR v2 OR ~v3) AND (~v1 OR v2 OR v3) AND (v1 OR ~v2 OR v3)
    AND (~v1 OR ~v2 OR v3) AND (v1 OR ~v2 OR v3)
    '''

    def __init__(self):
        pass
    
    def fit(self, k_sat, num_its=None, device=None):
        '''
        Paramters:

            `k_sat`: numpy array that encodes your clause
            ex. np.array([[1, 1, -1], [1, -1, 1]]) -->
                (v1 OR v2 OR v3) AND (v1 OR ~v2 OR v3)

            `device`: backend to run algorithm on, be it a simulator
            or an actual QPU
        '''
        
        # prep some useful variables
        num_rows = k_sat.shape[0]
        num_variables = k_sat.shape[1]
        variable_wires = [i for i in range(num_variables)]
        clause_wires = [(i + num_variables) for i in range(num_rows)]
        ancilla_wires = [(i + num_variables + num_rows) for i in range(num_rows - 1)]
        oracle_wire = [num_variables + num_rows + len(ancilla_wires)]
        
        # set defaults
        if device == None:
            device = qml.device('default.qubit', wires=(num_variables + num_rows + len(ancilla_wires) + 1))
        if num_its == None:
            num_its = num_its= int(np.pi*np.sqrt(2**k_sat.shape[1])/4)
        
        @qml.qnode(device)
        def search(k_sat):


            # prep some useful variables
            num_rows = k_sat.shape[0]
            num_variables = k_sat.shape[1]
            variable_wires = [i for i in range(num_variables)]
            clause_wires = [(i + num_variables) for i in range(num_rows)]
            ancilla_wires = [(i + num_variables + num_rows) for i in range(num_rows - 1)]
            oracle_wire = [num_variables + num_rows + len(ancilla_wires)]

            # place variable qubits in equal superposition:
            for qubit in variable_wires:
                qml.Hadamard(qubit)

            # prepare oracle qubit 1/sqrt(2)[|0> - |1>]
            qml.PauliX(oracle_wire)
            qml.Hadamard(oracle_wire)

            # Grover iterations:
            for _ in range(num_its):

                # Compute f(x) as a series of unitaries
                for i in range(num_rows):
                    QuantumOR(k_sat[i], variable_wires, ancilla_wires[: num_variables - 1] + [clause_wires[i]])

                # "tag" the correct state
                n_bit_Toffoli(clause_wires + ancilla_wires + oracle_wire, num_rows)

                # Uncompute
                for i in range(num_rows-1, -1, -1):
                    QuantumOR(k_sat[i], variable_wires, ancilla_wires[: num_variables - 1] + [clause_wires[i]])

                # Amplitude amplification
                for qubit in variable_wires:
                    qml.Hadamard(qubit)
                    qml.PauliX(qubit)
                if num_variables == 2:
                    qml.CZ(wires=variable_wires)
                else:
                    n_controlled_pauli_z(variable_wires[: -1] + ancilla_wires[: len(variable_wires) - 2] + [variable_wires[-1]], num_variables-1)
                for qubit in variable_wires:
                    qml.PauliX(qubit)
                    qml.Hadamard(qubit)

            return [qml.expval.PauliZ(qubit) for qubit in variable_wires] 
        result = [round(expval) for expval in search(k_sat)]
        
        
        self.solution = [int(-0.5*val + 0.5) for val in result]
        
