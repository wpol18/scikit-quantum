"""
Various functions and code for Hamiltonian manipulation

Author: Will Pol
"""

def maxcut_cost_ham(graph):
  """
  Expects a networkx graph and produces a Hermitian matrix denoted as the Hamiltonian
  """
    ham = QubitOperator('', 0.0)
    for i, j in graph.edges():
        operator_i = 'Z' + str(i)
        operator_j = 'Z' + str(j)
        ham += QubitOperator(operator_i + ' ' + operator_j, 0.5) + QubitOperator('', -0.5)
    return ham


def maxcut_mixer_ham(graph):
    """
    Creates a common mixer hamiltonian given a graph
    Graph  ==>  ∑ sigmaX_i
    """
    ham = QubitOperator('', 0.0)
    for i in graph.nodes():
        operator = 'X' + str(i)
        ham += QubitOperator(operator, -1.0)
    return ham
