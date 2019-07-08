"""
================================================================
Solve the Weighted MaxCut with QAOA
================================================================
QAOA relies on trotterization of the adiabatic pathway. QAOA relies on the Adiabatic Theorem
which says that we can start with a system at the global minimum of an easy optimization problem 
and if we evolve it to another problem, it will be in the global optima of the second problem.
It's not hard to see that this is a powerful law of nature that can be harnessed to solve a wide range of computationl problems.

In this example, we solve the weighted max cut problem on a random networkx graph.
"""
from skquantum.combinatorial import WeightedMaxCut
import random
import networkx as nx

nodes = 8

graph = nx.Graph()
for i in range(nodes):
    for j in range(nodes):
        graph.add_edge(i, j, weight=random.random())

problem = WeightedMaxCut()
problem.fit(graph)

print("Found Beta and Gammas", problem.var_gd)
print("Found Beta and Gammas", problem.solution)
