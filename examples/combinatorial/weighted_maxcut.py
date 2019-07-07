"""
================================================================
Solve the QAOA with gradient descent
================================================================
Builds a basic QAOA with PennyLane
"""
import sys
sys.path.append("/Users/mat/Documents/scikit-quantum")

from skquantum.combinatorial import WeightedMaxCut
import random
import networkx as nx

nodes = 8

graph = nx.Graph()
for i in range(nodes):
    for j in range(nodes):
        graph.add_edge(i, j, weight=random.random())

model = WeightedMaxCut()
model.fit(graph)

print("Found Beta and Gammas", model.var_gd)
