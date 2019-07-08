import numpy as np
import networkx as nx

from skquantum.core.qaoa import QAOA
from qiskit.aqua.translators.ising.max_cut import get_max_cut_qubitops

class QuantumWeightedMaxCut():
	def __init__(self):
		pass

	def fit(self, graph):
		H1 = get_max_cut_qubitops(nx.adjacency_matrix(graph).toarray())
		H1[0].to_matrix()
		H1 = H1[0]._matrix.todense()
		H1 = np.asarray(H1)
		qaoa = QAOA()
		qaoa.fit(H1)

		self.var_gd = qaoa.var_gd
		self.solution = qaoa.solution
