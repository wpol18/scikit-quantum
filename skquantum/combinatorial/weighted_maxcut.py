import numpy as np

from skquantum.core.qaoa import QAOA
from skquantum.core.hamiltonian import graph_cuts

class WeightedMaxCut():
	def __init__(self):
		pass

	def fit(self, graph):
		H1 = graph_cuts(graph)
		H1 = np.matrix(H1.reshape(int(np.sqrt(H1.size)), int(np.sqrt(H1.size))).T)
		H1 = np.asarray(H1)
		qaoa = QAOA()
		qaoa.fit(H1)

		self.var_gd = qaoa.var_gd
