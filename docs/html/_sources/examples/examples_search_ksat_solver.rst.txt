
==================================================
Solve the Satisfiability Clause with Grover's
==================================================
Grover's search for a kSAT problem.

Given `clauses` below, should expect solution to be [1, 1, 1]

.. code-block::


	
	from skquantum.search import QuantumKSATSolver
	import numpy as np
	
	'''
	numpy array that encodes your clause
	ex. np.array([[1, 1, -1], [1, -1, 1]]) -->
	(v1 OR v2 OR ~v3) AND (v1 OR ~v2 OR v3)
	'''
	
	clauses = np.array([[1, 1, 1], 
	                    [-1, 1, 1], 
	                    [1, -1, 1], 
	                    [1, 1, -1],
	                    [-1, -1, 1],
	                    [-1, 1, -1],
	                    [1, -1, -1]])
	
	model = QuantumKSATSolver()
	model.fit(clauses)
	solution = model.solution
	
	print(solution)
