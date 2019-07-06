import numpy as np

from skquantum.machine_learning import QuantumMultiLayerPerceptron


def test_qmlp_trains_for_a_little_bit():

	X = [[0,0, 0],[1,0, 0],[0,1, 0],[1,1, 0]]
	Y = [0,1,1,0]

	clf = QuantumMultiLayerPerceptron(iterations=1)
	clf.fit(X, Y)

	assert len(clf.predict(X)) == 4
