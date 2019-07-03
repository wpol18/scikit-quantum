"""
================================================================
Solve the XOR problem with a Quantum Multi-Layer Perceptron
================================================================
This builds a basic multi-layer perceptron and trains it on a XOR problem.
"""
from skquantum.machine_learning import QuantumMultiLayerPerceptron

X = [[0,0, 0],[1,0, 0],[0,1, 0],[1,1, 0]]
Y = [0,1,1,0]

clf = QuantumMultiLayerPerceptron()
clf.fit(X, Y)

print(clf.predict(X))
