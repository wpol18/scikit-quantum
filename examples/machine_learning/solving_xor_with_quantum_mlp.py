"""
================================================================
Solve the XOR problem with a Quantum Multi-Layer Perceptron
================================================================
This builds a basic multi-layer perceptron and trains it on a XOR problem.
"""
import numpy as np

from skquantum.machine_learning import QuantumMultiLayerPerceptron

X = [[0,0, 0],[1,0, 0],[0,1, 0],[1,1, 0]]
Y = [0,1,1,0]

clf = QuantumMultiLayerPerceptron(iterations=10)
clf.fit(X, Y)

print(clf.predict(X))
