"""Continuous-variable quantum neural network example.
In this demo we implement the photonic quantum neural net model
from Killoran et al. (arXiv:1806.06871) with the example
of function fitting.

Adapted from https://github.com/XanaduAI/pennylane/blob/8c54d5c16d8a455a757655ced84d4178da866097/examples/CV2_quantum-neural-net.py
"""

import pennylane as qml
import itertools
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

n_features = 3

dev = qml.device('strawberryfields.fock', wires=n_features, cutoff_dim=10)


def layer(v):
    """ Single layer of the quantum neural net.
    Args:
        v (array[float]): array of variables for one layer
    """

    used_variable_count = 0
    def _get_and_increment():
        nonlocal used_variable_count
        current_variable_count = used_variable_count
        used_variable_count+=1        
        return current_variable_count

    for i in range(n_features):
        # Matrix multiplication of input layer
        # Matrix-vector multiplications, which are used heavily in artificial neural networks,
        # can be done efficiently in photonic circuits
        qml.Rotation(v[_get_and_increment()], wires=i)
        qml.Squeezing(v[_get_and_increment()], 0., wires=i)
        qml.Rotation(v[_get_and_increment()], wires=i)

        # Bias
        qml.Displacement(v[_get_and_increment()], 0., wires=i)

    # Implements an Interferometer using beam splitters
    # The simplest explanation is that the beam-splitter acts as a classical coin-flip, randomly
    # sending each photon one way or the other.
    # BeamSplitters are parameterized by BeamSplitters(variable1, variable2)
    # Just like a normal beam splitter / prism -- it breaks the beam of light into two.
    for n, wires in enumerate(list(itertools.combinations(range(n_features), 2))):
        qml.Beamsplitter(v[_get_and_increment()],v[_get_and_increment()], wires=wires)

    # Element-wise nonlinear transformation
    for i in range(n_features):
        qml.Kerr(v[_get_and_increment()], wires=i)


@qml.qnode(dev)
def quantum_neural_net(var, x=None):
    """The quantum neural net variational circuit.
    Args:
        var (array[float]): array of variables
        x (array[float]): single input vector
    Returns:
        float: expectation of Homodyne measurement on Mode 0
    """
    # Encode input x into quantum state
    for i in range(n_features):
        qml.Displacement(x[i], 0., wires=i)

    # "layer" subcircuits
    for v in var:
        layer(v)

    return qml.expval.X(0)


def square_loss(labels, predictions):
    """ Square loss function
    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: square loss
    """
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)

    return loss


def cost(var, features, labels):
    """Cost function to be minimized.
    Args:
        var (array[float]): array of variables
        features (array[float]): 2-d array of input vectors
        labels (array[float]): 1-d array of targets
    Returns:
        float: loss
    """
    # Compute prediction for each input in data batch
    preds = [quantum_neural_net(var, x=x) for x in features]

    return square_loss(labels, preds)


class QuantumMultiLayerPerceptron():
    def __init__():
        pass

    def predict(X):
        preds = [quantum_neural_net(var, x=x) for x in X]
        for i in range(len(preds)):
            print("X: {0} | Predicted: {1} | Label: {2}".format(X[i], preds[i], Y[i]))

    def fit(X, Y):
        # X = [[0,0, 0],[1,0, 0],[0,1, 0],[1,1, 0]]
        # Y = [0,1,1,0]

        # initialize weights
        np.random.seed(0)
        num_layers = 1

        # Each hidden node in each layer is paramterized by 7 variables in order.
        # We optimize these paramaters opt.step(cost_function, variables)
        # We can't increase the number of hidden nodes per layer because we would need more qumodes.
        var_init = 0.05 * np.random.randn(num_layers, n_features*7)

        # create optimizer
        opt = AdamOptimizer(0.1, beta1=0.9, beta2=0.999)

        # train
        var = var_init
        iterations = 400
        for it in range(iterations):
            var = opt.step(lambda v: cost(v, X, Y), var)
            if it:
                print("{:0.2f}% Iter: {:5d} | Cost: {:0.7f} ".format(((it+1)/iterations)*100, it + 1, cost(var, X, Y)))
