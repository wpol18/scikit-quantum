"""
Code derived from the CV-variable quantum neural networks from Xanadu: https://arxiv.org/abs/1806.06871
"""

import pennylane as qml
import itertools
import sklearn

from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer


def layer(v, n_features=None):
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


class QuantumMultiLayerPerceptron(sklearn.base.BaseEstimator):
    def __init__(self, optimizer=None, iterations=400, num_layers=1):
        # Hyper-parameters
        self.optimizer = optimizer or AdamOptimizer(0.1, beta1=0.9, beta2=0.999)
        self.iterations = iterations
        self.num_layers = num_layers

        # Internal Variables
        self._quantum_neural_net = None
        self.n_features = None
        self.var = None


    def _cost(self, var, features, labels):
        """Cost function to be minimized.
        Args:
            var (array[float]): array of variables
            features (array[float]): 2-d array of input vectors
            labels (array[float]): 1-d array of targets
        Returns:
            float: loss
        """
        preds = [self._quantum_neural_net(var, x=x) for x in features]

        return square_loss(labels, preds)


    def predict(self, X):
        preds = [self._quantum_neural_net(self.var, x=x) for x in X]
        return preds


    def fit(self, X, Y):
        self.n_features = len(X[0])
        self.dev = qml.device('strawberryfields.fock', wires=self.n_features, cutoff_dim=10)

        @qml.qnode(self.dev)
        def _quantum_neural_net(var, x=None):
            """The quantum neural net variational circuit.
            Args:
                var (array[float]): array of variables
                x (array[float]): single input vector
            Returns:
                float: expectation of Homodyne measurement on Mode 0
            """

            # Encode input x into quantum state
            for i in range(self.n_features):
                qml.Displacement(x[i], 0., wires=i)

            # "layer" subcircuits
            for v in var:
                layer(v, n_features=self.n_features)

            return qml.expval.X(0)

        self._quantum_neural_net = _quantum_neural_net

        # Each hidden node in each layer is paramterized by 7 variables in order.
        # We optimize these paramaters opt.step(cost_function, variables)
        # We can't increase the number of hidden nodes per layer because we would need more qumodes.
        var_init = 0.05 * np.random.randn(self.num_layers, self.n_features*7)

        # create optimizer
        opt = self.optimizer

        # train
        self.var = var_init
        iterations = self.iterations
        for it in range(iterations):
            self.var = opt.step(lambda v: self._cost(v, X, Y), self.var)
            if it:
                print("{:0.2f}% Iter: {:5d} | Cost: {:0.7f} ".format(((it+1)/iterations)*100, it + 1, self._cost(self.var, X, Y)))
