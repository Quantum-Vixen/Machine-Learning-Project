from mlproject.utils.ListOfArrays import ListOfArrays, ListOfMatrices, ListOfVectors
from mlproject.models.MachineLearningModel import NeuralNetwork
from mlproject.models.Layer import Layer
import numpy as np

class RegularizationTerm:
    def __init__(self):
        self.network: NeuralNetwork = None  # Has to be initialized after initializing RegularizationTerm, using the set_network method.

    def set_network(self, network: NeuralNetwork) -> None:
        self.network = network

    def __call__(self) -> float:
        pass

    def gradient(self) -> tuple[ListOfArrays, ListOfArrays]:
        pass

class NoRegularization(RegularizationTerm):
    def __init__(self):
        pass

    def __call__(self) -> float:
        return 0
    
    def gradient(self) -> tuple[ListOfArrays, ListOfArrays]:
        layers = self.network.layers_with_weights
        return ListOfArrays([np.zeros_like(l.weights) for l in layers]), ListOfArrays([np.zeros_like(l.biases) for l in layers])
    
    def __str__(self):
        return "No reg."

class Tikhonov(RegularizationTerm):
    """
    A regularization penalty term of the form constant*(sum of squares of weights and biases).

    Parameters
    ----------
    penalty: float
        The constant factor multiplying the sum of squares.
    
    Attributes
    ----------
    penalty: float
        The constant factor multiplying the sum of squares.
    network: NeuralNetwork
        The NeuralNetwork that weights and biases are read from.
    """
    def __init__(self, penalty: float):
        self.penalty: float = penalty
        self.network: NeuralNetwork = None

    def __call__(self) -> float:
        layers: list[Layer] = self.network.layers_with_weights
        weights_term = np.sum([np.sum(layer.weigths**2) for layer in layers])  # The sum of squares of all the weights in the NN.
        biases_term = np.sum([np.sum(layer.biases**2) for layer in layers])
        return self.penalty * (weights_term + biases_term) / 2

    def gradient(self) -> tuple[ListOfArrays, ListOfArrays]:
        layers: list[Layer] = self.network.layers_with_weights
        gradient_on_weights: ListOfArrays = ListOfArrays([-self.penalty * l.weights for l in layers])
        gradient_on_biases: ListOfArrays = ListOfArrays([-self.penalty * l.biases for l in layers])
        return gradient_on_weights, gradient_on_biases
    
    def __str__(self):
        return f"Tikhonov({self.penalty})"

class Lasso(RegularizationTerm):
    def __init__(self, penalty: float, epsilon: float = 0):
        super().__init__()
        self.penalty: float = penalty
        self.eps: float = epsilon
    
    def __call__(self) -> float:
        layers: list[Layer] = self.network.layers_with_weights
        weights_term: float = np.sum([np.sum(np.sqrt(layer.weights ** 2 + self.eps)) for layer in layers])
        biases_term: float = np.sum([np.sum(np.sqrt(layer.biases ** 2 + self.eps)) for layer in layers])
        return self.penalty * (weights_term + biases_term)
    
    def gradient(self) -> tuple[ListOfMatrices, ListOfVectors]:
        layers: list[Layer] = self.network.layers_with_weights
        gradient_on_weights: ListOfMatrices = ListOfMatrices([-self.penalty * (l.weights**2 + self.eps)**(-0.5) * l.weights for l in layers])
        gradient_on_biases: ListOfVectors = ListOfVectors([-self.penalty * (l.biases**2 + self.eps)**(-0.5) * l.biases for l in layers])
        return gradient_on_weights, gradient_on_biases
    
    def __str__(self):
        return f"Lasso({self.penalty}, epsilon = {self.eps})"