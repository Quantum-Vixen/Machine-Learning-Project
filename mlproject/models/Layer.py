import numpy as np

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array



from .LayerInitializationStrategy import *
from .ActivationFunction import ActivationFunction


class Layer:
    """
    A Layer component of a NeuralNetwork.

    Attributes
    ----------
    unit_number: int
        The number of nodes/units in the Layer.
    init_strat: LayerInitializationStrategy
        The initialization strategy for the weights and biases of the Layer
    activation_function: ActivationFunction
        The function.
    """

    def __init__(self, number_of_units: int,
                 initialization_strategy: LayerInitializationStrategy,
                 activation_function: ActivationFunction):
        self.unit_number: int = number_of_units  # The number of nodes/units in the Layer.
        self.init_strat: LayerInitializationStrategy = initialization_strategy
        self.activation_function: ActivationFunction = activation_function

        # The values computed by the units, based on the outputs of the previous layer. Stored for later backprop.
        self.linear_output: Vector = None
        self.output: Vector = None
        
        # The layer preceding the current one in the Neural Network. The NN should connect layers during initialization.
        self.previous_layer: Layer = None
        self.next_layer: Layer = None

        # Weights and biases connecting the layer with the previous layer of the neural network.
        self.weights: Matrix = None; self.biases: Vector = None

        # A variable that needs to be computed from the delta of next layer in the
        # Backprop TrainingAlgorithm
        self.delta: Vector = None
        

    def initialize_weights(self) -> None:
        """
        Initialize the weights and biases of this Layer according to its init_strat.
        """
        self.weights, self.biases = self.init_strat.run(self.previous_layer.unit_number, self.unit_number)

    def compute_output(self):
        """
        Computes the output of this layer as activation_function(np.dot(input, weights) + biases), where
        the input is the output of the previous layer.
        Stores the output as well as just the linear_output np.dot(input, weights) + biases, as it's useful in typical training algorithms.
        """
        self.linear_output = np.dot(self.previous_layer.output, self.weights) + self.biases
        self.output: Vector = self.activation_function(self.linear_output)
        return self.output

class InputLayer(Layer):
    """
    The first Layer of a NeuralNetwork. It has no previous layer, and thus no weights and biases to connect it with.
    Its activation function is the Identity.

    Attributes
    ----------
    unit_number: int
        The number of nodes/units in the Layer.
    """
    def __init__(self, number_of_units: int):
        super().__init__(number_of_units, None, None)
        # An input layer has no previous layer to connect to, so attributes referring to a previous layer are deleted.
        del self.previous_layer, self.weights, self.biases, self.init_strat, self.activation_function
    
    def feed_input(self, value: Vector) -> None:
        """
        Sets the input (which is also the output) of the InputLayer, and thus of the whole NeuralNetwork, to value.
        """
        self.output: Vector = value
    
    def initialize_weights(self):
        raise NotImplementedError("InputLayer does not require weight initialization.")

    def compute_output(self) -> Vector:
        """
        Returns the output (which is also the input) of the InputLayer
        """
        return self.output

class HiddenLayer(Layer):
    def backward(self):
        """
        Computes the delta of this layer from the delta of the next layer
        as np.dot(self.next_layer.weights, self.next_layer.delta) * self.activation_function.derivative(self.linear_output)
        """
        self.delta = np.dot(self.next_layer.weights, self.next_layer.delta) * self.activation_function.derivative(self.linear_output)

class OutputLayer(Layer):
    def __init__(self, number_of_units: int,
                 initialization_strategy: LayerInitializationStrategy,
                 activation_function: ActivationFunction):
        super().__init__(number_of_units, initialization_strategy, activation_function)
        del self.next_layer