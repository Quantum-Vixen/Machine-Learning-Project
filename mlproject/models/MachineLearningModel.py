"""
This module contains the NeuralNetwork class and NeuralNetworkArchitecture, which is an auxiliary class to constructor NeuralNetwork via the alternative .fromArchitecture constructor.
The standard __init__, instead, allows for more control in finer manipulations of the NN architecture.
"""

import numpy as np
import pandas as pd

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array


from .ActivationFunction import ActivationFunction
from .LayerInitializationStrategy import LayerInitializationStrategy
from .Layer import Layer, InputLayer, HiddenLayer, OutputLayer
from .DataPreprocessingMethod import DataPreprocessingMethod, NoPreprocessing



class MachineLearningModel:
    pass

class NeuralNetworkArchitecture:
    """
    An utility class for storing information about the number of Layers and of units for each Layer of a NeuralNetwork, as well as other
    useful data.

    Used for an alternative constructor of NeuralNetwork when no fine-control for initialization is needed.
    """
    def __init__(self, sizes_of_layers: list[int], activation_functions: list[ActivationFunction], initialization_strategy: LayerInitializationStrategy):
        self.sizes_of_layers: list[int] = sizes_of_layers
        self.activation_functions: list[ActivationFunction] = activation_functions
        self.initialization_strategy: LayerInitializationStrategy = initialization_strategy
    
    def __str__(self):
        network_shape: str = "(" + ", ".join(map(str, self.sizes_of_layers)) + ")"
        return f"{network_shape}, {self.activation_functions[0]}"

class NeuralNetwork(MachineLearningModel):
    """
    A NeuralNetwork.

    Attributes
    ----------
    layers: list[Layer]
        The list of Layer that make the NeuralNetwork.
        Most NeuralNetwork methods work by invoking the appropriate Layer-level methods in the appropriate order.
    input_layer: Layer
        The first Layer of the NN
    hidden_layers: list[Layer]
        The list of non-first-nor-last Layers.
    output_layer: Layer
        The last Layer of the NN
    layers_with_weights: list[Layer]
        The list of all Layers, except the InputLayer.
    """
    def __init__(self, layers: list[Layer]):
        self.layers: list[Layer] = layers
        # Maybe here I should ensure that layers are correctly typed (layers[0] should be an InputLayer, layers[-1] an OutputLayer, all other layers should be HiddenLayer).
        self.input_layer: InputLayer = layers[0]; self.hidden_layers: list[HiddenLayer] = layers[1: -1]; self.output_layer: OutputLayer = layers[-1]
        self.layers_with_weights: list[Layer] = self.layers[1: ]
        self.connect_layers()
        self.initialize_weights()
        self.preprocessing_method: DataPreprocessingMethod = NoPreprocessing(None)

    @classmethod
    def FromArchitecture(cls, architecture: NeuralNetworkArchitecture):
        sizes: list[int] = architecture.sizes_of_layers; act_funs = architecture.activation_functions; init_strat = architecture.initialization_strategy
        il: InputLayer = InputLayer(sizes[0])
        hls: list[HiddenLayer] = [HiddenLayer(n, init_strat, act_fun) for n, act_fun in zip(sizes[1: -1], act_funs[: -1])]
        ol: OutputLayer = OutputLayer(sizes[-1], init_strat, act_funs[-1])
        layers: list[Layer] = [il] + hls + [ol]
        return cls(layers)

    def connect_layers(self) -> None:
        for (i, layer) in enumerate(self.layers):
            if not isinstance(layer, InputLayer): layer.previous_layer = self.layers[i - 1]
            if not isinstance(layer, OutputLayer): layer.next_layer = self.layers[i + 1]

    def initialize_weights(self) -> None:
        for layer in self.layers_with_weights: layer.initialize_weights()
    
    def feed_input(self, value: np.ndarray) -> None:
        self.input_layer.feed_input(value)

    def activate_network(self) -> np.ndarray:
        for i in range(len(self.layers)): self.layers[i].compute_output()
        return self.output_layer.output
    
    def compute_output(self, value: np.ndarray) -> np.ndarray:
        transformed_value = self.preprocessing_method.transform_single_datum(value)
        self.feed_input(transformed_value)
        return self.activate_network()
    
    def backward(self) -> None:
        for l in reversed(self.hidden_layers):
            l.backward()
    
    def compute_multiple_outputs(self, x_data: pd.DataFrame | np.ndarray) -> np.ndarray[np.ndarray]:
        if isinstance(x_data, pd.DataFrame): x_data = x_data.to_numpy()
        outputs = np.array(
            [
                self.compute_output(x_data[i]) for i in range(len(x_data))
            ]
        )
        return outputs
    
    def set_data_preprocessing_method(self, method: DataPreprocessingMethod):
        self.preprocessing_method: DataPreprocessingMethod = method or NoPreprocessing(None)