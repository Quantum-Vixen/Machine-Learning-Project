"""
This module contains strategies to initialize a NeuralNetwork's weights and biases.
Those strategeies are actually performed Layer-wise, however in most use cases each Layer of a NN will use the same InitStrat.
"""


import numpy as np

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array



class LayerInitializationStrategy:
    """Parent class for initialization strategies of weights and biases in Layer."""
    def run(self, size_of_previous_layer: int, size_of_current_layer: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("The 'run' method is implemented only in child classes")

class RandomUniform(LayerInitializationStrategy):
    """
    Initialization strategy sampling weights and biases uniformly in a given interval.

    Attributes
    ----------
    scale: float
        The half-lenght of the interval [center-scale, center+scale] from which values are sampled.
    center: float
        The center of the aforementioned interval.

    Methods
    -------
    run(self, size_of_previous_layer: int, size_of_current_layer: int) -> tuple[np.ndarray, np.ndarray]
        Returns the tuple (random_weights, random_biases), where random_weights and random_biases are np.ndarrays of the appropriate shape.
    """
    def __init__(self, scale_of_interval: float, center_of_interval: float = 0):
        self.scale: float = scale_of_interval
        self.center: float = center_of_interval
        
    
    def run(self, size_of_previous_layer: int, size_of_current_layer: int) -> tuple[np.ndarray, np.ndarray]:
        random_weights: np.ndarray = np.random.uniform(
            -self.scale + self.center, self.scale + self.center,
            (size_of_previous_layer, size_of_current_layer)
            )
        random_biases: np.ndarray = np.random.uniform(
            -self.scale + self.center, self.scale + self.center,
            size_of_current_layer
            )
        return random_weights, random_biases
    
    def __str__(self):
        return f"Initialization: RandomUniform in [{self.center - self.scale}, {self.center + self.scale}]"




class Xavier(LayerInitializationStrategy):
    def run(self, size_of_previous_layer, size_of_current_layer):
        fan_avg: float = (size_of_previous_layer + size_of_current_layer) / 2
        std: float = np.sqrt(1 / fan_avg)
        random_weights: Matrix = np.random.normal(
            0,
            std,
            (size_of_previous_layer, size_of_current_layer)
        )
        biases: Vector = np.zeros(size_of_current_layer)
        return random_weights, biases
    
    def __str__(self):
        return f"Initialization: Normal distribution with variance = 2 / (fan_in + fan_out) for each layer."

class He(LayerInitializationStrategy):
    def run(self, size_of_previous_layer, size_of_current_layer):
        fan_in: int = size_of_previous_layer
        std: float = np.sqrt(2 / fan_in)
        random_weights: Matrix = np.random.normal(
            0,
            std,
            (size_of_previous_layer, size_of_current_layer)
        )
        biases: Vector = np.zeros(size_of_current_layer)
        return random_weights, biases