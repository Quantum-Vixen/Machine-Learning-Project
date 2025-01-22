"""
This module contains Id, Sigmoid, Tanh, ReLU, and LeakyReLU
"""


import numpy as np

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array




class ActivationFunction:
    """Parent class for activation functions of neural nodes."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The '__call__' method must be implemented in child classes")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The 'derivative' method must be implemented in child classes")

class Identity(ActivationFunction):
    """Identity activation function. f(x) = x"""
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)
    
    def __str__(self):
        return "Identity act. fun."

class Sigmoid(ActivationFunction):
    """Sigmoid activation function. f(x) = 1 / (1 + np.exp(-x))"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = self(x)  # Reuse the __call__ method to compute sigmoid
        return sigmoid * (1 - sigmoid)
    
    def __str__(self):
        return "Sigmoid act. fun."

class Tanh(ActivationFunction):
    """Hyperbolic Tangent activation function. f(x) = np.tanh(x)"""
    def __call__(self, x: np.ndarray):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - self(x)**2
    
    def __str__(self):
        return "Tanh act. fun."

class ReLU(ActivationFunction):
    """Rectified Linear Unit activation function. f(x) = x if x > 0, else 0"""
    def __call__(self, x):
        return np.where(x > 0, x, 0)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def __str__(self):
        return "ReLU act. fun."

class LeakyReLU(ActivationFunction):
    def __init__(self, leak: float):
        self.leak: float = leak
    
    def __call__(self, x):
        return np.where(x > 0, x, self.leak * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.leak)
    
    def __str__(self):
        return f"LeakyReLU({self.leak})"