import numpy as np

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array


class ErrorFunction:
    def __call__(self, y_data: np.ndarray, y_predicted: np.ndarray) -> float:
        pass
    
    def simple_gradient(self, y_data: np.ndarray, y_predicted: np.ndarray) -> np.ndarray:
        pass

class MSE(ErrorFunction):
    def __call__(self, y_data: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        Returns the average over the dataset of the square euclidean distance between the training outputs and the predictions.
        """
        #num_patterns = 1 if y_data.ndim == 1 else len(y_data)
        num_patterns = len(y_data)
        return 0.5 * np.sum((y_data - y_predicted)**2) / num_patterns
    
    def simple_gradient(self, y_data: np.ndarray, y_predicted: np.ndarray) -> np.ndarray:
        """
        Returns y_data - y_predicted. It is meant to be used on a single pattern at a time, during backpropagation.
        """
        return (y_data - y_predicted)
    
    def __str__(self):
        return "MSE"

class MEE(ErrorFunction):
    def __init__(self, epsilon: float = 0):
        self.eps: float = epsilon

    def __call__(self, y_data: Matrix, y_predicted: Matrix) -> float:
        """
        Returns the average over the dataset of the l-1 distance between the training outputs and the predictions.
        """
        num_patterns = len(y_data)
        return np.sum( np.sqrt(((y_data - y_predicted)**2).sum(axis=1)  +  self.eps) ) / num_patterns
    
    def simple_gradient(self, y_data: Vector, y_predicted: Vector) -> Vector:
        return (y_data - y_predicted) * (np.sum((y_data - y_predicted)**2) + self.eps)** -0.5
    
    def __str__(self):
        return "MEE"

class MSEPlusMEE(ErrorFunction):
    def __init__(self, MSE_term: float = 0.5, MEE_term: float = 0.5, epsilon: float = 0):
        self.mse_term: float = MSE_term
        self.mee_term: float = MEE_term
        self.epsilon: float = epsilon
        self.MSE: MSE = MSE()
        self.MEE: MEE = MEE(self.epsilon)
    
    def __call__(self, y_data, y_predicted):
        return self.mse_term * self.MSE(y_data, y_predicted) + self.mee_term * self.MEE(y_data, y_predicted)
    
    def simple_gradient(self, y_data, y_predicted):
        return self.mse_term * self.MSE.simple_gradient(y_data, y_predicted) + self.mee_term * self.MEE.simple_gradient(y_data, y_predicted)
    
    def __str__(self):
        return f"{self.mse_term}*MSE + {self.mee_term}*MEE"