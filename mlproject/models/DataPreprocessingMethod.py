"""
This module contains a preprocessing function that has to be attached to a TrainingAlgorithm to extract statistics on the input features of the training data, and later to the trained NeuralNetwork to preprocess future data using the same preprocessing function
extracted from the training data. This function is incomplete and not general enough to be used in a professional setting, and should be ignored until it is developed further.
"""

import numpy as np

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array

class DataPreprocessingMethod:
    def __init__(self, data: Matrix):
        pass

    def transform_single_datum(self, datum: Vector) -> Vector:
        return datum
    
    def transform_multiple_data(self, data: Matrix) -> Matrix:
        return data
    
    def inverse_transform_single_datum(self, datum: Vector) -> Vector:
        return datum
    
    def inverse_transform_multiple_data(self, data: Matrix) -> Vector:
        return data

class NoPreprocessing(DataPreprocessingMethod):
    pass

class Standardization(DataPreprocessingMethod):
    def __init__(self, data: Matrix):
        self.means: Vector = data.mean(axis= 0)
        self.stds: Vector = data.std(axis= 0)
    
    def transform_single_datum(self, datum: Vector) -> Vector:
        return (datum - self.means) / self.stds
    
    def transform_multiple_data(self, data: Matrix) -> Matrix:
        return np.array(
            [
                self.transform_single_datum(datum) for datum in data
            ]
        )
    
    def inverse_transform_single_datum(self, datum: Vector) -> Vector:
        return datum * self.stds + self.means
    
    def inverse_transform_multiple_data(self, data: Matrix) -> Vector:
        return np.array(
            [
                self.inverse_transform_single_datum(datum) for datum in data
            ]
        )