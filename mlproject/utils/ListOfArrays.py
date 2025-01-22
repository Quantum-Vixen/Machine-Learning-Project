"""
An utility class for defining element-wise operations on lists containing heteromorphic np.ndarrays.
Useful for conveniently manipulating network weights and biases in MLP-architecture NeuralNetworks.
"""


import numpy as np

from typing import TypeAlias
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array


class ListOfArrays:
    """
    An utility class for defining element-wise operations on lists containing heteromorphic np.ndarrays.
    Useful for conveniently manipulating network weights and biases in MLP-architecture NeuralNetworks.
    """
    def __init__(self, arrays: list[np.ndarray]):
        self.arrays: list[np.ndarray] = arrays
    
    def __repr__(self):
        return f"ListOfArrays{(self.arrays)}"

    def __getitem__(self, index):
        return self.arrays[index]

    def __setitem__(self, index, value):
        self.arrays[index] = value
    
    def __add__(self, other):
        if isinstance(other, ListOfArrays):
            return ListOfArrays([x + y for x, y in zip(self.arrays, other.arrays)])
        elif np.isscalar(other):
            return ListOfArrays([x + other for x in self.arrays])
        else:
            raise TypeError("Operand must be a ListOfArrays or a scalar.")

    def __radd__(self, other):
        return self.__add__(other)        
    
    def __mul__(self, other):
        if isinstance(other, ListOfArrays):
            return ListOfArrays([x * y for x, y in zip(self.arrays, other.arrays)])
        elif np.isscalar(other):
            return ListOfArrays([x * other for x in self.arrays])
        else:
            raise TypeError("Operand must be a ListOfArrays or a scalar.")
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ListOfArrays):
            return ListOfArrays([x / y for x, y in zip(self.arrays, other.arrays)])
        elif np.isscalar(other):
            return ListOfArrays([x / other for x in self.arrays])
        else:
            raise TypeError("Operand must be a ListOfArrays or a scalar.")
    
    def __pow__(self, power: float):
        return ListOfArrays([x**power for x in self.arrays])
    
    def sum(self) -> float:
        return np.sum([np.sum(array) for array in self.arrays])
    
    def set_all_values_to(self, value: float) -> None:
        for a in self.arrays:
            a[:] = value
    
    def copy(self):
        return ListOfArrays([x.copy() for x in self.arrays])


class ListOfVectors(ListOfArrays):
    """
    An utility class for convenient manipulation of lists of Vectors of different lengths.
    Useful for network biases in MLP-architecture NeuralNetworks.
    """
    def __init__(self, arrays: list[Vector]):
        super().__init__(arrays)

class ListOfMatrices(ListOfArrays):
    """
    An utility class for convenient manipulation of lists of Matrices of different number of rows and cols.
    Useful for network weights in MLP-architecture NeuralNetworks.
    """
    pass
    def __init__(self, arrays: list[Matrix]):
        super().__init__(arrays)