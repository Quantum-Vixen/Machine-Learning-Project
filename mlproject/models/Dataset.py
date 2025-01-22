from __future__ import annotations
import numpy as np
import pandas as pd

from typing import TypeAlias, Iterable
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array



class Dataset:
    def __init__(self, x_data: Matrix, y_data: Matrix):
        """
        Initializes the dataset with input (x_data) and output (y_data).
        """

        if isinstance(x_data, pd.DataFrame): x_data = x_data.to_numpy()
        if isinstance(y_data, pd.DataFrame): y_data = y_data.to_numpy()
        
        if x_data.ndim == 1 or y_data.ndim == 1:
            raise ValueError(f"x_data and y_data should be matrices, where each row represents a pattern and each column a feature, but got arguments of shape {x_data.shape} and {y_data.shape}")

        self.x: Matrix = x_data
        self.y: Matrix = y_data
    
    def __len__(self):
        """
        Returns the number of patterns in the dataset.
        """
        return len(self.x)
    
    def __getitem__(self, index):
        """
        Retrieves the input-output pair at the specified index.
        """
        if isinstance(index, slice):
            return Dataset(self.x[index], self.y[index])
        # This should be refactored in the future. The behaviour should be the same as numpy, regardless of index type.
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return Dataset(self.x[index], self.y[index])
        return self.x[index], self.y[index]
    
    def shuffle(self) -> None:
        indices: np.ndarray = np.arange(len(self))
        np.random.shuffle(indices)
        self.x = self.x[indices, :]
        self.y = self.y[indices, :]
    
    def split(self, fraction: float, shuffle: bool = True) -> tuple[Dataset, Dataset]:
        """
        Returns two datasets, one with fraction*N data and the other with (1-fraction)*N.
        """
        indices: np.ndarray = np.arange(len(self))
        
        if shuffle:
            np.random.shuffle(indices)
        splitting_number: int = int(fraction * len(self))
        indices_1: np.ndarray = indices[:splitting_number]
        
        indices_2: np.ndarray = indices[splitting_number:]
        ds_1: Dataset = Dataset(self.x[indices_1, :], self.y[indices_1, :])
        ds_2: Dataset = Dataset(self.x[indices_2, :], self.y[indices_2, :])
        return ds_1, ds_2
    
    @classmethod
    def concatenate(cls, datasets: Iterable[Dataset]) -> Dataset:
        x = np.concatenate([ds.x for ds in datasets])
        y = np.concatenate([ds.y for ds in datasets])
        return Dataset(x, y)


class DataManager:
    """An auxiliary class for extracting minibatches from a Dataset."""
    def __init__(self, dataset: Dataset, minibatch_size: int = None, shuffle: bool = True):
        self.dataset: Dataset = dataset
        self.minibatch_size: int = minibatch_size or len(dataset)
        self.shuffle: bool = shuffle
    
    def __iter__(self):
        """
        An iterator yielding minibatches.
        """
        indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(indices), self.minibatch_size):
            minibatch_indices = indices[start:start + self.minibatch_size]
            minibatch_x = self.dataset.x[minibatch_indices, :]
            minibatch_y = self.dataset.y[minibatch_indices, :]
            yield minibatch_x, minibatch_y