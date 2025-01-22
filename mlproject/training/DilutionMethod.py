import numpy as np
from typing import TypeAlias, Iterable
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array

from mlproject.utils.ListOfArrays import ListOfMatrices, ListOfVectors
from mlproject.models.MachineLearningModel import NeuralNetwork
from mlproject.models.Layer import OutputLayer


class DilutionMethod:
    def __init__(self, drop_probabilities_of_layers: list[float]
                 #input_drop_probability: float, hidden_drop_probability: float
                 ):
        #self.in_drop: float = input_drop_probability
        #self.hid_drop: float = hidden_drop_probability
        self.drop_probs: list[float] = drop_probabilities_of_layers

        self.network: NeuralNetwork = None

        self.weights_storages: ListOfMatrices = None
        self.masks: ListOfVectors = None  # Mask values: 1 if the unit is active, 0 if the unit has been dropped out.

        self.weights_masks: ListOfMatrices = None
        
    
    @property
    def biases_masks(self) -> ListOfVectors:
        return self.masks[1:]
    
    @property
    def n_layers(self) -> int:
        return len(self.network.layers)
    
    @property
    def layer_sizes(self) -> list[int]:
        return [l.unit_number for l in self.network.layers]
    
    @property
    def n_layers_with_weights(self) -> int:
        return self.n_layers - 1


    def set_network(self, network: NeuralNetwork) -> None:
        self.network = network
        self.weights_storages = ListOfMatrices([np.empty_like(l.weights) for l in self.network.layers_with_weights])
        
        self.masks = ListOfVectors([np.ones(l.unit_number) for l in self.network.layers])

    def generate_new_mask(self) -> None:
        for i, l in enumerate(self.network.layers):
            #drop_prob: float = self.in_drop if isinstance(l, InputLayer) else self.hid_drop if isinstance(l, HiddenLayer) else 0
            drop_prob = self.drop_probs[i] if not isinstance(l, OutputLayer) else 0
            self.masks[i] = (np.random.random(l.unit_number) > drop_prob).astype(int)
        self.compute_weights_mask()

    def compute_weights_mask(self) -> None:
        list_of_mask_matrices: list[Matrix] = []
        for i in range(self.n_layers_with_weights):
            previous_layer_mask = self.masks[i]
            current_layer_mask = self.masks[i+1]
            list_of_mask_matrices.append(np.outer(previous_layer_mask, current_layer_mask))
        self.weights_masks = ListOfMatrices(list_of_mask_matrices)    
    
    def apply_mask(self) -> None:
        for i, l in enumerate(self.network.layers_with_weights):
            self.weights_storages[i] = l.weights.copy()
            #normalizing_factor = 1 - self.in_drop if i == 0 else 1 - self.hid_drop
            normalizing_factor = 1 - self.drop_probs[i]
            l.weights = l.weights * self.weights_masks[i] / normalizing_factor
            
    
    def drop_some_units(self) -> None:
        self.generate_new_mask()
        self.apply_mask()

    def restore_dropped_units(self) -> None:
        for i, l in enumerate(self.network.layers_with_weights):
            l.weights = self.weights_storages[i].copy()

class Dropout(DilutionMethod):
    pass

class MinibatchDropout(DilutionMethod):
    pass

class StaticDropout(DilutionMethod):
    pass
            

class NoDilution(DilutionMethod):
    def __init__(self):
        pass

    def set_network(self, network: NeuralNetwork):
        pass

    def restore_dropped_units(self) -> None:
        pass
    
    def drop_some_units(self) -> None:
        pass