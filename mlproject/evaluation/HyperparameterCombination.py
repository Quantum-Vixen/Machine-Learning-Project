import inspect
import itertools
from mlproject.training.TrainingAlgorithm import TrainingAlgorithm
from mlproject.models.MachineLearningModel import NeuralNetwork, NeuralNetworkArchitecture
from mlproject.training.ErrorFunction import ErrorFunction
from mlproject.models.Dataset import Dataset
from mlproject.utils.Keys import Keys

def check_tr_hyperparams(training_hyperparameters: dict, algorithm_class: type[TrainingAlgorithm]):
    signature = inspect.signature(algorithm_class.__init__)
    expected_hyparams = signature.parameters
    for hyparam_name in training_hyperparameters.keys():
        if hyparam_name not in expected_hyparams:
            raise ValueError(f"Unexpected parameter: {hyparam_name}")



def compute_network_error(network: NeuralNetwork, dataset: Dataset, error_function: ErrorFunction) -> float:
    y_prediction = network.compute_multiple_outputs(dataset.x)
    return error_function(dataset.y, y_prediction)



class HyperparameterCombination:
    def __init__(self,
                 architecture: NeuralNetworkArchitecture,
                 algorithm_class: type[TrainingAlgorithm],
                 training_hyperparameters: dict[str, object],
                 ensure_hyperparameter_compatibility: bool = True):
        self.architecture: NeuralNetworkArchitecture = architecture
        self.alg_cls: type[TrainingAlgorithm] = algorithm_class
        self.tr_hyp: dict[str, object] = training_hyperparameters

        if ensure_hyperparameter_compatibility:
            check_tr_hyperparams(self.tr_hyp, algorithm_class)
    
    def keys(self) -> list[str]:
        return [Keys.ARCHITECTURE, Keys.TR_ALGORITHM] + list(self.tr_hyp.keys())
    
    def training_hyperparameters_keys(self) -> list[str]:
        return self.tr_hyp.keys()
    
    def as_dict(self) -> dict[str, object]:
        return {Keys.ARCHITECTURE: str(self.architecture),
                Keys.TR_ALGORITHM: str(self.alg_cls.__name__),
                **{k: str(v) for k, v in self.tr_hyp.items()}
                }


class HyperparameterGrid:
    def __init__(self,
                 list_of_architectures: list[NeuralNetworkArchitecture],
                 algorithm_class: type[TrainingAlgorithm],
                 lists_of_training_hyperparameters: dict[str, list[object]]
                 ):
        self.architectures: list[NeuralNetworkArchitecture] = list_of_architectures
        self.alg_cls: type[TrainingAlgorithm] = algorithm_class
        self.tr_hyp_lists: dict[str, list[object]] = lists_of_training_hyperparameters

        self._list: list[HyperparameterCombination] = None  # Store the result of the to_list() method the first time it's called

        check_tr_hyperparams(self.tr_hyp_lists, algorithm_class)
    
    def to_list(self) -> list[HyperparameterCombination]:
        if self._list is None:
            self._list = []
            for arc in self.architectures:
                for tr_hyp_values in itertools.product(*self.tr_hyp_lists.values()):
                    tr_hyp: dict[str, object] = dict(zip(self.tr_hyp_lists.keys(), tr_hyp_values))
                    self._list.append(
                        HyperparameterCombination(arc, self.alg_cls, tr_hyp)
                    )
        return self._list
