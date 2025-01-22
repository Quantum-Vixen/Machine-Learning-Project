import numpy as np
import pandas as pd
from typing import TypeAlias, Iterable
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array
from joblib import Parallel, delayed



from mlproject.training.TrainingAlgorithm import TrainingAlgorithm
from mlproject.models.MachineLearningModel import NeuralNetwork

from mlproject.training.ErrorFunction import ErrorFunction
from mlproject.models.Dataset import Dataset
from mlproject.utils.Keys import Keys
from HyperparameterCombination import HyperparameterCombination, compute_network_error




def parenthetical_uncertainty_format(number: float, uncertainty: float, uncertainty_digits: int = 2) -> str:
    import math
    num_exponent = math.floor(math.log10(abs(number))) if number != 0 else 0
    unc_exponent = math.floor(math.log10(abs(uncertainty))) if uncertainty != 0 else 0

    n = num_exponent - unc_exponent + 1  # Number of significant digits
    if n <= 0: return f"{number} +- {uncertainty}"
    scaled_number = number * 10 ** -num_exponent
    num_string = f"{scaled_number:.{n - 1 + uncertainty_digits - 1}f}"

    scaled_uncertainty = uncertainty * 10 ** -unc_exponent
    unc_string = f"{scaled_uncertainty:.{uncertainty_digits - 1}f}".replace('.', '')

    return f"{num_string}({unc_string})e{num_exponent}"






class ModelEvaluationMethod:
    pass

class SelectionMethod(ModelEvaluationMethod):
    def __init__(self,
                 hyperparameter_combinations: list[HyperparameterCombination],
                 risk_function: ErrorFunction):
        self.hyp_combs: list[HyperparameterCombination] = hyperparameter_combinations
        self.risk_fun: ErrorFunction = risk_function
        self.dataset: Dataset = None  # Has to be loaded after creating the SelectionMethod. Different SelectionMethod subclasses will have different methods to load it.
        self.results: dict[HyperparameterCombination, float] = {hyp_comb: None for hyp_comb in self.hyp_combs}  # Validation error for each combination of hyperparams will be reported in this dict.
    
    def load_dataset(self, dataset: Dataset, shuffle_data: bool = True) -> None:
        self.dataset = dataset
        if shuffle_data: self.dataset.shuffle()
    
    def shuffle_dataset(self) -> None:
        pass

    def report_results_as_dataframe(self) -> pd.DataFrame:
        out: list[dict] = []  # List of rows of the DataFrame that will be returned by this function.
        for hyp_comb, vl_err in self.results.items():  # A row of the output DataFrame shows the hyperparameters of the hyp_comb and the validation error associated to it.
            hyp_comb_dict = hyp_comb.as_dict()
            hyp_comb_dict.update({Keys.VL_ERROR: vl_err})
            out.append(hyp_comb_dict)
        return pd.DataFrame(out)

    def run(self, max_training_epochs: int) -> None:
        raise NotImplementedError("This method has to be implemented in child classes")
    
    @property
    def best_hyperparameter_combination(self) -> HyperparameterCombination:
        """Returns the hyperparameter combination that minimizes the validation error."""
        return min(self.results, key = self.results.get)
    
    @property
    def best_result(self) -> float:
        """Returns the validation error of the best hyperparameter combination."""
        return self.results[self.best_hyperparameter_combination]

class HoldOutSelection(SelectionMethod):
    def __init__(self, hyperparameter_combinations, risk_function, fraction_of_data_to_be_used_as_validation: float):
        super().__init__(hyperparameter_combinations, risk_function)
        self.val_frac: float = fraction_of_data_to_be_used_as_validation
        self.tr_set: Dataset = None
        self.vl_set: Dataset = None
        self.training_errors: dict[HyperparameterCombination, float] = {hyp_comb: None for hyp_comb in self.hyp_combs}
    
    def load_dataset(self, dataset, shuffle_data: bool = True):
        super().load_dataset(dataset, shuffle_data)
        self.vl_set, self.tr_set = self.dataset.split(self.val_frac, shuffle = False)  # No need to shuffle the data during the split, since it's already been shuffled by super().load_dataset
    
    def load_already_split_datasets(self, training_set: Dataset, validation_set: Dataset) -> None:
        self.dataset = Dataset.concatenate([validation_set, training_set])
        self.tr_set: Dataset = training_set
        self.vl_set: Dataset = validation_set
        if len(self.vl_set) / len(self.dataset) != self.val_frac:
            print(f"Warning: the fraction of data used as vl is {len(self.vl_set) / len(self.dataset)}. It should be {self.val_frac}")
    
    def shuffle_dataset(self) -> None:
        super().shuffle_dataset()
        frac = len(self.vl_set) / len(self.dataset)
        self.vl_set, self.tr_set = self.dataset.split(frac, shuffle = True)
    
    def construct_training_algorithm_for_single_hyp_comb(self, hyp_comb: HyperparameterCombination) -> TrainingAlgorithm:
        arc, alg_cls, tr_hyp = hyp_comb.architecture, hyp_comb.alg_cls, hyp_comb.tr_hyp
        network = NeuralNetwork.FromArchitecture(arc)
        alg: TrainingAlgorithm = alg_cls(self.tr_set.x, self.tr_set.y, network, **tr_hyp)
        return alg
    
    def process_single_hyp_comb(self, hyp_comb: HyperparameterCombination, max_epochs: int) -> float:
        alg = self.construct_training_algorithm_for_single_hyp_comb(hyp_comb)
        alg.run(max_epochs)
        tr_err = alg.history[Keys.TR_ERROR][-1]
        vl_err: float = compute_network_error(alg.network, self.vl_set, self.risk_fun)
        return tr_err, vl_err
    
    def report_error_curves_for_single_hyp_comb(self, hyp_comb: HyperparameterCombination, max_epochs: int) -> tuple[list[float], list[float]]:
        """Returns training_curve, validation_curve."""
        alg = self.construct_training_algorithm_for_single_hyp_comb(hyp_comb)
        tr_curve, vl_curve = [], []
        for epoch in range(max_epochs):
            alg.run(1)
            vl_curve.append(compute_network_error(alg.network, self.vl_set, self.risk_fun))
        tr_curve = alg.history[Keys.TR_ERROR]
        return tr_curve, vl_curve
    
    def run(self, max_training_epochs) -> None:
        single_hyp_comb_results: list[tuple[float, float]] = Parallel(n_jobs=-1)(
            delayed(self.process_single_hyp_comb)(hyp_comb, max_training_epochs) for hyp_comb in self.hyp_combs
        )
        for hyp_comb, (tr_err, result) in zip(self.hyp_combs, single_hyp_comb_results):
            self.training_errors[hyp_comb] = tr_err
            self.results[hyp_comb] = result
    
    def report_error_curves(self, max_training_epochs: int) -> tuple[dict, dict]:
        """Returns training_curves and validation_curves, dicts of the form {hyp_comb: list_of_errors_at_each_epoch.}"""
        single_hyp_comb_results: list[tuple] = Parallel(n_jobs=-1)(
            delayed(self.report_error_curves_for_single_hyp_comb)(hyp_comb, max_training_epochs) for hyp_comb in self.hyp_combs
        )
        tr_curves, vl_curves = {}, {}
        single_hyp_comb_tr_curves, single_hyp_comb_vl_curves = zip(*single_hyp_comb_results)
        for hyp_comb, tr_curve, vl_curve in zip(self.hyp_combs, single_hyp_comb_tr_curves, single_hyp_comb_vl_curves):
            tr_curves[hyp_comb] = tr_curve
            vl_curves[hyp_comb] = vl_curve
        return tr_curves, vl_curves
    
    def report_results_as_dataframe(self, include_tr_errs: bool = False) -> pd.DataFrame:
        out: list[dict] = []  # List of rows of the DataFrame that will be returned by this function.
        for hyp_comb, vl_err in self.results.items():  # A row of the output DataFrame shows the hyperparameters of the hyp_comb and the validation error associated to it.
            hyp_comb_dict = hyp_comb.as_dict()
            if include_tr_errs:
                hyp_comb_dict.update({Keys.TR_ERROR: self.training_errors[hyp_comb]})
            hyp_comb_dict.update({Keys.VL_ERROR: vl_err})
            out.append(hyp_comb_dict)
        return pd.DataFrame(out)

class KFoldCrossValidationSelection(SelectionMethod):
    def __init__(self, hyperparameter_combinations, risk_function, number_of_folds: int):
        super().__init__(hyperparameter_combinations, risk_function)
        self.n_folds: int = number_of_folds
        self.fold_sets: list[Dataset] = None
        self.results_for_each_single_fold: dict[HyperparameterCombination, Vector] = {hyp_comb: np.empty(self.n_folds) for hyp_comb in self.hyp_combs}
    
    @property
    def fold_length(self) -> int:
        return len(self.dataset) // self.n_folds
    
    @property
    def std_of_results_over_folds(self) -> dict[HyperparameterCombination, float]:
        return {hyp_comb: vector.std(ddof = 1) for hyp_comb, vector in self.results_for_each_single_fold.items()}
    
    def load_dataset(self, dataset: Dataset, shuffle_data: bool = True) -> None:
        super().load_dataset(dataset, shuffle_data)
        effective_end = (len(self.dataset) // self.fold_length) * self.fold_length
        starts = range(0, effective_end, self.fold_length)
        self.fold_sets: list[Dataset] = [self.dataset[start:start + self.fold_length] for start in starts]
        # self.results_for_each_single_fold = {hyp_comb: np.empty(self.n_folds) for hyp_comb in self.hyp_combs}

    def prepare_hold_out_method_for_a_given_fold(self, fold_set: Dataset) -> HoldOutSelection:
        vl_set: Dataset = fold_set
        tr_set: Dataset = Dataset.concatenate([other_fold_set for other_fold_set in self.fold_sets if other_fold_set != fold_set])
        holdout = HoldOutSelection(self.hyp_combs, self.risk_fun, len(vl_set) / (len(tr_set) + len(vl_set)))
        holdout.load_already_split_datasets(tr_set, vl_set)
        return holdout
    
    def run(self, max_training_epochs: int) -> None:
        for k, fold_set in enumerate(self.fold_sets):
            holdout = self.prepare_hold_out_method_for_a_given_fold(fold_set)
            holdout.run(max_training_epochs)
            for hyp_comb in self.hyp_combs:
                self.results_for_each_single_fold[hyp_comb][k] = holdout.results[hyp_comb]
        for hyp_comb in self.hyp_combs:
            self.results[hyp_comb] = self.results_for_each_single_fold[hyp_comb].mean()

class AssessmentMethod(ModelEvaluationMethod):
    def __init__(self, selection_method: SelectionMethod):
        self.selection_method = selection_method
        self.dataset: Dataset = None
        self.result: float = None  # The average risk of a MachineLearningModel trained and selected according to the selection_method.

    def pass_development_dataset_to_selection_method(self, development_dataset: Dataset, shuffle_data = True) -> None:
        self.selection_method.load_dataset(development_dataset, shuffle_data)

    def load_dataset(self, dataset: Dataset, shuffle_data: bool = True) -> None:
        self.dataset = dataset
        if shuffle_data: self.dataset.shuffle()
    
    def run(self, max_training_epochs: int) -> None:
        raise NotImplementedError("This method has to be implemented in child classes")

class HoldOutAssessment(AssessmentMethod):
    def __init__(self, selection_method, fraction_of_data_to_be_used_as_test: float):
        super().__init__(selection_method)
        self.test_frac: float = fraction_of_data_to_be_used_as_test
        self.test_set: Dataset = None
        self.dev_set: Dataset = None
    
    def load_dataset(self, dataset, shuffle_data = True):
        super().load_dataset(dataset, shuffle_data)
        self.test_set, self.dev_set = self.dataset.split(self.test_frac, shuffle = False)
        self.pass_development_dataset_to_selection_method(self.dev_set, shuffle_data)
    
    def load_already_split_datasets(self, development_set: Dataset, test_set: Dataset) -> None:
        self.dataset = Dataset.concatenate([test_set, development_set])
        self.dev_set = development_set
        self.test_set = test_set
        if len(self.test_set) / len(self.dataset) != self.test_frac:
            print(f"Warning: the fraction of data used as vl is {len(self.test_set) / len(self.dataset)}. It should be {self.test_frac}")
        self.pass_development_dataset_to_selection_method(self.dev_set)
    
    def final_training_algorithm_from_best_hyp_comb(self, hyp_comb: HyperparameterCombination) -> TrainingAlgorithm:
        arc, alg_cls, tr_hyp = hyp_comb.architecture, hyp_comb.alg_cls, hyp_comb.tr_hyp
        network = NeuralNetwork.FromArchitecture(arc)
        alg: TrainingAlgorithm = alg_cls(self.dev_set.x, self.dev_set.y, network, **tr_hyp)
        return alg

    def run(self, max_training_epochs):
        self.selection_method.run(max_training_epochs)
        best_hyp_comb: HyperparameterCombination = self.selection_method.best_hyperparameter_combination
        final_retraining: TrainingAlgorithm = self.final_training_algorithm_from_best_hyp_comb(best_hyp_comb)
        final_retraining.run(max_training_epochs)
        self.result = compute_network_error(final_retraining.network, self.test_set, self.selection_method.risk_fun)

class KFoldCrossValidationAssessment(AssessmentMethod):
    def __init__(self, selection_method, number_of_folds: int):
        super().__init__(selection_method)
        self.n_folds: int = number_of_folds
        self.fold_sets: list[Dataset] = None
        self.result_for_each_single_fold: Vector = np.empty(self.n_folds)
    
    @property
    def fold_length(self) -> int:
        return len(self.dataset) // self.n_folds
    
    @property
    def std_of_results_over_folds(self) -> float:
        return np.std(self.result_for_each_single_fold, ddof = 1)
    
    def load_dataset(self, dataset, shuffle_data = True):
        super().load_dataset(dataset, shuffle_data)
        effective_end = (len(self.dataset) // self.fold_length) * self.fold_length
        starts = range(0, effective_end, self.fold_length)
        self.fold_sets: list[Dataset] = [self.dataset[start:start + self.fold_length] for start in starts]
        # self.results_for_each_single_fold = {hyp_comb: np.empty(self.n_folds) for hyp_comb in self.hyp_combs}
    
    def prepare_holdout_method_for_a_given_fold(self, fold_set: Dataset) -> HoldOutAssessment:
        test_set: Dataset = fold_set
        dev_set: Dataset = Dataset.concatenate([other_fold_set for other_fold_set in self.fold_sets if other_fold_set != fold_set])
        holdout: HoldOutAssessment = HoldOutAssessment(self.selection_method, len(test_set) / (len(dev_set) + len(test_set)))
        holdout.load_already_split_datasets(dev_set, test_set)
        return holdout
    
    def run(self, max_training_epochs) -> None:
        for k, fold_set in enumerate(self.fold_sets):
            holdout = self.prepare_holdout_method_for_a_given_fold(fold_set)
            holdout.run(max_training_epochs)
            self.result_for_each_single_fold[k] = holdout.result
        self.result = self.result_for_each_single_fold.mean()