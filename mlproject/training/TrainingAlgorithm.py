import numpy as np
import pandas as pd
from typing import TypeAlias, Iterable
Vector: TypeAlias = np.ndarray  # A 1-D array
Matrix: TypeAlias = np.ndarray  # A 2-D array



from mlproject.utils.ListOfArrays import ListOfMatrices, ListOfVectors, ListOfArrays
from mlproject.models.MachineLearningModel import NeuralNetwork
from mlproject.models.DataPreprocessingMethod import DataPreprocessingMethod, NoPreprocessing
from mlproject.utils.Keys import Keys
from mlproject.models.Dataset import Dataset, DataManager
from mlproject.training.DilutionMethod import DilutionMethod, NoDilution, Dropout, MinibatchDropout, StaticDropout
from mlproject.training.ErrorFunction import ErrorFunction
from mlproject.training.StoppingCondition import StoppingCondition
from mlproject.training.MomentumRule import MomentumRule
from mlproject.training.RegularizationTerm import RegularizationTerm, NoRegularization
from mlproject.training.MomentumRule import NoMomentum, MomentumRule





class TrainingAlgorithm:
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, network: NeuralNetwork,
                 preprocessing_method_cls: type[DataPreprocessingMethod] = None):
        self.network: NeuralNetwork = network

        if isinstance(x_train, pd.DataFrame): x_train = x_train.to_numpy()
        if isinstance(y_train, pd.DataFrame): y_train = y_train.to_numpy()

        self.training_set = Dataset(x_train, y_train)

        self.current_tr_err: float = float('inf')

        self.history: dict[list] = {Keys.TR_ERROR: []}

        if preprocessing_method_cls is None: preprocessing_method_cls = NoPreprocessing
        self.preprocessing_method: DataPreprocessingMethod = preprocessing_method_cls(x_train)
        self.network.set_data_preprocessing_method(self.preprocessing_method)

class GradientDescentAlgorithm(TrainingAlgorithm):
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, network: NeuralNetwork,
                 learning_rate: float,
                 error_function: ErrorFunction,
                 stopping_condition: StoppingCondition,
                 preprocessing_method_cls: type[DataPreprocessingMethod] = None,
                 regularization_term: RegularizationTerm = None,
                 momentum_rule: MomentumRule = None,
                 dilution_method: DilutionMethod = None,
                 minibatch_size: int = None
                 ):
        # Initialize parameters given in __init__
        super().__init__(x_train, y_train, network, preprocessing_method_cls)
        self.learning_rate: float = learning_rate
        self.err_fun: ErrorFunction = error_function
        self.stop_cond: StoppingCondition = stopping_condition; self.stop_cond.set_alg(self)
        self.regularization_term: RegularizationTerm = regularization_term or NoRegularization()
        self.regularization_term.set_network(self.network)
        self.momentum: MomentumRule = momentum_rule or NoMomentum()
        self.dilution_method: DilutionMethod = dilution_method or NoDilution()
        self.dilution_method.set_network(self.network)
        self.minibatch_size: int = minibatch_size or len(self.training_set)

        # Initialize attributes used for later computations.
        self.weights_gradient: ListOfMatrices = ListOfMatrices([np.zeros_like(l.weights) for l in self.network.layers_with_weights])  # Gradients of the - loss function (= error function + regularization penalty)
        self.biases_gradient: ListOfVectors = ListOfVectors([np.zeros_like(l.biases) for l in self.network.layers_with_weights])      # with respect to, respectively, weights and biases of the NeuralNetwork.

        self.weights_update: ListOfMatrices = ListOfMatrices([np.zeros_like(l.weights) for l in self.network.layers_with_weights])  # Update terms for, respectively, weights and biases of the NeuralNetwork, computed 
        self.biases_update: ListOfVectors = ListOfVectors([np.zeros_like(l.biases) for l in self.network.layers_with_weights])      # according to the particular GradientDescentAlgorithm's update rules. Momentum is not included.

        self.minibatch_generator: DataManager = DataManager(self.training_set, self.minibatch_size,
                                                            shuffle = (self.minibatch_size != len(self.training_set))
                                                            )
        self.current_mb_size: int = None
        self.current_mb_x: Matrix = None; self.current_mb_y: Matrix = None
    

    def run(self, max_epochs: int) -> None:
        epoch: int = 0
        while epoch < max_epochs:
            epoch += 1
            self.do_epoch_training()
            self.compute_training_error()
            self.store_training_error()
            if self.stop_cond.is_satisfied: break
    
    def do_epoch_training(self) -> None:
        for minibatch_x, minibatch_y in self.minibatch_generator:
            self.update_minibatch_metadata(minibatch_x, minibatch_y)
            self.compute_necessary_quantities_for_network_update()
            self.update_network_parameters()

    
    def update_minibatch_metadata(self, minibatch_x: Matrix, minibatch_y: Matrix):
        self.current_mb_x = minibatch_x; self.current_mb_y = minibatch_y
        self.current_mb_size = len(minibatch_x)

    def compute_necessary_quantities_for_network_update(self) -> None:
        raise NotImplementedError("This method should be implemented in child classes of GradientDescentAlgorithm.")

    def compute_gradients(self) -> None:
        # Reset gradients to 0.
        self.reset_gradients()
        # Sum the gradient of error function term, obtained via backpropagation, for each data point in the minibatch.
        if isinstance(self.dilution_method, MinibatchDropout): self.dilution_method.drop_some_units()
        for x, y in zip(self.current_mb_x, self.current_mb_y):
            if isinstance(self.dilution_method, Dropout): self.dilution_method.drop_some_units()
            # Manually compute and set the delta for the output layer.
            predicted_y = self.network.compute_output(x)
            out_l = self.network.output_layer
            out_l.delta = self.err_fun.simple_gradient(y, predicted_y)*out_l.activation_function.derivative(out_l.linear_output)
            # Use backpropagation to set the deltas for all layers.
            self.network.backward()
            # Sum the appropriate quantities to the gradient.
            self.weights_gradient += ListOfMatrices([np.outer(l.previous_layer.output, l.delta) for l in self.network.layers_with_weights])
            self.biases_gradient += ListOfVectors([l.delta for l in self.network.layers_with_weights])
            if isinstance(self.dilution_method, Dropout): self.dilution_method.restore_dropped_units()
            #print("Masks of input layer\n", self.dilution_method.masks[0]); print("Masks of first HL\n", self.dilution_method.masks[1])
            #print("output of first HL\n", self.network.layers_with_weights[0].output)
            #print("deltas of HL \n", self.network.layers_with_weights[0].delta)
            #print("Weights grads of HL\n", self.weights_gradient[0])
            #print('\n\n\n')
        if isinstance(self.dilution_method, MinibatchDropout): self.dilution_method.restore_dropped_units()
        # Divide by the number of examples in the minibatch to get an average.
        self.weights_gradient /= self.current_mb_size; self.biases_gradient /= self.current_mb_size
        # Add regularization term contributions to the gradients.
        contribution_to_w, contribution_to_b = self.regularization_term.gradient()
        self.weights_gradient += contribution_to_w; self.biases_gradient += contribution_to_b

    def reset_gradients(self) -> None:
        # Reset gradients to 0.
        self.weights_gradient.set_all_values_to(0)
        self.biases_gradient.set_all_values_to(0)

    def update_momentum_terms(self) -> None:
        self.momentum.update(self.weights_update, self.biases_update)

    def update_network_parameters(self) -> None:
        for i, l in enumerate(self.network.layers_with_weights):
            l.weights += (self.weights_update + self.momentum.weights_term)[i]
            l.biases += (self.biases_update + self.momentum.biases_term)[i]
    
    def compute_training_error(self) -> None:
        y_prediction = self.network.compute_multiple_outputs(self.training_set.x)
        self.current_tr_err = self.err_fun(self.training_set.y, y_prediction)
    
    def store_training_error(self) -> None:
        self.history[Keys.TR_ERROR] += [self.current_tr_err]
    
    @staticmethod
    def update_moving_average(moving_average: ListOfArrays, forgetting_factor: float, new_value: ListOfArrays):
        """
        Returns f * moving_average + (1 - f) * new_value, where f is the forgetting_factor.
        Used as a helper method in various TrainingAlgorithms
        """
        return forgetting_factor * moving_average + (1 - forgetting_factor) * new_value

class ClassicalBackprop(GradientDescentAlgorithm):
    def compute_necessary_quantities_for_network_update(self) -> None:
        self.compute_gradients()

        factor = self.learning_rate*self.current_mb_size / len(self.training_set)
        self.weights_update = factor * self.weights_gradient
        self.biases_update = factor * self.biases_gradient

        self.momentum.update(self.weights_update, self.biases_update)

class AdaGrad(GradientDescentAlgorithm):
    def __init__(self, x_train, y_train, network, learning_rate, error_function, stopping_condition,
                 preprocessing_method_cls: type[DataPreprocessingMethod] = None,
                 regularization_term = None, momentum_rule = None,
                 dilution_method: DilutionMethod = None,
                 minibatch_size = None,
                 epsilon: float = 1e-10
                 ):
        super().__init__(x_train, y_train, network, learning_rate, error_function, stopping_condition, preprocessing_method_cls, regularization_term, momentum_rule, dilution_method, minibatch_size)
        self.epsilon: float = epsilon  # A small numerical value used in the denominator of the weights and biases updates to prevent division by 0.

        self.sum_w_grad_squares: ListOfMatrices = ListOfMatrices([np.zeros_like(l.weights) for l in self.network.layers_with_weights])
        self.sum_b_grad_squares: ListOfVectors = ListOfVectors([np.zeros_like(l.biases) for l in self.network.layers_with_weights])
    
    def compute_necessary_quantities_for_network_update(self) -> None:
        self.compute_gradients()

        self.accumulate_grad_squares()
        factor = self.learning_rate*self.current_mb_size / len(self.training_set)
        self.weights_update = factor * self.weights_gradient / (self.sum_w_grad_squares + self.epsilon)**0.5
        self.biases_update = factor * self.biases_gradient / (self.sum_b_grad_squares + self.epsilon)**0.5
        
        self.update_momentum_terms()
    
    def accumulate_grad_squares(self) -> None:
        self.sum_w_grad_squares += self.weights_gradient**2
        self.sum_b_grad_squares += self.biases_gradient**2

class RMSProp(GradientDescentAlgorithm):
    def __init__(self, x_train, y_train, network, learning_rate, error_function, stopping_condition,
                 forgetting_factor: float,
                 preprocessing_method_cls: type[DataPreprocessingMethod] = None,
                 regularization_term = None, momentum_rule = None, dilution_method: DilutionMethod = None, minibatch_size = None,
                 epsilon: float = 1e-10):
        super().__init__(x_train, y_train, network, learning_rate, error_function, stopping_condition, preprocessing_method_cls, regularization_term, momentum_rule, dilution_method, minibatch_size)
        self.epsilon: float = epsilon  # A small numerical value used in the denominator of the weights and biases updates to prevent division by 0.
        self.forgetting_factor: float = forgetting_factor

        self.avg_w_grad_squares: ListOfMatrices = ListOfMatrices([np.zeros_like(l.weights) for l in self.network.layers_with_weights])
        self.avg_b_grad_squares: ListOfVectors = ListOfVectors([np.zeros_like(l.biases) for l in self.network.layers_with_weights])
    
    def compute_necessary_quantities_for_network_update(self) -> None:
        self.compute_gradients()

        self.update_averages()
        factor = self.learning_rate*self.current_mb_size / len(self.training_set)
        self.weights_update = factor * self.weights_gradient / (self.avg_w_grad_squares + self.epsilon)**0.5
        self.biases_update = factor * self.biases_gradient / (self.avg_b_grad_squares + self.epsilon)**0.5
        
        self.update_momentum_terms()

    def update_averages(self) -> None:
        self.avg_w_grad_squares = self.forgetting_factor * self.avg_w_grad_squares + (1 - self.forgetting_factor) * self.weights_gradient ** 2
        self.avg_b_grad_squares = self.forgetting_factor * self.avg_b_grad_squares + (1 - self.forgetting_factor) * self.biases_gradient ** 2

class AdamVariant(GradientDescentAlgorithm):
    def __init__(self, x_train, y_train, network, learning_rate,
                 beta_1: float, beta_2: float,
                 error_function, stopping_condition,
                 preprocessing_method_cls: type[DataPreprocessingMethod] = None, regularization_term = None, momentum_rule = None, dilution_method: DilutionMethod = None, minibatch_size = None,
                 epsilon: float = 1e-10):
        super().__init__(x_train, y_train, network, learning_rate, error_function, stopping_condition, preprocessing_method_cls, regularization_term, momentum_rule, dilution_method, minibatch_size)
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2
        self.epsilon: float = epsilon
        self.t: int = 0  # The current training iteration. It increases at each minibatch being processed.

        self.w_m: ListOfMatrices = ListOfMatrices([np.zeros_like(l.weights) for l in self.network.layers_with_weights])  # Moving averages of the gradients of weights and biases.
        self.b_m: ListOfVectors = ListOfVectors([np.zeros_like(l.biases) for l in self.network.layers_with_weights])

        self.w_v: ListOfMatrices = ListOfMatrices([np.zeros_like(l.weights) for l in self.network.layers_with_weights])  # Moving averages of the squares of the gradients of weights and biases.
        self.b_v: ListOfVectors = ListOfVectors([np.zeros_like(l.biases) for l in self.network.layers_with_weights])
    
    @property
    def w_m_hat(self):
        return self.w_m / (1 - self.beta_1**self.t)
    
    @property
    def b_m_hat(self):
        return self.b_m / (1 - self.beta_1**self.t)
    
    @property
    def w_v_hat(self):
        return self.w_v / (1 - self.beta_2**self.t)
    
    @property
    def b_v_hat(self):
        return self.b_v / (1 - self.beta_2**self.t)
    
    def update_averages(self) -> None:
        uma = self.update_moving_average  # Give short name for convenience.
        self.w_m = uma(self.w_m, self.beta_1, self.weights_gradient)
        self.b_m = uma(self.b_m, self.beta_1, self.biases_gradient)
        
        self.w_v = uma(self.w_v, self.beta_2, self.weights_gradient**2)
        self.b_v = uma(self.b_v, self.beta_2, self.biases_gradient**2)

class Adam(AdamVariant):
    def compute_necessary_quantities_for_network_update(self) -> None:
        self.compute_gradients()

        self.update_averages()
        factor = self.learning_rate*self.current_mb_size / len(self.training_set)
        self.t += 1
        self.weights_update = factor * self.w_m_hat / (self.w_v_hat**0.5 + self.epsilon)
        self.biases_update = factor * self.b_m_hat / (self.b_v_hat**0.5 + self.epsilon)

        self.update_momentum_terms()

class NAdam(AdamVariant):
    def compute_necessary_quantities_for_network_update(self) -> None:
        self.compute_gradients()

        self.update_averages()
        factor = self.learning_rate*self.current_mb_size / len(self.training_set)
        self.t += 1
        self.weights_update = factor * (self.beta_1 * self.w_m_hat + (1 - self.beta_1)*self.weights_gradient / (1 - self.beta_1 ** self.t)) / (self.w_v_hat**0.5 + self.epsilon)
        self.biases_update = factor * (self.beta_1 * self.b_m_hat + (1 - self.beta_1)*self.biases_gradient / (1 - self.beta_1 ** self.t)) / (self.b_v_hat**0.5 + self.epsilon)

        self.update_momentum_terms()