import numpy as np

import mlproject.models as mod
import mlproject.training as tr
from mlproject.utils.Keys import Keys


np.random.seed(42)

architecture = mod.NeuralNetworkArchitecture([5, 20, 7], [mod.Sigmoid(), mod.ActivationFunction.Identity()], mod.Xavier())
nn = mod.NeuralNetwork.FromArchitecture(architecture)


dummy_x = np.random.normal(size= (100, 5))  # Dummy x data.
dummy_y = np.random.normal(size=  (100, 7))  # Dummy y data.

train_alg = tr.ClassicalBackprop(dummy_x, dummy_y, nn, 0.01, tr.MSE(), tr.ThresholdOnTrainingError(0, 1))
train_alg.run(50)

tr_errs = train_alg.history[Keys.TR_ERROR]

assert tr_errs[0] > tr_errs[-1], "The training error seems to be going up instead of down."