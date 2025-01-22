import mlproject.models as mod
import numpy as np

architecture: mod.NeuralNetworkArchitecture = mod.NeuralNetworkArchitecture([10, 20, 5], 2 * [mod.Sigmoid()],
                                                                            mod.Xavier())
nn = mod.NeuralNetwork.FromArchitecture(architecture)
dummy_data: np.ndarray = np.random.random((100, 10))
outputs = nn.compute_multiple_outputs(dummy_data)
assert outputs.shape == (100, 5), "Outputs are not of expected shape"  # It should be (100, 5), but for now I am writing a wrong shape on purpose to check how the assert keyword works
