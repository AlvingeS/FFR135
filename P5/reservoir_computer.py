import numpy as np

class ReservoirComputer:
    def __init__(self, n_inputs, n_outputs, n_res_neurons, mean=0, input_variance=0.002, res_variance=2/500):
        self.n_reservoir = n_res_neurons
        self.input_to_res_weights = np.random.normal(mean, np.sqrt(input_variance), (n_res_neurons, n_inputs))
        self.res_to_res_weights = np.random.normal(mean, np.sqrt(res_variance), (n_res_neurons, n_res_neurons))
        self.res_to_output_weights = np.zeros((n_outputs, n_res_neurons))
        self.reservoir_state = np.zeros((n_res_neurons, 1))
        self.output_state = np.zeros((n_outputs, 1))
        
    def propagate(self, input_signals):
        # Update reservoir state
        self.reservoir_state = np.tanh(np.matmul(self.res_to_res_weights, self.reservoir_state) + np.matmul(self.input_to_res_weights, input_signals))
        self.output_state = np.matmul(self.res_to_output_weights, self.reservoir_state)