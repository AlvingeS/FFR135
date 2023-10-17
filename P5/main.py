import numpy as np
import csv
from reservoir_computer import ReservoirComputer
from sklearn.linear_model import Ridge
from icecream import ic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Feed input data into reservoir through input nodes
# 2. Store the states of reservoir
#   - Set the input neurons to input data
#   - Propagate the states of the reservoir
#   - Store the states of the reservoir
# 3. Store your states in X and future states in Y
# 4. Run ridge regression on X and Y to get W_out
# 5. Set the weights of output layer
# 6. Feed test data into reservoir through input nodes
# 7. Feed the output of your reservoir after all test data to predict the future states
# 8. Store the predictions

def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    return np.array(data)

# Load data from file
training_data = load_data('training-set.csv')
testing_data = load_data('test-set-10.csv')

# Parameters
n_res_neurons = 500
n_inputs = 3
n_outputs = 3
regularization_param = 0.01
n_future_predictions = 500
n_training_samples = training_data.shape[1]
n_testing_samples = testing_data.shape[1]

reservoir_computer = ReservoirComputer(n_inputs, n_outputs, n_res_neurons)
X = np.zeros((n_training_samples, n_res_neurons))

for t in range(n_training_samples):
    data_point = training_data[:, t].reshape(-1, 1)
    reservoir_computer.propagate(data_point)    
    X[t, :] = reservoir_computer.reservoir_state.reshape(1, n_res_neurons)

X = X[:, :]
Y = training_data[:, :].T # CHECK THIS ONE!!!

# Ridge regression
model = Ridge(alpha=regularization_param)
model.fit(X, Y)
res_to_output_weights = model.coef_
reservoir_computer.res_to_output_weights = res_to_output_weights

# Reset reservoir state
reservoir_computer.reservoir_state = np.zeros((n_res_neurons, 1))

# Loop through testing data
for t in range(n_testing_samples):
    data_point = testing_data[:, t].reshape(-1, 1)
    reservoir_computer.propagate(data_point)

# Predict future states
predictions = np.zeros((n_outputs, n_future_predictions))
predictions[:, 0] = reservoir_computer.output_state.reshape(-1)

for t in range(n_future_predictions - 1):
    reservoir_computer.propagate(reservoir_computer.output_state)
    predictions[:, t + 1] = reservoir_computer.output_state.reshape(-1)

# Save only y-coordinate to csv-file on single csv-row
with open('prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(predictions[1, :])

# 3D plot of test data and predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(testing_data[0, :], testing_data[1, :], testing_data[2, :], label='Test data')
ax.plot(predictions[0, :], predictions[1, :], predictions[2, :], label='Predictions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
#plt.show()
plt.savefig('predictions.png')