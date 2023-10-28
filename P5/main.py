import numpy as np
import csv
from reservoir_computer import ReservoirComputer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def main():
    # Load data from file
    training_data = np.loadtxt('training-set.csv', delimiter=',', dtype=np.float32)
    testing_data = np.loadtxt('test-set-10.csv', delimiter=',', dtype=np.float32)

    # Parameters
    n_res_neurons = 500
    n_outputs = 3
    regularization_param = 0.01
    n_future_predictions = 500
    (n_inputs, n_training_samples) = training_data.shape
    n_testing_samples = testing_data.shape[1]

    # Train reservoir computer
    reservoir_computer = ReservoirComputer(n_inputs, n_outputs, n_res_neurons)
    X = np.zeros((n_res_neurons, n_training_samples), dtype=np.float32)

    for t in range(n_training_samples):
        X[:, t] = reservoir_computer.reservoir_state.reshape(-1,)
        data_point = training_data[:, t].reshape(-1, 1)
        reservoir_computer.propagate(data_point)    

    # Ridge regression
    model = Ridge(alpha=regularization_param, fit_intercept=False)
    model.fit(X.T, training_data.T)
    reservoir_computer.res_to_output_weights = model.coef_

    # Reset reservoir state
    reservoir_computer.reservoir_state = np.zeros((n_res_neurons, 1))

    # Loop through testing data
    for t in range(n_testing_samples):
        data_point = testing_data[:, t].reshape(-1, 1)
        reservoir_computer.propagate(data_point)

    # Predict future states
    predictions = np.zeros((n_outputs, n_future_predictions), dtype=np.float32)

    for t in range(n_future_predictions):
        predictions[:, t] = reservoir_computer.output_state.reshape(-1,)
        reservoir_computer.propagate(predictions[:, t].reshape(-1, 1))

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
    plt.savefig('predictions.png')

if __name__ == '__main__':
    main()