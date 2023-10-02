import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./output_parameters/validation_results.csv')

# Create the scatter plot
plt.scatter(df[df['Label'] == -1]['X'], df[df['Label'] == -1]['Y'], color='red')
plt.scatter(df[df['Label'] == 1]['X'], df[df['Label'] == 1]['Y'], color='green')

# Save the plot as a PNG file
plt.savefig('validation_results.png')

# Close the plot
plt.close()