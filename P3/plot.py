import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('validation_results.csv')

# Filter data by label
df_minus_one = df[df['Label'] == -1]
df_plus_one = df[df['Label'] == 1]

# Create the plot
plt.scatter(df_minus_one['X'], df_minus_one['Y'], color='red', label='-1')
plt.scatter(df_plus_one['X'], df_plus_one['Y'], color='green', label='+1')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Validation Results')
plt.legend()

# Show the plot
plt.savefig('validation_results.png')