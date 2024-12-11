import pandas as pd
import numpy as np

# Create DataFrame from the results
# First create a list of column names
columns = ['look_back', 'two_lstm_layers', 'lstm_units', 'lstm_activation',
           'lstm_recurrent_activation', 'epochs', 'mae', 'rmse']

# Create the DataFrame (assuming data is stored in a variable called 'data')
# For this example, I'll show how to analyze the impact:

df = pd.read_csv('btc_results2.csv')

# Calculate average RMSE for each parameter value
param_impacts = {}

# Numerical parameters
for param in ['look_back', 'lstm_units', 'epochs']:
    means = df.groupby(param)['rmse'].mean().sort_values()
    param_impacts[param] = means

# Categorical parameters
for param in ['lstm_activation', 'lstm_recurrent_activation']:
    means = df.groupby(param)['rmse'].mean().sort_values()
    param_impacts[param] = means

# Calculate correlations for numerical parameters
correlations = df[['look_back', 'lstm_units', 'epochs', 'rmse']].corr()['rmse']

print("Correlations with RMSE:")
print(correlations)

print("\nAverage RMSE by parameter values:")
for param, impact in param_impacts.items():
    print(f"\n{param}:")
    print(impact)
