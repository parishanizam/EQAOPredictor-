
# Author: Mitchell Weingust, Allison Cook, Parisha Nizam
# Created: December 4, 2024
# License: MIT License
# Purpose: This python file includes code for testing and evaluating the model 

import math
import pickle
import keras
import numpy as np 
import pandas as pd
import matplotlib as plt

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score


"""
Loading and formating testing data
"""
print("Loading model and testing data...")
data_folder = '../data/'
test_data = pd.read_csv(data_folder + 'test_data.csv')
training_data = pd.read_csv(data_folder + 'output_file.csv')
target_column = 'Percentage of Students Identified as Gifted'

# Create a dictionary where each school has a list of yearly data
school_yearly_data = {}
y_data = {}
i = 0
for school_name, group in training_data.groupby(['School Name', 'Board Name']):
    group_sorted = group.sort_values('Year')
    school_data = group_sorted.drop(columns=['School Name', 'Year', 'Board Name', target_column]).values
    school_yearly_data[i] = school_data[:-1]  # Use all but the last year for training
    y_data[i] = school_data[:-1, -1]  # Use the last column as the target
    i += 1

# Pad sequences for x_train (ensure the correct shape for input values)
x_train = pad_sequences(
    list(school_yearly_data.values()), 
    maxlen=5, 
    dtype='float32', 
    padding='post'
)

# Pad sequences for y_train (ensure the correct shape for target values)
y_train = pad_sequences(
    list(y_data.values()), 
    maxlen=1,  # Targets will be one step fewer
    dtype='float32', 
    padding='post'
)

# Create a dictionary where each school has a list of yearly data
school_yearly_data_test = {}
y_test_data = {}
i = 0
for school_name, group in test_data.groupby(['School Name', 'Board Name']):
    # group_sorted = group.sort_values('Year')
    school_data = group.drop(columns=['School Name', 'Year', 'Board Name', target_column]).values
    school_yearly_data_test[i] = school_data[:]
    y_test_data[i] = school_data[:,-1]  # Use the last column as the target
    i += 1

# Pad sequences for x_train (ensure the correct shape for input values)
x_test = pad_sequences(
    list(school_yearly_data_test.values()), 
    maxlen=1, 
    dtype='float32', 
    padding='post'
)

# Pad sequences for y_train (ensure the correct shape for target values)
y_test = pad_sequences(
    list(y_test_data.values()), 
    maxlen=1,  # Targets will be one step fewer
    dtype='float32', 
    padding='post'
)

# load the model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions and targets, flatten results
train_predict = loaded_model.predict(x_train)
actual_y = y_train.flatten()[:train_predict.size]

# testing
test_predict = loaded_model.predict(x_test)
actual_test_y = y_test.flatten()[:test_predict.size]

"""
Accuracy, Precision, Recall Calculations
"""

# Threshold predictions to binary classification (e.g., 0.5 threshold)
train_predict_binary = (train_predict.flatten() > 0.5).astype(int)
y_train_binary = (y_train.flatten() > 0.5).astype(int)

test_predict_binary = (test_predict.flatten() > 0.5).astype(int)
y_test_binary = (y_test.flatten() > 0.5).astype(int)

# Calculate precision, recall, and accuracy for training data
train_precision = precision_score(y_train_binary, train_predict_binary)
train_accuracy = accuracy_score(y_train_binary, train_predict_binary)

# Calculate precision, recall, and accuracy for testing data
test_precision = precision_score(y_test_binary, test_predict_binary)
test_accuracy = accuracy_score(y_test_binary, test_predict_binary)

# Print results
print("Training Metrics:")
print(f"Precision: {train_precision:.4f}")
print(f"Accuracy: {train_accuracy:.4f}")

print("\nTesting Metrics:")
print(f"Precision: {test_precision:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")


"""
Calculate RMSE and Plot Results of the Model
"""

# RMSE calculation 
rmse = math.sqrt(mean_squared_error(actual_y, train_predict.flatten()))
print("RMSE:", rmse)
rmse_test = math.sqrt(mean_squared_error(actual_test_y, test_predict.flatten()))
print("Test RMSE:", rmse_test)

# ploting 
actual = np.append(y_train, y_test)
predictions = np.append(train_predict, test_predict)
rows = len(actual)
plt.figure(figsize=(15, 6), dpi=80)
plt.plot(range(rows), actual)
plt.plot(range(rows), predictions, color='o', linestyle='--')
plt.axvline(x=len(y_train), color='g')
plt.legend(['Actual', 'Predictions'])
plt.xlabel('Time Steps')
plt.ylabel('Scaled Data')
plt.suptitle('RNN Prediction Demo', fontsize=16)
plt.title('Train {green line} Test')
plt.show()