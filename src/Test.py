# Author: Mitchell Weingust, Allison Cook, Parisha Nizam
# Created: December 4, 2024
# License: MIT License
# Purpose: This python file includes code for testing and evaluating the model 

import math
import pickle
import keras
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score
import Training


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
# NOTE: This pickle model does not capture all the complexities of the h5 file.
#       Therefore, we recommend loading in the model.h5 file instead. 
# with open('model.pkl', 'rb') as f:
#     model_path = pickle.load(f)
# loaded_model = keras.models.load_model('model.pkl')

# load the model
loaded_model = keras.models.load_model('model.h5')

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
Looking at possible bias in the model
"""
print("Examining Possible Bias")
test = test_predict.reshape(-1)
difference = actual_test_y - test

max = difference.argmax()
min = difference.argmin()
print(f"Actual Value: {actual_test_y[max]:.4f}")
print(f"Predicted Value: {test[max]:.4f}")
print("Other factors:")
print(f"Percentage of Students Whose First Language Is Not English: {x_test[max][0][0]}")
print(f"Percentage of Students Who Are New to Canada from a Non-English Speaking Country: {x_test[max][0][2]}")
print(f"Percentage of Students Receiving Special Education Services: {x_test[max][0][5]}")
print(f"Percentage of School-Aged Children Who Live in Low-Income Households: {x_test[max][0][12]}")
print(f"Percentage of Students Whose Parents Have No Degree, Diploma or Certificate: {x_test[max][0][13]}")
print(f"Actual Value: {actual_test_y[min]:.4f}")
print(f"Predicted Value: {test[min]:.4f}")
print(f"Percentage of Students Whose First Language Is Not English: {x_test[min][0][0]}")
print(f"Percentage of Students Who Are New to Canada from a Non-English Speaking Country: {x_test[min][0][2]}")
print(f"Percentage of Students Receiving Special Education Services: {x_test[min][0][5]}")
print(f"Percentage of School-Aged Children Who Live in Low-Income Households: {x_test[min][0][12]}")
print(f"Percentage of Students Whose Parents Have No Degree, Diploma or Certificate: {x_test[min][0][13]}")

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
plt.plot(range(rows), predictions, color='r', linestyle='--')
plt.axvline(x=len(y_train), color='g')
plt.legend(['Actual', 'Predictions'])
plt.xlabel('Time Steps')
plt.ylabel('Scaled Data')
plt.suptitle('RNN Prediction', fontsize=16)
plt.title('Train {green line} Test')
plt.show()