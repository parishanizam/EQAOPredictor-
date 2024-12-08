
# Author: Mitchell Weingust, Allison Cook, Parisha Nizam
# Created: December 4, 2024
# License: MIT License
# Purpose: This python file includes code for reading data from the files in the data folder, creating and training a model

# Usage: python training.py

# Dependencies: None
# Python Version: 3.6+

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, precision_score
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from keras.layers import LSTM
import tensorflow as tf


"""
Loading and preprocessing of Data
"""
# Load datapoints in a pandas dataframe
data_folder = '../data/'
encoding = 'ISO-8859-1'

# loading data from csv file from years 2018-2022
print("Loading dataset...")
data_1 = pd.read_csv(data_folder + 'sif_data_table_2017_2018_en.csv', encoding=encoding)
data_2 = pd.read_csv(data_folder + 'sif_data_table_2018_2019_en.csv', encoding=encoding)
data_3 = pd.read_csv(data_folder + 'sif_data_table_2019_2020_en.csv', encoding=encoding)
data_4 = pd.read_csv(data_folder + 'sif_data_table_2020_2021_en.csv', encoding=encoding)
data_5 = pd.read_csv(data_folder + 'sif_data_table_2021_2022_en.csv', encoding=encoding)

data_list = [data_1, data_2, data_3, data_4, data_5]

# Drop first and last column
categorical_columns = ['School Name', 'Board Name', 'School Level', 'Grade Range']
continuous_columns = ['Enrolment',
                      'Percentage of Students Whose First Language Is Not English',
                      'Percentage of Students Whose First Language Is Not French',
                      'Percentage of Students Who Are New to Canada from a Non-English Speaking Country',
                      'Percentage of Students Who Are New to Canada from a Non-French Speaking Country',
                      'Percentage of Students Receiving Special Education Services',
                      'Percentage of Students Identified as Gifted',
                      'Percentage of School-Aged Children Who Live in Low-Income Households',
                      'Percentage of Students Whose Parents Have No Degree, Diploma or Certificate']

string_continous = ['Percentage of Grade 3 Students Achieving the Provincial Standard in Reading',
                    'Percentage of Grade 3 Students Achieving the Provincial Standard in Writing',
                    'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics',
                    'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
                    'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
                    'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics']

# coloumns to drop from dataset
columns_to_drop = ['Board Number', 'Board Type', 'School Number', 'School Type', 'School Special Condition Code',
                   'School Language', 'Building Suite', 'P.O. Box', 'Street', 'Municipality',
                   'City', 'Province', 'Postal Code', 'Phone Number', 'Fax Number',
                   'School Website', 'Board Website', 'Latitude', 'Longitude',
                   'Percentage of Grade 9 Students Achieving the Provincial Standard in Academic Mathematics',
                   'Change in Grade 9 Academic Mathematics Achievement Over Three Years',
                   'Change in Grade 9 Mathematics Achievement Over Three Years',
                   'Percentage of Grade 9 Students Achieving the Provincial Standard in Applied Mathematics',
                   'Percentage of Grade 9 Students Achieving the Provincial Standard in Mathematics',
                   'Change in Grade 9 Applied Mathematics Achievement Over Three Years',
                   'Percentage of Students That Passed the Grade 10 OSSLT on Their First Attempt',
                   'Change in Grade 10 OSSLT Literacy Achievement Over Three Years',
                   'Extract Date',                       
                   'Change in Grade 3 Reading Achievement Over Three Years',
                   'Change in Grade 3 Writing Achievement Over Three Years',
                   'Change in Grade 3 Mathematics Achievement Over Three Years',
                   'Change in Grade 6 Reading Achievement Over Three Years',
                   'Change in Grade 6 Writing Achievement Over Three Years',
                   'Change in Grade 6 Mathematics Achievement Over Three Years']

invalid_entries = ["NA", "N/R", "N/D", "SP"]

critical_columns = ['Percentage of Grade 3 Students Achieving the Provincial Standard in Reading',
                    'Percentage of Grade 3 Students Achieving the Provincial Standard in Writing',
                    'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics',
                    'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
                    'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
                    'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
                    'Enrolment']

current_year = 2018

# Process each dataframe
for i in range(len(data_list)):
    df = data_list[i]

    # Drop columns_to_drop
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop the rows that have grade 3 or grade 6 scores as invalid entries
    # Check for invalid entries in each column of critical_columns
    invalid_mask = df[critical_columns].isin(invalid_entries)

    # Check if any invalid entries are present in each row (across the critical columns)
    rows_to_drop = invalid_mask.any(axis=1)

    # Drop rows with invalid entries
    df = df.loc[~rows_to_drop]

    # Drop rows with empty values
    df = df.dropna()

    # Add a 'Year' column
    # df['Year'] = '1/1/' + str(current_year + i)
    df['Year'] = pd.to_datetime(f'{current_year + i}-01-01')

    # Encode school name and board name
    label_encoder = LabelEncoder()

    df['School Name'] = label_encoder.fit_transform(df['School Name'])# .astype('category').cat.codes
    df['Board Name'] = label_encoder.fit_transform(df['Board Name']) #.astype('category').cat.codes

    # Rearrange columns so the target column is the last one
    target_column = 'Percentage of Grade 3 Students Achieving the Provincial Standard in Reading'
    all_columns = list(df.columns)
    all_columns.remove(target_column)  # Remove the target column
    all_columns.append(target_column)  # Append it to the end
    df = df[all_columns]

    df = df.drop(columns=categorical_columns[2:], errors='ignore')

    # Convert string percentages to numeric
    for col in string_continous:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Min-max scaling to normalize data 
    scalar = MinMaxScaler()
    all_numeric_columns = continuous_columns + string_continous
    df[all_numeric_columns] = scalar.fit_transform(df[all_numeric_columns])

    # Update the cleaned DataFrame in the list
    data_list[i] = df

# Combine all cleaned DataFrames into a single DataFrame
cleaned_data = pd.concat(data_list, ignore_index=True)

# Print summary
print("Data processing complete.")
print(f"Final dataset shape: {cleaned_data.shape}")

# save cleaned data into a csv 
cleaned_data.to_csv(data_folder + 'output_file.csv', index=False)

# process and organize input data to do the following; 
# A year has data with school A, B, C ...
# we have 5 years of data. 
# group data using timestep such that we have lists of data for school A, from each year, then a list for school B with data from each year etc

# Group data by school and year
print("Grouping data by school and year...")
grouped_data = cleaned_data.groupby(['School Name', 'Board Name'])

# Create a dictionary where each school has a list of yearly data
school_yearly_data = {}
y_data = {}
i = 0
for school_name, group in cleaned_data.groupby(['School Name', 'Board Name']):
    group_sorted = group.sort_values('Year')
    school_data = group_sorted.drop(columns=['School Name', 'Year', 'Board Name', target_column]).values
    school_yearly_data[i] = school_data[:-1]  # Use all but the last year for training
    y_data[i] = school_data[:-1, -1]  # Use the last column as the target
    i += 1
# print("Columns after preprocessing:", group_sorted.drop(columns=['School Name', 'Year']).columns)

# # Ensure the training data (features) and target data are separated correctly
# print("\nFeature columns for training data (x_train):")
# feature_columns = list(cleaned_data.columns.difference(['School Name', 'Year', target_column]))
# print(feature_columns)

print("\nTarget column for training data (y_train):")
print(target_column)

# Pad sequences for x_train (ensure the correct shape for input values)
x_train = pad_sequences(
    list(school_yearly_data.values()), 
    maxlen=5, 
    dtype='float32', 
    padding='post'
)

# print(x_train)


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
for school_name, group in data_list[-1].groupby(['School Name', 'Board Name']):
    # group_sorted = group.sort_values('Year')
    school_data = group.drop(columns=['School Name', 'Year', 'Board Name', target_column]).values
    school_yearly_data_test[i] = school_data[:]
    y_test_data[i] = school_data[:,-1]  # Use the last column as the target
    i += 1

# maximum time steps to have consistent sequence 
# max_time_steps = max(len(data) for data in school_yearly_data_test.values())

# x_test = pd.Series(list(school_yearly_data_test.values()))
# y_test = pd.DataFrame(list(y_test_data.values()))

# Pad sequences for x_train (ensure the correct shape for input values)
x_test = pad_sequences(
    list(school_yearly_data_test.values()), 
    maxlen=5, 
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

# Ensure the data shapes are correct for RNN
samples, time_steps, features = x_train.shape
print(x_train.shape)
print(x_test.shape)

# # Example output for a specific school
# example_school = school_yearly_data[0]  # Get an example school name
# # print(f"Yearly data for School : {school_name} , {example_school}:")
# # print(school_yearly_data)

"""
Building Model
"""

class RNN:
    
    def __init__(self):
        # self.data = data
        # self.time_steps = time_steps
        self.model = None

    def build_model(self, hidden_units, dense_units, input_shape, activation):
        self.model = Sequential()
        # self.model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))

        # trying LTSM technique 
        self.model.add(LSTM(hidden_units, input_shape=input_shape, activation=activation[0]))
        self.model.add(Dense(units=dense_units, activation=activation[1]))
        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

        # Using MSE for model 
        # self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        return self.model
    
    def train(self, x_train, y_train, epochs):
        self.model.fit(x_train, y_train, epochs=epochs)

    def predict(self, input):
        return self.model.predict(input)


# defining the time_steps we have (by year)
time_steps = 5

"""
Training Model
"""

my_rnn = RNN()
# my_rnn.build_model(hidden_units=50, dense_units=1, input_shape=(time_steps, features), activation=['tanh', 'linear'])
my_rnn.build_model(hidden_units=75, dense_units=1, input_shape=(time_steps, features), activation=['relu', 'linear'])
my_rnn.train(x_train, y_train, epochs=10)

# Make predictions and targets, flatten results
train_predict = my_rnn.predict(x_train)
actual_y = y_train.flatten()[:train_predict.size]

# testing
test_predict = my_rnn.predict(x_test)
actual_test_y = y_test.flatten()[:test_predict.size]

"""
Calculate RMSE and Plot Results of the Model
"""

# RMSE calculation 
rmse = math.sqrt(mean_squared_error(actual_y, train_predict.flatten()))
print("RMSE:", rmse)
rmse_test = math.sqrt(mean_squared_error(actual_test_y, test_predict.flatten()))
print("Test RMSE:", rmse_test)

# Plot the actual results vs predicted results 
# plt.figure(figsize=(10, 6))
# plt.plot(actual_test_y, label='Actual')
# plt.plot(test_predict.flatten(), label='Predicted', linestyle='--')
# plt.legend()
# plt.xlabel('Samples')
# plt.ylabel('Target Values')
# plt.title('RNN Prediction vs Actual')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(actual_y, label='Actual')
# plt.plot(train_predict.flatten(), label='Predicted', linestyle='--')
# plt.legend()
# plt.xlabel('Samples')
# plt.ylabel('Target Values')
# plt.title('RNN Prediction vs Actual')
# plt.show()
actual = np.append(y_train, y_test)
predictions = np.append(train_predict, test_predict)
rows = len(actual)
plt.figure(figsize=(15, 6), dpi=80)
plt.plot(range(rows), actual)
plt.plot(range(rows), predictions, color='r')
plt.axvline(x=len(y_train), color='g')
plt.legend(['Actual', 'Predictions'])
plt.xlabel('Time Steps')
plt.ylabel('Scaled Data')
plt.suptitle('RNN Prediction Demo', fontsize=16)
plt.title('Train {green line} Test')
plt.show()