# Author: Mitchell Weingust
# Created: December 4, 2024
# License: MIT License
# Purpose: This python file includes code for reading data from the files in the data folder

# Usage: python ReadData.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 0 - added boilerplate code

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


"""
Loading and preprocessing of Data
"""
#Load datapoints in a pandas dataframe
data_folder = '../data/'
encoding = 'ISO-8859-1'

print("Loading dataset...")
data_1 = pd.read_csv(data_folder + 'sif_data_table_2017_2018_en.csv', encoding=encoding)
data_2 = pd.read_csv(data_folder + 'sif_data_table_2018_2019_en.csv', encoding=encoding)
data_3 = pd.read_csv(data_folder + 'sif_data_table_2019_2020_en.csv', encoding=encoding)
data_4 = pd.read_csv(data_folder + 'sif_data_table_2020_2021_en.csv', encoding=encoding)
data_5 = pd.read_csv(data_folder + 'sif_data_table_2021_2022_en.csv', encoding=encoding)

data_list = [data_1, data_2, data_3, data_4, data_5]

# drop first and last column 
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
dates = []
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
                    'Change in Grade 6 Mathematics Achievement Over Three Years',]

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

    # NOTE: ADD File year column
    df['Year'] = '1/1/' + str(current_year + i)

    # Encode school name and board name
    df['School Name'] = df['School Name'].astype('category').cat.codes
    df['Board Name'] = df['Board Name'].astype('category').cat.codes

    #convert the string col to numeric values
    for col in string_continous:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Update DataFrame in the list
    data_list[i] = df

# # Combine all cleaned DataFrames into a single DataFrame
# cleaned_data = pd.concat(data_list, ignore_index=True)

# # Print summary
# print("Data processing complete.")
# print(f"Final dataset shape: {cleaned_data.shape}")
# print(cleaned_data.head())

# cleaned_data.to_csv(data_folder + 'output_file.csv', index=False)

# preprocessing the cleaned data 
for i in range(len(data_list)):
    df = data_list[i]
    # min max scaling 
    scalar = MinMaxScaler()
    all_numeric_coloumn = continuous_columns + string_continous
    df[all_numeric_coloumn] = scalar.fit_transform(df[all_numeric_coloumn])

    # changing the date column
    df['Year'] = pd.to_datetime(
        df['Year']
    )

    data_list[i] = df

print(data_list[4].tail())

# Combine all cleaned DataFrames into a single DataFrame
cleaned_data = pd.concat(data_list, ignore_index=True)

# Print summary
print("Data processing complete.")
print(f"Final dataset shape: {cleaned_data.shape}")
print(cleaned_data.head())

cleaned_data.to_csv(data_folder + 'output_file.csv', index=False)

training_data =  pd.concat(data_list[0:(len(data_list)-1)], ignore_index=True)
test_data = data_list[len(data_list)]

"""
Building and Training Model
"""
class RNN:
    
    def __init__(self,data,time_steps):
        self.data = data
        self.time_steps = time_steps
        self.model = None

    def build_model(self, hidden_units, dense_units, input_shape, activation):
        self.model = Sequential()
        self.model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
        self.model.add(Dense(units=dense_units, activation=activation[1]))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
        return self.model
    
    def train(self, x_train, y_train, epochs):
        self.model.fit(x_train, y_train, epochs=epochs)

    def predict(self, input):
        return self.model.predict(input)


# defining the time_steps we have (by year)
time_steps = 1

# create an instance of the rnn
my_rnn = RNN(cleaned_data,time_steps)

# splitting into training and testing data
training_data =  pd.concat(data_list[0:(len(data_list)-1)], ignore_index=True)
test_data = data_list[len(data_list)]

# build and train the model
my_rnn.build_model(hidden_units=3, dense_units=1, input_shape=(), activation=['tanh', 'sigmoid'])
my_rnn.train_model()
