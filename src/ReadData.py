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

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

#Load datapoints in a pandas dataframe
print("Loading dataset...")
data_1 = pd.read_csv('sif_data_table_2017_2018_en.csv')
data_2 = pd.read_csv('sif_data_table_2018_2019_en.csv')
data_3 = pd.read_csv('sif_data_table_2019_2020_en.csv')
data_4 = pd.read_csv('sif_data_table_2020_2021_en.csv')
data_5 = pd.read_csv('sif_data_table_2021_2022_en.csv')

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
                      'Percentage of Grade 3 Students Achieving the Provincial Standard in Reading',
                      'Change in Grade 3 Reading Achievement Over Three Years',
                      'Percentage of Grade 3 Students Achieving the Provincial Standard in Writing',
                      'Change in Grade 3 Writing Achievement Over Three Years',
                      'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics',
                      'Change in Grade 3 Mathematics Achievement Over Three Years',
                      'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
                      'Change in Grade 6 Reading Achievement Over Three Years',
                      'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
                      'Change in Grade 6 Writing Achievement Over Three Years',
                      'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
                      'Change in Grade 6 Mathematics Achievement Over Three Years',
                      'Percentage of School-Aged Children Who Live in Low-Income Households',
                      'Percentage of Students Whose Parents Have No Degree, Diploma or Certificate']
dates = []
columns_to_drop = ['Board Number', 'Board Type', 'School Number', 'School Type', 'School Special Condition Code',
                   'School Language', 'Building Suite', 'P.O. Box', 'Street', 'Municipality',
                   'City', 'Province', 'Postal Code', 'Phone Number', 'Fax Number',
                   'School Website', 'Board Website', 'Latitude', 'Longitude',
                   'Percentage of Grade 9 Students Achieving the Provincial Standard in Academic Mathematics',
                   'Change in Grade 9 Academic Mathematics Achievement Over Three Years',
                   'Percentage of Grade 9 Students Achieving the Provincial Standard in Applied Mathematics',
                   'Change in Grade 9 Applied Mathematics Achievement Over Three Years',
                   'Percentage of Students That Passed the Grade 10 OSSLT on Their First Attempt',
                   'Change in Grade 10 OSSLT Literacy Achievement Over Three Years',
                   'Extract Date']

for df in data_list:
    # Drop columns_to_drop
    df = df.drop(columns=columns_to_drop)

    # Drop rows with empty values
    # NOTE: Drop the rows that have grade 3 or grade 6 scores as NA
    df = df.dropna()

    # NOTE: ADD File year column

    # I think this drops headings?

    # 1. Extract column headings
    # 2. remove column headings from all files

    df.drop(df.columns[[-1, 0]], axis=1, inplace=True)

    # Combine data sets at the end?
    # Re-add column headings? (is this necessary?)