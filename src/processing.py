import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

# Load datapoints in a pandas dataframe
data_folder = '../data/'
encoding = 'ISO-8859-1'

print("Loading dataset...")
data_1 = pd.read_csv(data_folder + 'sif_data_table_2017_2018_en.csv', encoding=encoding)
data_2 = pd.read_csv(data_folder + 'sif_data_table_2018_2019_en.csv', encoding=encoding)
data_3 = pd.read_csv(data_folder + 'sif_data_table_2019_2020_en.csv', encoding=encoding)
data_4 = pd.read_csv(data_folder + 'sif_data_table_2020_2021_en.csv', encoding=encoding)
data_5 = pd.read_csv(data_folder + 'sif_data_table_2021_2022_en.csv', encoding=encoding)

data_list = [data_1, data_2, data_3, data_4, data_5]

# Define columns
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

target_columns = ['Percentage of Grade 3 Students Achieving the Provincial Standard in Reading',
                  'Percentage of Grade 3 Students Achieving the Provincial Standard in Writing',
                  'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics',
                  'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
                  'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
                  'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics']

columns_to_drop = ['Board Number', 'Board Type', 'School Number', 'School Type', 'School Special Condition Code',
                   'School Language', 'Building Suite', 'P.O. Box', 'Street', 'Municipality',
                   'City', 'Province', 'Postal Code', 'Phone Number', 'Fax Number',
                   'School Website', 'Board Website', 'Latitude', 'Longitude']

invalid_entries = ["NA", "N/R", "N/D", "SP"]

current_year = 2018

# Process each dataframe
for i in range(len(data_list)):
    df = data_list[i]

    # Drop unnecessary columns
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop invalid entries in target columns
    invalid_mask = df[target_columns].isin(invalid_entries)
    rows_to_drop = invalid_mask.any(axis=1)
    df = df.loc[~rows_to_drop]

    # Drop rows with empty values
    df = df.dropna()

    # Add a 'Year' column
    df['Year'] = f'1/1/{current_year + i}'

    # Encode school name and board name
    df['School Name'] = df['School Name'].astype('category').cat.codes
    df['Board Name'] = df['Board Name'].astype('category').cat.codes

    # Convert percentage strings to numeric
    for col in target_columns:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Update cleaned DataFrame in the list
    data_list[i] = df

# Combine all cleaned DataFrames into a single DataFrame
cleaned_data = pd.concat(data_list, ignore_index=True)

# Separate features and targets
features = cleaned_data.drop(columns=target_columns)
targets = cleaned_data[target_columns]

# Print summary
print("Data processing complete.")
print(f"Final dataset shape: {cleaned_data.shape}")
print(cleaned_data.head())

# Group data by School Name and organize yearly data
print("Grouping data by school and organizing yearly data...")
school_yearly_data = {}

for school_name, school_data in cleaned_data.groupby('School Name'):
    yearly_data = []
    for year, year_data in school_data.groupby('Year'):
        yearly_features = year_data.drop(columns=['School Name', 'Year'] + target_columns).values.tolist()
        yearly_targets = year_data[target_columns].values.tolist()
        yearly_data.append({'features': yearly_features, 'targets': yearly_targets})
    school_yearly_data[school_name] = yearly_data

# Example output for a specific school
example_school = next(iter(school_yearly_data.keys()))  # Get an example school name
print(f"Data for School {example_school}:")
print(school_yearly_data[example_school])
