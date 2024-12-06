"""
Created on Sat Sep 14 2024

@author: Zhaleh R.


Description: implementation of a simple recurrent neural network (RNN) for time series prediction

Refernces:
- https://www.kaggle.com/code/ludovicocuoghi/electric-production-forecast-lstm-sarima-mape-2-5    
- https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras

"""


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


# load dataset
df= pd.read_csv('EQAOPredictor-\data\output_file.csv',usecols=[1], engine='python')
data = np.array(df.values.astype('float32'))

"""A class to implement a simple recurrent neural network (RNN) for time series prediction"""
class SimpleRecurrentNeuralNetwork:
    
    def __init__(self,data,time_steps):
        self.data=data
        self.time_steps = time_steps
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None    

    """ Split data into training and testing sets"""
    def split_data(self, split_percent=0.8):
        # scale data
        data = self.scaler.fit_transform(self.data).flatten()
        n = len(data)
        
        # point for splitting data into train and test
        split = int(n*split_percent)
        # split the data into training and testing sets.
        train_data = data[range(split)]
        test_data = data[split:]
        return train_data, test_data, data
        
    """ Prepare the input X and target Y as a dataset for supervised learning""" 
    def supervised_df(self, input):
        
        # create an array of indices that will be used to extract the target values (Y) 
        # from the input data 
        Y_ind = np.arange(self.time_steps, len(input), self.time_steps)
        
        # use indices from Y_ind to extract the target values (Y) from the input data 
        Y = input[Y_ind]
        rows_x = len(Y)
        
        # extract the input data (X) from the original input data
        X = input[range(self.time_steps*rows_x)]
        
        # reshape the input data (X) into a 3D array 
        # array dimensions: (number of samples, number of time steps, number of features)
        X = np.reshape(X, (rows_x, self.time_steps, 1))    
        
        return X, Y
    
    """ Build a simpleRNN Keras model and compile it"""
    def build_model(self, hidden_units, dense_units, input_shape, activation):
        self.model= Sequential()
        self.model.add(SimpleRNN(hidden_units, input_shape= input_shape, activation=activation[0]))
        self.model.add(Dense(units=dense_units, activation=activation[1]))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model
    
    """ Train the simpleRNN model using traing data"""
    def train_model(self, X_train, y_train, epochs=20, batch_size=1):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    """ Predict new data using trained model"""
    def predict(self, input):        
        return self.model.predict(input) 
    

# Time Interval (frequency) to use for our model (1 means monthly data for Electric Production dataset)
# for example change it to 12 and it change the frequency of sampling to every 12 months (1 year)
time_steps = 12

# Create an instance of the SimpleRecurrentNeuralNetwork class
RNN = SimpleRecurrentNeuralNetwork(df, time_steps)

# Split the data into training and testing sets
train_data, test_data, data = RNN.split_data()

# Prepare the input X and target Y for training and testing sets
X_train, y_train = RNN.supervised_df(train_data)
X_test, y_test = RNN.supervised_df(test_data)

# Build and train the RNN model
RNN.build_model(hidden_units=3, dense_units=1, input_shape=(time_steps, 1), activation=['tanh', 'sigmoid'])
RNN.train_model(X_train, y_train)

# Make predictions on the training data and test data
train_predictions = RNN.predict(X_train)
test_predictions = RNN.predict(X_test)


# Calculate the root mean squared error (RMSE) for test set
print("RMSE:", 100 * math.sqrt(mean_squared_error(y_test, test_predictions)))

# Plot the results
actual = np.append(y_train, y_test)
predictions = np.append(train_predictions, test_predictions)
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