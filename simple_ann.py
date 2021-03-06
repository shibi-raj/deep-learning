#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shibi
"""

# -*- coding: utf-8 -*-

""" Preprocessing """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Sex
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Dummies for country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Drop redundant dummy
X = X[:,1:]


# Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



""" Make the ANN """
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN with successive layers
classifier = Sequential()

# Create input and first hidden layer (RELU internal activations, avg input/output for # nodes)
classifier.add(Dense(output_dim=6,init='uniform', activation='relu', input_dim=11))

# Add another layer
classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))

# Sigmoid output layer for probabilistic interpretation, segementation
classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



""" Fit the ANN to the training set """
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)



""" Predicting the Test set results """
y_pred = classifier.predict(X_test)
# threshold data
y_pred = (y_pred > 0.5)



""" Confusion Matrix """
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



""" Predict new data """
x = sc.transform(np.array([[0., 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))

y_pred0 = classifier.predict(x)

y_pred0 = (y_pred0 > 0.5)