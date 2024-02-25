#!/usr/bin/env python3

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris Data
iris = load_iris()

# Creating pd DataFrames
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
target_df = pd.DataFrame(data = iris.target, columns = ['species'])

# Generate labels
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'verginica'

target_df['species'] = target_df['species'].apply(converter)

# Concatenate the DataFrames
iris_df = pd.concat([iris_df, target_df], axis = 1)

# Iris data statistics
# Trying to use linear regression to predict any of
# the properties of the iris
print(iris_df.describe())

#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count         150.000000        150.000000         150.000000        150.000000
# mean            5.843333          3.057333           3.758000          1.199333
# std             0.828066          0.435866           1.765298          0.762238
# min             4.300000          2.000000           1.000000          0.100000
# 25%             5.100000          2.800000           1.600000          0.300000
# 50%             5.800000          3.000000           4.350000          1.300000
# 75%             6.400000          3.300000           5.100000          1.800000
# max             7.900000          4.400000           6.900000          2.500000

# Looking at the table above, the goal is to predict things. So drop
# one column like petal width, and based on the other 3 columns,
# predict what the petal width would be. Drop the species, and use all
# columns to predict the species

from sklearn.metrics import mean_squared_error, r2_score

# Converting Objects to Numerical Type
iris_df.drop('species', axis = 1, inplace = True)
target_df = pd.DataFrame(columns=['species'], data = iris.target)
iris_df = pd.concat([iris_df, target_df], axis = 1)

# Variables (Dropping sepal length to show we can predict it)
X = iris_df.drop(labels='sepal length (cm)', axis=1)
y = iris_df['sepal length (cm)']

# Splitting the Dataset
# In the real world, we test only a small sample
# Use minimum amount of samples to pridict vast majority
# 10% train, 90% test. 20% train, 80% test. 50% train, 50% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=111)

lr = LinearRegression()

# Fit LR Model
lr.fit(X_train, y_train)

# LR Prediction
#lr.predict(X_test)
y_pred = lr.predict(X_test)

# Quantitative Analysis - evaluate LR performance

# LR coefficients - beta/slope
print('LR beta/slope coefficient: ', lr.coef_)

# LR coefficients - alpha/slope intercept
print('LR alpha/slope intercept coefficient: ', lr.intercept_)

# coefficient of dtermination: 1 is the perfect prediction
print('Coefficient of determination: ', r2_score(y_test, y_pred))

# Model performance - Error (Quantitative analysis)
print('Root Mean Squared Error (RMSE): ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))

# LR beta/slope coefficient:  [ 0.61207486  0.78013546 -0.35124986 -0.34942695]
# The 4 values are because there are 4 different X's

# LR alpha/slope intercept coefficient:  1.8102535296995388
# This is the prediction for sepal length, when everything is 0.
# It'll start at the length 1.8cm

# Coefficient of determination:  0.8779831575882425
# Root Mean Squared Error (RMSE):  0.2934103969442157
# This is the error. It is a little high. Generally want this to be really low

# Mean Squared Error (MSE):  0.08608966103496224
# We want to find absolute value, don't care positive or negative

# Try adjusting test_size = 90%

# LR beta/slope coefficient:  [ 1.19265148  1.08254024 -1.87795011  0.63919001]
# LR alpha/slope intercept coefficient:  -0.1801215992495555
# Coefficient of determination:  0.6931549115115272
# Root Mean Squared Error (RMSE):  0.4558346450891425 <--- this increased in value
# Mean Squared Error (MSE):  0.20778522366354452      <--- this increased in value

# RMSE went up 0.29 --> 0.45 when test was 90%
# THIS IS CALCULATING WITHIN .45 CM OF ERROR
# Just trained the model on 10% of the data, so it's not very accurate
# Lower you train, the faster it is to train

# MOVING ON TO PREDICTION

# Predicting a new data point
print(iris_df.loc[16])

# Create a new dataframe
d = {
        'sepal length (cm)' : [5.4],
        'sepal width (cm)' : [3.9],
        'petal length (cm)' : [1.3],
        'petal width (cm)' : [0.4],
        'species' : 0
    }

pred_df = pd.DataFrame(data=d)

# Display the Dataframe
print('Data frame to be predicted: ')
print(pred_df)

# Predict the new data point using LR
pred = lr.predict(X_test)
print(pred)
print('Predicted Sepal Length (cm): ', pred[0])
print('Actual Sepal Length (cm): ', 5.4)

