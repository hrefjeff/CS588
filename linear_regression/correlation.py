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
df = pd.concat([iris_df, target_df], axis = 1)

#====Get Basic Info About Dataset
#print(iris_df.shape[0]) # prints rows
#print(iris_df.head())   # prints headers

# Iris data statistics
# Trying to use linear regression to predict any of
# the properties of the iris
#print(iris_df.describe())
print(df.info())

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 150 entries, 0 to 149
#Data columns (total 5 columns):
# #   Column             Non-Null Count  Dtype  
#---  ------             --------------  -----  
# 0   sepal length (cm)  150 non-null    float64
# 1   sepal width (cm)   150 non-null    float64
# 2   petal length (cm)  150 non-null    float64
# 3   petal width (cm)   150 non-null    float64
# 4   species            150 non-null    object 
#dtypes: float64(4), object(1)
#memory usage: 6.0+ KB

print(df.sample(10))
print(df.columns)
print(df.shape)

# Compute the correlation coefficient for iris dataset
print(df.corr(numeric_only=True))

#                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
#sepal length (cm)           1.000000         -0.117570           0.871754          0.817941
#sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126
#petal length (cm)           0.871754         -0.428440           1.000000          0.962865
#petal width (cm)            0.817941         -0.366126           0.962865          1.000000

# Notice diagonal have 1 to 1 corrolation because the values are the same
# The goal is to figure out how to attributes relate to each other
# Sepal length and sepal width = -0.117570 (weak corrolation)
# Sepal length and petal length = 0.871754 (strong corrolation, features that help identify between species)
# Only need half of table because the mirror image is on the bottom
# PETAL LENGTH & PETAL WIDTH = 0.962865 high corrolation. 
# We don't want too many features being corrolated and tightly bound, then we need something more to separate classifications

# Visualize iris features as a heatmap (Descriptive Analytics)
cor_eff = df.corr(numeric_only=True)
plt.figure(figsize = (6,6))
sns.heatmap(cor_eff, linecolor='white', linewidths=1, annot=True)
plt.savefig("corr_heatmap.png", bbox_inches="tight")

# How to print upper vs. lower portion of triangular matrix
fig, ax = plt.subplots(figsize=(6,6))
# compute correlation matrix
mask = np.zeros_like(cor_eff)

# mask = 0; display the correlation matrix, mask = 1; display the unique lower triangle values
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(cor_eff, linecolor='white', linewidths=1, mask=mask, ax=ax, annot=True)

plt.savefig("corr_bottom_heatmap.png", bbox_inches="tight")

g = sns.pairplot(df, hue='species') # class

plt.savefig("corr_pairplot.png")
