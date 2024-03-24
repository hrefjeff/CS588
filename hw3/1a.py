#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load data

## Iris
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

## Indian Pines
df = scipy.io.loadmat(r'indianR.mat')
print(df.keys())
print(df['num_bands'])

x = np.array(df['X'])
gth = np.array(df['gth'])
num_rows = np.array(df['num_rows'])
num_cols = np.array(df['num_cols'])
num_bands = np.array(df['num_bands'])
bands,samples = x.shape

### load ground truth data (class labels to the data)
gth_mat = scipy.io.loadmat(r'indian_gth.mat')
gth_mat = {i:j for i, j in gth_mat.items() if i[0] != '_'}
gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})

### Preprocessing of indian pines data set
scaler_model = MinMaxScaler()
scaler_model.fit(x.astype(float))
x = scaler_model.transform(x)

# part i

## Iris PCA
iris_pca = PCA(n_components=4)
iris_PCs = iris_pca.fit_transform(X.T)
iris_ev = iris_pca.explained_variance_ratio_

## Indian Pines PCA
indianpines_pca = PCA(n_components=10)
indianpines_PCs = indianpines_pca.fit_transform(x)
indianpines_ev = indianpines_pca.explained_variance_ratio_

# Percentage of variance explained for each component
print(f'Explained variance ratio of Iris (includes all features): {iris_ev}')

# Percentage of variance explained for each component
print(f'Explained variance ratio of Indian Pines: {indianpines_ev}')

# display the PCs (or eigenvectors)
print('\nThe PCs for Iris data X generate = \n', iris_PCs)
print('\nThe PCs for Indian Pines data X generate = \n', indianpines_PCs)

# Display contruition of each pc's
print('\n Iris explained variance ration/variance in each PCA \n = ' , (iris_ev*100))
print('\n Indian Pines explained variance ration/variance in each PCA \n = ' , (indianpines_ev*100))

# clear plot picture
plt.figure()

# Plot variance/pc
plt.bar([1,2,3,4], list(iris_ev*100), label='Principal Components', color='b')
plt.legend()
plt.xlabel('Principal Components')
pc = []
for i in range(4):
    pc.append('PC'+str(i+1))
plt.xticks([1,2,3,4], pc, fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio of Iris Dataset (All 4 Features)')
plt.savefig('explained-variance-iris.png')

# clear plot picture
plt.figure()

plt.bar([1,2,3,4,5,6,7,8,9,10], list(indianpines_ev*100), label='Principal Components', color='b')
plt.legend()
plt.xlabel('Principal Components')
pc=[]
for i in range(10):
    pc.append('PC'+str(i+1))
plt.xticks([1,2,3,4,5,6,7,8,9,10],pc,fontsize=8,rotation=30)
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio of INDIAN PINES Dataset')
plt.savefig('explained-variance-indianpines.png')

# part ii
pca = PCA(n_components=2)
PCs = pca.fit_transform(X.T)

# PCA transformed data
Y = np.matmul(X, PCs)

# PCA visualization (plot eig val and eig vectors)
C = np.cov(Y.T)

eig_vec, eig_val = np.linalg.eig(C)
print('The eig values computed for x = \n', eig_val)
print('\nThe eig vectors (PCs) computed for X = \n', eig_vec)

plt.scatter(Y[:,0], Y[:,1])
for e, v in zip(eig_vec, eig_val.T):
    plt.plot([0, 5*np.sqrt(e)*v[0]], [0, 5*np.sqrt(e)*v[1]], 'k-', lw=2)
plt.title('PCA Transformed data-Y=T(X)')
plt.axis('equal')
plt.savefig('iris-pcplot.png')

# Part iii

# There are 3 classes, LDA is N-1 for separation of each class
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Visualization time
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.savefig('iris-lda.png')
