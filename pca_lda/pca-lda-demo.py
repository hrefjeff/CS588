#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

########################### Pt 1. Random Generated Data

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Generate normal distribution data with N(0,1)
x1 = np.random.normal(0, 1, 1000)
x2 = np.random.normal(0, 1, 1000)
x3 = np.random.normal(0, 1, 1000)
X = np.vstack((x1, x2, x3)).T

plt.scatter(X[:, 0], X[:, 1])
plt.title('Normal Distribution Data N(0,1) - X')
plt.axis('equal')
plt.savefig('normal-dist.png')

# clear plot picture
plt.figure()

# Calculate covariance matrix & create heat map viz
X_df = pd.DataFrame(X, columns=['1', '2', '3'])
cov_matrix = np.cov(X_df.T, bias=True)
print(cov_matrix)

# Visualization
sns.heatmap(cov_matrix, annot=True)
plt.savefig('normal-dist-heatmap.png')

# clear plot picture
plt.figure()

# Find principal components
pca = PCA(n_components=2)
PCs = pca.fit_transform(X.T)

# display the PCs (or eigenvectors)
print('The PCs for data X generate = \n', PCs)

# Display contruition of each pc's
ev = pca.explained_variance_ratio_
print('\n The explained variance ration/variance in each PCA \n = ' , (ev*100))

# Plot variance/pc
plt.bar([1,2], list(ev*100), label='Principal Components', color='b')
plt.legend()
plt.xlabel('Principal Components')
pc = []
for i in range(2):
    pc.append('PC'+str(i+1))
plt.xticks([1,2], pc, fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio of Normal Distributed Data X')
plt.savefig('varianc-ratio-normal-dist-data.png')

# clear plot picture
plt.figure()

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
plt.savefig('normal-dist-pcplot.png')
########################### Pt 2. Iris Data

# Load data
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

# Only 2 features out of 4 being retained
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# There are 3 classes, LDA is N-1 for separation of each class
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each component
print(f'Explained variance ratio (first two components): {pca.explained_variance_ratio_}')

# Visualization time
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.savefig('iris-pca.png')

plt.figure()
for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.savefig('iris-lda.png')
