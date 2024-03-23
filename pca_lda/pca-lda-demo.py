#!/usr/bin/env python3

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
