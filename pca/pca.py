#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io

from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA

# load dataset into Pandas DataFrame
#data = files.upload()

# load groundtruth data
#gth_data = files.upload()


def main():
    df = scipy.io.loadmat(r'indianR.mat')
    print(df.keys())
    print(df['num_bands'])

    x = np.array(df['X'])
    gth = np.array(df['gth'])
    num_rows = np.array(df['num_rows'])
    num_cols = np.array(df['num_cols'])
    num_bands = np.array(df['num_bands'])
    bands,samples = x.shape

    # load ground truth data (class labels to the data)
    gth_mat = scipy.io.loadmat(r'indian_gth.mat')
    gth_mat = {i:j for i, j in gth_mat.items() if i[0] != '_'}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})

    # List features
    n = []
    ind = []
    for i in range(bands):
        n.append(i+1)
    for i in range(bands):
        ind.append('band'+str(n[i]))

    features = ind

    # Normalize the features
    # PREPROSSING DATASET
    from sklearn.preprocessing import MinMaxScaler
    scaler_model = MinMaxScaler()
    scaler_model.fit(x.astype(float))
    x = scaler_model.transform(x)

    # Finding the principle components
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(x)

    # Display contribution of each pc's
    ev = pca.explained_variance_ratio_
    print(ev)
    # These values are normalized to a scale of 1.
    # The first principal component shows .95 (95%) of data
    # The second principal component shows .39 (3%-4$) of data
    # So with the first 2 PC's, we get 99% of information being preserved
    # [9.57944966e-01 3.91813660e-02 1.42072841e-03 8.83148843e-04
    # 1.60425500e-04 1.09512171e-04 5.47040669e-05 2.96210469e-05
    # 2.11792766e-05 1.84743584e-05]

    # Select the number of components to retain
    # -----------------------------------------
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents,
                               columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10'])

    # Adding labels
    finalDf = pd.concat([principalDf, gt], axis = 1)

    # Dimensionality reduction using PCA
    x1 = x.transpose()
    X_pca = np.matmul(x1, principalComponents)

    X_pca.shape

    # Model x as a dataframe
    x_pca_df = pd.DataFrame(data = X_pca,
                            columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10'])

    # Adding labels
    X_pca_df = pd.concat([x_pca_df, gt], axis = 1)

    # Display the new PCA reduced ata
    print(X_pca_df)

    # ADDING SOME VISUALIZATION
    # Bar graph for explained variance ratio
    plt.bar([1,2,3,4,5,6,7,8,9,10], list(ev*100), label='Principal Components', color='b')
    plt.legend()
    plt.xlabel('Principal Components')
    pc=[]
    for i in range(10):
        pc.append('PC'+str(i+1))
    plt.xticks([1,2,3,4,5,6,7,8,9,10],pc,fontsize=8,rotation=30)
    plt.ylabel('Variance Ratio')
    plt.title('Variance Ratio of INDIAN PINES Dataset')
    plt.savefig('indianpines-variance-ratio.png')

    # ------------------------------
    # Most important takeaways
    # Dataset visualization with PCA
    # vs without PCA applied
    # ------------------------------
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC-1', fontsize = 15)
    ax.set_ylabel('PC-2', fontsize = 15)
    ax.set_title('PCA on INDIAN PINES Dataset', fontsize = 20)
    class_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    colors = ['r','g','b','y','m','c','k','r','g','b','y','m','c','k','b','r']
    markerm = ['o','o','o','o','o','o','o','+','+','+','+','+','+','+','*','*']
    for target, color, m in zip(class_num,colors,markerm):
        indicesToKeep = X_pca_df['gth'] == target
        ax.scatter(X_pca_df.loc[indicesToKeep, 'PC-1'],
                   X_pca_df.loc[indicesToKeep, 'PC-2'],
                   c = color, marker = m, s = 9)
    ax.legend(class_num)
    ax.grid()
    plt.savefig('pca-indianpines.png')

if __name__ == '__main__':
    main()
