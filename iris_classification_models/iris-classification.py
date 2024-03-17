#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
    # Load Iris Data
    iris = load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Splitting the Dataset
    # In the real world, we test only a small sample
    # Use minimum amount of samples to pridict vast majority
    # 10% train, 90% test. 20% train, 80% test. 50% train, 50% test
    X_train, X_validation, Y_train, Y_validation = train_test_split(
                                            X,              # Data
                                            y,              # Labels
                                            test_size=0.20, #
                                            random_state=1,
                                            shuffle=True
                                        )

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) # More probablistic in nature
    models.append(('LDA', LinearDiscriminantAnalysis())) # LDA = max separate between, min separate within class
    models.append(('KNN', KNeighborsClassifier())) # K Nearest Neighbors
    models.append(('NB', GaussianNB())) #
    models.append(('SVM', SVC(gamma='auto')))

    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f %% (%f)' % (name, cv_results.mean()*100, cv_results.std()))

    # Compare algorithms
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.savefig('algo-comparison.png')

if __name__ == '__main__':
    main()
