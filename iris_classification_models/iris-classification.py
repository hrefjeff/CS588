#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from matplotlib import pyplot


from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix


def plot_learning_curve(classifier, X, y, steps=10, train_sizes=np.linspace(0.1, 1.0, 10), label="", color="r", axes=None):
    estimator = Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    train_scores = []
    test_scores = []
    train_sizes = []
    for i in range(0, X_train.shape[0], X_train.shape[0] // steps):
        if (i == 0):
            continue
        X_train_i = X_train[0:i, :]
        y_train_i = y_train[0:i]
        estimator.fit(X_train_i, y_train_i)
        train_scores.append(estimator.score(X_train_i, y_train_i) * 100)
        test_scores.append(estimator.score(X_test, y_test) * 100)
        train_sizes.append(i + 1)

    if (X_train.shape[0] % steps != 0):
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.score(X_train, y_train) * 100)
        test_scores.append(estimator.score(X_test, y_test) * 100)
        train_sizes.append(X_train.shape[0])

    if axes is None:
        _, axes = plt.subplot(2)

    train_s = np.linspace(10, 100, num=5)
    axes[0].plot(train_sizes, test_scores, "o-", color=color, label=label)
    axes[1].plot(train_sizes, train_scores, "o-", color=color, label=label)

    print("Training Accuracy of ", label, ": ", train_scores[-1], "%")
    print("Testing Accuracy of ", label, ": ", test_scores[-1], "%")
    print("")
    return plt

def plot_per_class_accuracy(classifier, X, y, label, feature_selection = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=101)
    pipeline = Pipeline([("scalar", MinMaxScaler()), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)
    disp = plot_confusion_matrix(pipeline, X_test, y_test, cmap=plt.cm.Blues)
    plt.title(label)
    plt.savefig(f'cm-{label}.png')
    true_positive = disp.confusion_matrix[1][1]
    false_negative = disp.confusion_matrix[1][0]
    print(label + " - Sensitivity: ", true_positive/(true_positive+false_negative))
    print()

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

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    num_steps = 10
    classifier_labels = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=1), "red"),
        "Random Forest": (RandomForestClassifier(random_state=1), "green"),
        "SVM - Linear": (SVC(kernel="linear", random_state=1), "blue"),
        "SVM - RBF": (SVC(kernel="rbf", random_state=1), "yellow"),
        "SVM - Poly": (SVC(kernel="poly", random_state=1), "orange"),
        "kNN": (KNeighborsClassifier(n_neighbors=5), "purple"),
        "Guassian Naive Bayes": (GaussianNB(), "lime"),
    }

    for label in classifier_labels:
        classifier = classifier_labels[label][0]
        color = classifier_labels[label][1]
        plot_learning_curve(classifier, X, y, steps=num_steps, label=label, color=color, axes=axes)

    axes[0].set_xlabel('% of Training Exmples')
    axes[0].set_ylabel('Overall Classification Accuracy')
    axes[0].set_title('Model Evaluation - Cross-validation Accuracy')
    axes[0].legend()

    axes[1].set_xlabel('% of Training Exmples')
    axes[1].set_ylabel('Training/Recall Accuracy')
    axes[1].set_title('Model Evaluation - Training Accuracy')
    axes[1].legend()

    plt.savefig("comparison.png")

    for label in classifier_labels:
        classifier = classifier_labels[label][0]
        plot_per_class_accuracy(classifier, X, y, label)

if __name__ == '__main__':
    main()
