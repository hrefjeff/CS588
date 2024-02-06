#!/usr/bin/env python3

# Linear regression using least square regression
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Our "hand made" linear regression
# Big idea is that we're calculating the table
# on Week4b.pdf slide #19
def lin_reg(x, y):
    # number of observatiions/points
    n = np.size(x)

    # mean of x and y vector
    # calculating the parameter calculation table
    # Summation of x, then divide by n
    # Summation of y, then divide by n
    # (Slides: Week4b.pdf pg 16)
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n * m_y * m_x
    SS_xx = np.sum(x*x) - n * m_x * m_x

    # calculating regression coefficients
    beta = SS_xy / SS_xx 
    alpha = m_y - beta*m_x # this is the slope (rate of change)

    return (alpha, beta)

def plot_lin_reg_model(x, y, a, b, filename):
    # plotting the actual points as a scatter plot
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    # predicted response vector
    # y_pred = alpha + beta * x
    y_pred = a + (b * x)

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show as plot
    plt.savefig(filename)

def main():
    # observations
    x = np.array([4, 6, 10, 12])       # Pesticides (In DV)
    y = np.array([3.0, 5.5, 6.5, 9.0]) # Crop Yield (Dep V)

    # estimate coefficiaents
    a, b = lin_reg(x,y)
    print("Estimated coefficients:\n alpha (slope intercept) = {}    \n beta (slope) = {}".format(a, b))

    # plotting regression line
    plot_lin_reg_model(x, y, a, b, "handmade_lin_reg.png")

    # Compare with sklearn
    X = np.array([4, 6, 10, 12])       # Pesticides (In DV)
    Y = np.array([3.0, 5.5, 6.5, 9.0]) # Crop Yield (Dep V)
    XX = np.reshape(X, (-1, 1))
    reg = LinearRegression().fit(XX, Y)

    print("Estimated coefficients:\n alpha (slope intercept) = {}    \n beta (slope) = {}".format(reg. intercept_, reg.coef_))

    # THE REAL POWER OF LINEAR REGRESSION
    # Predictive analytics
    new_x = 20 # try values like 3, 20, or 120
    new_y = reg.predict(np.reshape(new_x, (-1, 1)))
    print(f'For new x = {new_x} the estimated new y prediction = {new_y}')

    plot_lin_reg_model(np.append(x, new_x), np.append(y, new_y), reg.intercept_, reg.coef_, "new_lin_reg.png")

if __name__ == '__main__':
    main()
