import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from standardGD import gradient_descent, sigmoid, percentage

fig, axes = plt.subplots(1, 4, figsize=(18, 6))

axes[0].set_xlim(0, 2.5)
axes[0].set_ylim(0, 2.5)

axes[1].set_xlim(0, 2.5)
axes[1].set_ylim(-0.5, 2.5)

axes[2].set_xlim(0, 2.5)
axes[2].set_ylim(0, 2.5)


def draw_raw_data(X, y):
    x1 = X[:, 0]
    x2 = X[:, 1]

    axes[0].plot(x1[y == 0], x2[y == 0], 'go', x1[y == 1], x2[y == 1], 'rD')
    axes[0].set_title('Raw Data')


def draw_boundary_lin(X, y, theta):
    ind = 1
    x1_min = 0.9 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = -(theta[0] + theta[1] * x1_min) / theta[2]
    x2_max = -(theta[0] + theta[1] * x1_max) / theta[2]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 0]
    x2 = X[:, 1]

    axes[1].plot(x1[y == 0], x2[y == 0], 'go', x1[y == 1], x2[y == 1], 'rD', x1lh, x2lh, 'b',
                 linewidth=3)
    axes[1].set_title('Raw Data \n+\n Linear Decision Boundary')
    axes[1].grid()


def draw_boundary_quad(X, y, theta):
    ind = 1
    x1_min = 0.9 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()

    x1lh = np.linspace(x1_min, x1_max, X.shape[0])
    x2lh = (-(theta[0] + theta[1] * x1lh + theta[3] * (x1lh ** 2))) / theta[2]
    x1 = X[:, 0]
    x2 = X[:, 1]

    axes[2].plot(x1[y == 0], x2[y == 0], 'go', x1[y == 1], x2[y == 1], 'rD', x1lh, x2lh, 'b', linewidth=3)
    axes[2].set_title('Raw Data \n+\n Quadratic Decision Boundary')
    axes[2].grid()


def drawCostFuncions(X, y, theta):
    fig2, axes2 = plt.subplots(2, 2, figsize=(6, 6))
    alpha = 0.01
    iterations = 100000
    theta1, J_iter = gradient_descent(X, y, theta, alpha, iterations)
    axes2[0, 0].plot(J_iter)
    axes2[0, 0].set_title('alpha: {}, iterations: {}'.format(alpha, iterations))

    alpha = 0.02
    iterations = 100000
    theta2, J_iter = gradient_descent(X, y, theta, alpha, iterations)
    axes2[0, 1].plot(J_iter)
    axes2[0, 1].set_title('alpha: {}, iterations: {}'.format(alpha, iterations))

    alpha = 0.03
    iterations = 100000
    theta3, J_iter = gradient_descent(X, y, theta, alpha, iterations)
    axes2[1, 0].plot(J_iter)
    axes2[1, 0].set_title('alpha: {}, iterations: {}'.format(alpha, iterations))

    alpha = 0.03
    iterations = 300000
    theta, J_iter = gradient_descent(X, y, theta, alpha, iterations)
    axes2[1, 1].plot(J_iter)
    axes2[1, 1].set_title('alpha: {}, iterations: {}'.format(alpha, iterations))


def predictLinear(X, theta):
    x0 = X[:, 0]
    x1 = X[:, 1]
    Z = theta[0] * x0 + theta[1] * x1
    h_theta = sigmoid(Z)
    Y = np.where(h_theta > 0.5, 1, 0)
    return Y


def predictQuadratic(X, theta):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_sq = x1.reshape((x1.shape[0], 1)) ** 2
    X = np.hstack((X, x1_sq))
    x3 = X[:, 2]
    Z = theta[0] + theta[1] * x1 + theta[2] * x2 + theta[3] * x3
    h_theta = sigmoid(Z)
    Y = np.where(h_theta > 0.5, 1, 0)
    return Y


def main():
    data_table = pd.read_csv("email_data_1.csv")
    data = data_table.to_numpy()
    X_orig = data[:, 0:2]
    y_orig = data[:, 2]
    draw_raw_data(X_orig, y_orig)
    X = np.c_[np.ones((np.shape(X_orig)[0], 1)), X_orig]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x1 = x1.reshape((x1.shape[0], 1))
    X = np.hstack((X, x1 ** 2))
    n = X.shape[1]
    theta = np.zeros((n, 1))
    y = y_orig.reshape([y_orig.shape[0], 1])

    # alpha = 0.01 and iterations = 100000 - almost no convergence
    # alpha = 0.02 and iterations = 100000 - Quadratic Decision Boundary still not good enough
    # alpha = 0.03 and iterations = 100000 - good enough but lets try to make it better
    # alpha = 0.03 and iterations = 300000 - Great, but we'll try to increase iterations even more
    # alpha = 0.03 and iterations = 700000 - No Difference
    # alpha = 0.02 and iterations = 700000 - Also No Difference
    # The best convergence is for alpha = 0.03 and iterations = 300000
    drawCostFuncions(X, y, theta)
    alpha = 0.03
    iterations = 300000
    theta, J_iter = gradient_descent(X, y, theta, alpha, iterations)

    axes[3].plot(J_iter)
    axes[3].set_title('Cost Function\nWith\n alpha: {}, iterations: {}'.format(alpha, iterations))

    draw_boundary_lin(X_orig, y_orig, theta)
    draw_boundary_quad(X_orig, y_orig, theta)
    plt.subplots_adjust(left=0.03, right=0.964, top=0.788)
    plt.show()

    data_table = pd.read_csv("email_data_test_2021.csv")
    data = data_table.to_numpy()
    X_test = data[:, 0:2]
    y_test = data[:, 2]
    m = len(y_test)

    y_pred_lin = predictLinear(X_test, theta)
    diff = (y_pred_lin == y_test)
    linear_percent = percentage(np.count_nonzero(diff), m)
    print("Linear Model Accuracy: {}".format(linear_percent))

    y_pred_quad = predictQuadratic(X_test, theta)
    diff = (y_pred_quad == y_test)
    quadratic_percent = percentage(np.count_nonzero(diff), m)
    print("Quadratic Model Accuracy: {}".format(quadratic_percent))


if __name__ == '__main__':
    main()
