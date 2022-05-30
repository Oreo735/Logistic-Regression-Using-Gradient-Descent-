import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def costF_log(theta, X, y):
    m = len(y)

    Z = np.dot(X, theta)
    h_theta = sigmoid(Z)
    cost_left = np.dot(y.T, np.log(h_theta))
    cost_right = np.dot((1 - y).T, np.log(1 - h_theta))
    J_theta = -(1 / m) * (cost_left + cost_right)
    grad_J = (1 / m) * np.dot(X.T, (h_theta - y))
    return J_theta, grad_J


def gradient_descent(X, y, theta, alpha, iterations):
    J_iter = np.zeros(iterations, )
    for iter in range(iterations):
        J_iter[iter], grad = costF_log(theta, X, y)
        theta = theta - alpha * grad
    return theta, J_iter


def percentage(part, whole):
    percent = 100 * float(part) / float(whole)
    return str(percent) + "%"
