import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from standardGD import gradient_descent, sigmoid, percentage


def draw_boundary_lin(X, y, theta):
    ind = 1
    x1_min = 0.9 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = -(theta[0] + theta[1] * x1_min) / theta[2]
    x2_max = -(theta[0] + theta[1] * x1_max) / theta[2]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])

    plt.grid()
    plt.plot(x1lh, x2lh)


# סעיף א
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # we only take the first two features.
y = iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(2, figsize=(8, 6))

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

X1 = iris.data[0:30, 2:4]
X2 = iris.data[50:80, 2:4]
X3 = iris.data[100:130, 2:4]

# Plot the training points
plt.scatter(X1[:, 0], X1[:, 1], cmap=plt.cm.Set3,
            edgecolor='k', color='red')
plt.scatter(X2[:, 0], X2[:, 1], cmap=plt.cm.Set3,
            edgecolor='k', color='green')
plt.scatter(X3[:, 0], X3[:, 1], cmap=plt.cm.Set3,
            edgecolor='k', color='blue')

# סוף סעיף א


plt.xlabel('x1')
plt.ylabel('x2')
plt.title('data')
plt.xlim(0, 8)
plt.ylim(0, 3)

# סעיף ב,ג
X_iris = np.concatenate((X1, X2, X3))

Y_zeros_ones = np.concatenate((y[:30], y[50:80]))
Y_all = np.concatenate((Y_zeros_ones, y[100:130]))

y_setosa = Y_all
y_setosa[30:] = 0
y_setosa[:30] = 1
y_setosa = y_setosa.reshape((y_setosa.shape[0], 1))

X = np.c_[np.ones((np.shape(X_iris)[0], 1)), X_iris]
n = X.shape[1]
theta = np.zeros((n, 1))

alpha = 0.05
iterations = 30000

theta_setosa, J_Iter = gradient_descent(X, y_setosa, theta, alpha, iterations)

draw_boundary_lin(X, y_setosa, theta_setosa)

theta = np.zeros((n, 1))
y_versicolor = Y_all
y_versicolor[:] = 0
y_versicolor[30:60] = 1
y_versicolor = y_versicolor.reshape((y_versicolor.shape[0], 1))
theta_versicolor, J_Iter = gradient_descent(X, y_versicolor, theta, alpha, iterations)

draw_boundary_lin(X, y_versicolor, theta_versicolor)

theta = np.zeros((n, 1))
y_virginica = Y_all
y_virginica[:] = 0
y_virginica[60:90] = 1
y_virginica = y_virginica.reshape((y_virginica.shape[0], 1))
theta_virginica, J_Iter = gradient_descent(X, y_virginica, theta, alpha, iterations)

draw_boundary_lin(X, y_virginica, theta_virginica)

plt.xlim(0, 8)
plt.ylim(0, 3)

plt.show()


# סוף סעיף ב,ג

# סעיף ד
def differenceList(y1, y2):
    m = len(y1)
    res = np.empty(np.shape(y1), dtype=bool)
    for i in range(m):
        if y1[i] == y2[i]:
            res[i] = True
        else:
            res[i] = False
    return res


def predict(X, y, theta_setosa, theta_versicolor, theta_virginica):
    m = X.shape[0]
    X = np.c_[np.ones((m, 1)), X]

    Z_setosa = np.dot(X, theta_setosa)
    Z_versicolor = np.dot(X, theta_versicolor)
    Z_virginica = np.dot(X, theta_virginica)

    h_theta_setosa = sigmoid(Z_setosa)
    h_theta_versicolor = sigmoid(Z_versicolor)
    h_theta_virginica = sigmoid(Z_virginica)
    h_theta_all = np.concatenate((h_theta_setosa, h_theta_versicolor, h_theta_virginica), axis=1)
    prediction = np.argmax(h_theta_all, axis=1)
    diff = differenceList(prediction, y)
    rate = percentage(np.count_nonzero(diff), m)
    return prediction, rate


X1 = iris.data[31:50, 2:4]
X2 = iris.data[81:100, 2:4]
X3 = iris.data[131:150, 2:4]
y1 = y[31:50]
y2 = y[81:100]
y3 = y[131:150]

prediction_x1, rate_x1 = predict(X1, y1, theta_setosa, theta_versicolor, theta_virginica)
prediction_x2, rate_x2 = predict(X2, y2, theta_setosa, theta_versicolor, theta_virginica)
prediction_x3, rate_x3 = predict(X3, y3, theta_setosa, theta_versicolor, theta_virginica)

print("Setosa:")
print("   Prediction:", prediction_x1)
print("   Rate:", rate_x1)
print("Versicolor:")
print("   Prediction:", prediction_x2)
print("   Rate:", rate_x2)
print("Virginica:")
print("   Prediction:", prediction_x3)
print("   Rate:", rate_x3)
# סוף סעיף ד


# סעיף ה
X_iris = iris.data
y = iris.target

y_setosa = y
y_setosa[50:] = 0
y_setosa[:50] = 1
y_setosa = y_setosa.reshape((y_setosa.shape[0], 1))

X = np.c_[np.ones((np.shape(X_iris)[0], 1)), X_iris]
n = X.shape[1]
theta = np.zeros((n, 1))
alpha = 0.005
iterations = 20000

theta_setosa, J_Iter = gradient_descent(X, y_setosa, theta, alpha, iterations)

theta = np.zeros((n, 1))
y_versicolor = y
y_versicolor[:] = 0
y_versicolor[51:100] = 1
y_versicolor = y_versicolor.reshape((y_versicolor.shape[0], 1))
theta_versicolor, J_Iter = gradient_descent(X, y_versicolor, theta, alpha, iterations)

theta = np.zeros((n, 1))
y_virginica = y
y_virginica[:] = 0
y_virginica[101:150] = 1
y_virginica = y_virginica.reshape((y_virginica.shape[0], 1))
theta_virginica, J_Iter = gradient_descent(X, y_virginica, theta, alpha, iterations)

X1 = iris.data[31:50]
X2 = iris.data[81:100]
X3 = iris.data[131:150]
y1 = y[31:50]
y2 = y[81:100]
y3 = y[131:150]

prediction_x1, rate_x1 = predict(X1, y1, theta_setosa, theta_versicolor, theta_virginica)
prediction_x2, rate_x2 = predict(X2, y2, theta_setosa, theta_versicolor, theta_virginica)
prediction_x3, rate_x3 = predict(X3, y3, theta_setosa, theta_versicolor, theta_virginica)

print("Setosa:")
print("   Prediction:", prediction_x1)
print("   Rate:", rate_x1)
print("Versicolor:")
print("   Prediction:", prediction_x2)
print("   Rate:", rate_x2)
print("Virginica:")
print("   Prediction:", prediction_x3)
print("   Rate:", rate_x3)
# סוף סעיף ה
