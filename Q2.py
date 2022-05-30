import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from standardGD import gradient_descent, costF_log, sigmoid, percentage
from map_feature import map_feature
from plotDecisionBoundaryfunctions import polyFeatureVector, plotDecisionBoundary1

Xdata = pd.read_csv("email_data_2.csv")
data = Xdata.to_numpy()
X_orig = data[:, 0:2]

y_orig = data[:, 2]
y = y_orig.reshape([y_orig.shape[0], 1])
m = y.size
fig, axis = plt.subplots(1, 3, figsize=(18, 6))


# סעיף א
def drawRawData(X, y):
    x1 = X[:, 1]
    x2 = X[:, 2]
    axis[0].plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'ro', x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go')
    axis[0].set_title("Raw Data")
    axis[0].set_xlim(-1.0, 1.5)
    axis[0].set_ylim(-1.0, 1.5)
    axis[0].grid()


# סוף סעיף א

# סעיף ב
def standardReg(X, y, theta, alpha, iterations):
    """
    פונקציה זו מבצעת רגרסיה לוגיסטית ומייצגת ישר הרגרסיה אבל התפלגות הערכים אינה
    מתאימה לחלוקה בינארית ולכן רגרסיה סטנדרטית לא מתאימה לדוגמאות האימון הנתונות
    """
    y = y.reshape([y.shape[0], 1])
    theta, J_iter = gradient_descent(X, y, theta, alpha, iterations)
    ind = 1
    x1_min = 0.9 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = -(theta[0] + theta[1] * x1_min) / theta[2]
    x2_max = -(theta[0] + theta[1] * x1_max) / theta[2]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 1]
    x2 = X[:, 2]
    axis[1].plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'ro', x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go', x1lh, x2lh,
                 linewidth=2)
    axis[1].set_title('Raw Data \n+\n Decision Boundary Using Standard Gradient Descent')
    axis[1].set_xlim(-1.0, 1.5)
    axis[1].set_ylim(-1.0, 1.5)
    axis[1].grid()
    return theta, J_iter


# סוף סעיף ב #

X = np.c_[np.ones((np.shape(X_orig)[0], 1)), X_orig]

n = X.shape[1]
theta = np.zeros((n, 1))
alpha = 0.1
iterations = 10000

drawRawData(X, y)
standardReg(X, y, theta, alpha, iterations)


# סעיף ג


def cost_reg(X, y, theta, lamda):
    m = y.size
    J_theta, grad = costF_log(theta, X, y)
    J_theta_reg = J_theta + (lamda / 2 * m) * np.sum(theta ** 2)

    if np.shape(X)[1] >= 1:
        grad_reg = grad + (lamda / m) * theta
        return J_theta_reg, grad_reg
    return J_theta_reg, grad


def gd_reg(X, y, theta, alpha, iterations, lamba):
    J_iter = np.zeros(iterations, )
    for iter in range(iterations):
        J_iter[iter], grad = cost_reg(X, y, theta, lamba)
        theta = theta - alpha * grad
    return theta, J_iter


# סוף סעיף ג #


# סעיף ד
x1 = X_orig[:, 0]
x2 = X_orig[:, 1]
X = map_feature(x1, x2)
n = X.shape[1]
theta = np.zeros((n, 1))
lamda = 0
alpha = 0.1
iterations = 10000
theta, J_iter = gd_reg(X, y, theta, alpha, iterations, lamda)
plotDecisionBoundary1(theta, X, y, 6)
# סוף סעיף ד #


# סעיף ה
# התנסות בערכי לאמדה שונים
# lamda = 200
# lamda = 150
# lamda = 100
# lamda = 0.1
# כל עוד הלאמדה תגדל אז שטח ההחלטה יגדל
lamda = 0.1
alpha = 0.5
iterations = 190000
theta, J_iter = gd_reg(X, y, theta, alpha, iterations, lamda)

plotDecisionBoundary1(theta, X, y, 6)


# סוף סעיף ה #


# סעיף ו
def predictRegularized(X, theta):
    n = X.shape[0]
    Z = np.dot(X, theta)
    h_theta = sigmoid(Z)
    Y = np.where(h_theta > 0.5, 1, 0)
    return Y


def differenceList(y1, y2):
    m = len(y1)
    res = np.empty(np.shape(y1), dtype=bool)
    for i in range(m):
        if y1[i] == y2[i]:
            res[i] = True
        else:
            res[i] = False
    return res


data_table = pd.read_csv("email_data_3_2021.csv")
data = data_table.to_numpy()
X_test = data[:, 0:2]
x1 = X_test[:, 0]
x2 = X_test[:, 1]
X = map_feature(x1, x2)
y_test = data[:, 2]
y_test = y_test.reshape([y_test.shape[0], 1])
m = len(y_test)

y_pred_lin = predictRegularized(X, theta)
diff = differenceList(y_pred_lin, y_test)
percent = percentage(np.count_nonzero(diff), m)
print("Regularized Model Accuracy: {}".format(percent))
# סוף סעיף ו #


plt.subplots_adjust(left=0.03, right=0.964, top=0.788)
plt.show()
