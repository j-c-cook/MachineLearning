# Jack C. Cook
# Sunday, August 30, 2020

import numpy as np


def process(path):
    """
    This function takes a path to a file and does stuff. [For documentation example.]

    :type path: string
    :param path: a path to the file which contains the features
    :return: None
    """
    print(path)
    return None


def least_squares_max_likelihood(phi_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the maximum likelihood for least squares
    :param phi_x: the input feature transformed
    :param y: the y values for regression
    :return: the maximum likelihood solution to least squares linear regression
    """
    # TODO: look into sklearn linear regression
    # Take the Moore-Penrose pseudo inverse
    # https://pythonhosted.org/algopy/examples/moore_penrose_pseudoinverse.html
    x_p = np.linalg.pinv(phi_x)
    # finish solving for maximum likelihood
    theta_ml = np.matmul(x_p, y)
    return theta_ml


def phi_polynomial(x: np.ndarray, order=1) -> np.ndarray:
    """
    A hypothesis or basis function which is a polynomial function of x up to order

    :type x: np.ndarray
    :type order: int

    :param x: the original input features used to generate the data
    :param order: the order of the basis function

    :return: the input feature vector after being processed by the basis function
    """
    # TODO: look into np.power()
    phi_x = np.ones_like(x)  # first term is one (bias)
    for i in range(1, order + 1):
        tmp = np.power(x, i)
        phi_x = np.hstack((phi_x, tmp))
    return phi_x


def predicted_values(theta, x):
    # TODO: remove the for loop by using phi_x
    m, n = theta.shape
    y_pred = np.zeros_like(x)
    for i in range(m):
        power = i
        y_pred += theta[i][0] * x ** power
    return y_pred


def predict_values(phi_x: np.ndarray, theta_ml: np.ndarray):
    """
    Predict the values of the maximum likely hood solution
    :param phi_x: the output of the feature vector after processing through the basis function (see phi_polynomial)
    :param theta_ml: the maximum likelihood solution for linear regression using least squares
    (see least_squares_maximum_likelihood)
    :return: the predicted values
    """
    return np.matmul(phi_x, theta_ml)


def average_least_square_error(y1: np.ndarray, y2: np.ndarray):
    """
    Compute the average or mean least squares error
    :param y1: one input vector
    :param y2: one input vector
    :return: the scalar value of the average least squares error
    """
    # https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
    mse = (np.square(y2 - y1)).mean()
    return mse


def weight_matrix(x: np.ndarray, training_data: np.ndarray, tao: float):
    """

    :param x: the input value to be locally weighted to
    :param training_data: the training data or examples you have available to train with
    :param tao: the bandwidth parameter
    :return: an identity matrix which is the weight matrix

    >>> w[0] = np.exp(- (training_data[0] - x)**2 / (2 * tao ** 2) )

    """
    # https://www.geeksforgeeks.org/implementation-of-locally-weighted-linear-regression/
    # M is the No of training examples
    M = training_data.shape[0]
    # Initialising W with identity matrix
    W = np.mat(np.eye(M))
    # calculating weights for query points
    # TODO: turn this into vectorized code
    for i in range(M):
        xi = training_data[i]
        denominator = (-2 * tao * tao)
        W[i, i] = np.exp(np.dot((xi - x), (xi - x).T) / denominator)
    return W


def predict(training_data: np.ndarray, Y: np.ndarray, x: np.ndarray, tao: float):
    """

    :param training_data: the feature vector
    :param Y: the output values or target values
    :param x: the query point
    :param tao: bandwidth
    :return: the prediction and theta maximum likelihood
    """
    M = training_data.shape[0]
    all_ones = np.ones((M, 1))
    # X_ = np.hstack((training_data, all_ones))
    X_ = training_data
    # a = np.ones((1, 1))
    # qx = np.mat([x, 1])
    # x = x[:, np.newaxis]
    # qx = np.hstack((x, a))
    qx = x
    W = weight_matrix(qx, X_, tao)
    # calculating parameter theta
    theta = np.linalg.pinv(X_.T*(W * X_))*(X_.T*(W * Y))
    # calculating predictions
    pred = np.dot(qx, theta)
    return theta, pred
