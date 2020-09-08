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
        tmp = x
        tmp = tmp ** i
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
