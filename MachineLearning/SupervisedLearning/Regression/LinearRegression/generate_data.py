# Jack C. Cook
# Wednesday, September 2, 2020

# any functions for generating data

import numpy as np


def poly_third_order(x: np.ndarray) -> np.ndarray:
    """
    A third order polynomial to generate the labels (outputs, targets) for the input x
    :type x: np.ndarray
    :param x: the input feature vector x
    :return: the output (target) vector given input feature vector x
    """
    y = x**3 - 6*x**2 + 4*x + 12
    return y
