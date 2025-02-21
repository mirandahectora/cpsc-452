"""
Deep Learning Theory and Applications, Problem Set 1
"""
# helpful libraries
import numpy as np
from sklearn.neighbors import (
    NearestNeighbors, # used to identify nearest neighbors, but not to classify
)  


def problem2_evaluate_function_on_random_noise(N, sigma):
    """Sample N points uniformly from the interval [-1,3],
    and output the function y = x^2 - 3x + 1 with random noise added to the outputs
    Hint: You can sort x before evaluate the function. This could help plot
    smooth polynomial lines later on

    Parameters
    ----------
    N : int
        The number of points
    sigma : float
        The standard deviation of noise to add to the randomly generated points.

    Returns
    -------
    x, y (list, list)
        x, the randomly generated points
        y, the function evaluated at these points, with added noise
    """

    return x, y


def problem2_fit_polynomial(x, y, degree, regularization=None):
    """Returns optimal coefficients for a polynomial of the given degree
    to fit the data, using the Moore-Penrose Pseudoinverse (specified in the assignment)
    Note: this function only needs to function for degrees 1,2, and 9 --
    but you are welcome build something that works for any degree.
    By incorporating the value of the regularization parameter, this function should work
    for both 2.2 and 2.3

    Parameters
    ----------
    x : list of floats
        The input x values
    y : list of floats
        The input y values
    degree : int
        The degree of the polynomial to fit
    regularization : float
        The parameter lambda which specifies the degree of regularization to apply. Default 0.

    Returns
    -------
    list of floats
        The coefficients of the polynomial.
    """

    return coeffs


def problem3_knn_classifier(train_data, train_labels, test_data, k):
    """A kth Nearest Neighbor classified. Accepts points and training labels,
    and returns predicted labels for each point in the dataset.

    Parameters
    ----------
    train_data : ndarray
        The training points, in an n x d array, where n is the number of points and d is the dimension.
    train_labels : list of classes
        The training labels. They should correspond directly to the points in the training data array.
    test_data : ndarray
        The unlabelled data, to be labelled by the classifier
    k : positive int
        The number of nearest neighbors to consult.

    Returns
    -------
    predicted_labels : list
        The labels outputted by the classifier for each of the test datapoints.
    """

    return predicted_labels
