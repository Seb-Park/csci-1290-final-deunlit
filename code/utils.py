import numpy as np


def gaussian_kernel(x, ai, psi):
    '''
    Multi-dimensional gaussian kernel according to equation 3
    '''
    d = x.shape[0]  # Dimensionality of the data
    normalization_factor = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(psi) ** 0.5)
    exponent = -0.5 * np.dot(np.dot((x - ai).T, np.linalg.inv(psi)), (x - ai))
    
    return normalization_factor * np.exp(exponent)