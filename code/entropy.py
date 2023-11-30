import numpy as np
from scipy.optimize import least_squares

import numpy as np

def calculate_pixel_differences(image):
    # Assuming image is grayscale
    image = image.astype(np.float32)

    # Initialize arrays to store differences
    # log_illumination_diffs = np.zeros_like(image)
    mean_log_intensity_diffs = np.zeros_like(image)
    log_image = np.log(image + 1)  # avoid log(0)

    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            neighborhood = log_image[i-1:i+2, j-1:j+2]
            diff = np.abs(neighborhood - log_image[i, j])
            # log_illumination_diffs[i, j] = np.sum(diff) - diff[1, 1]  # Exclude the center pixel
            mean_log_intensity_diffs[i, j] = np.abs(log_image[i, j] - np.mean(neighborhood))

    return mean_log_intensity_diffs


def energy_function(l, image, lambda_reg):
    """
    Define the energy function to be minimized.
    :param l: Current illumination estimate.
    :param image: Input image.
    :param lambda_reg: Regularization parameter.
    :return: Energy value for the current illumination estimate.
    """
    u = calculate_weights_u(image)
    v = calculate_weights_v(image)
    eta_bar = calculate_pixel_differences(image)

    # First term: smoothness of illumination field
    smoothness_term = np.sum(u * l**2)

    # Second term: texture term
    texture_term = lambda_reg * np.sum(v * (l - eta_bar)**2)

    return smoothness_term + texture_term

def minimize_energy(image, initial_l, lambda_reg):
    """
    Minimize the energy function using iterative optimization.
    :param image: Input image.
    :param initial_l: Initial estimate of illumination.
    :param lambda_reg: Regularization parameter.
    :return: Optimal illumination estimate.
    """
    result = least_squares(energy_function, initial_l, args=(image, lambda_reg))
    return result.x


def calculate_weights_u(image):
    pass

def calculate_weights_v(image):
    pass
