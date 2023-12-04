import numpy as np
from scipy.optimize import least_squares
from utils import gaussian_kernel

import matplotlib.pyplot as plt

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

rgb2gray_weightr = 0.2125 
rgb2gray_weightg = 0.7154
rgb2gray_weightb = 0.0721

def find_luminance_chrominance(image):
    luminance = np.average(image, axis=2, weights=[rgb2gray_weightr, \
                                                   rgb2gray_weightg, \
                                                   rgb2gray_weightb])
    chrom_r, chrom_g, chrom_b = image[:, :, 0] / luminance, \
                                image[:, :, 1] / luminance, \
                                image[:, :, 2] / luminance
    chrominance = np.dstack([np.clip(chrom_r, 0, 255), \
                             np.clip(chrom_g, 0, 255), \
                             np.clip(chrom_b, 0, 255)])
    plt.imshow(luminance, cmap="gray")
    # # plt.imshow(chrominance)
    plt.show()
    return np.clip(luminance, 0, 255), np.clip(chrominance, 0, 255)

def calculate_illumination(image):
    # SEPARATE OUT LUMINANCE AND CHROMINANCE
    # REALLY CLOSE TO ILLUMINATION AND REFLECTANCE
    # INTRINSIC IMAGE DECOMP
    pass