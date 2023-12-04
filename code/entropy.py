import numpy as np
from scipy.optimize import least_squares
from utils import gaussian_kernel
import matplotlib.pyplot as plt

EPSILON = 1e-6  # to avoid log 0


def energy_function(curr_L, image, lambda_reg, phi_l, phi_p, omega_t):
    """
    Define the energy function to be minimized according to equation 16.
    :param curr_L: Current illumination estimate, (h, w)
    :param image: Input image, (h, w)
    :param lambda_reg: Regularization parameter, (1,)
    :param phi_l, phi_p, omega_t: smoothing matrices to calculate weights u and v
    :return: Energy value for the current illumination estimate.
    """
    h, w = image.shape[0], image.shape[1]
    # u and v should be sparse matrix of (h*w, h*w)
    u = calculate_weights_u(image, phi_l, phi_p)
    v = calculate_weights_v(image, omega_t)
    eta_bar = mean_neighbor_log_intensity_differences(image)

    L_star = np.zeros((h, w))

    # using r and c for row and col to avoid confusing with pixel i and neighbor j
    for r in range(h):
        for c in range(w):
            # Handle boundaries
            top = max(r-1, 0)
            bottom = min(r+1, h-1)
            left = max(c-1, 0)
            right = min(c+1, w-1)

            # log illumination of current pixel
            log_L_i = np.log(curr_L[r, c] + EPSILON)
            l_ijs = [
                np.abs(log_L_i - np.log(curr_L[top, c] + EPSILON)),
                np.abs(log_L_i - np.log(curr_L[bottom, c] + EPSILON)),
                np.abs(log_L_i - np.log(curr_L[r, left] + EPSILON)),
                np.abs(log_L_i - np.log(curr_L[r, right] + EPSILON))
            ]
            u_ijs = [
                u[r*c, top*c],
                u[r*c, bottom*c],
                u[r*c, r*left],
                u[r*c, r*right],
            ]
            v_ijs = [
                v[r*c, top*c],
                v[r*c, bottom*c],
                v[r*c, r*left],
                v[r*c, r*right],
            ]

            smoothness_term = u_ijs * l_ijs**2
            texture_term = v_ijs * (l_ijs - eta_bar[r, c])**2

            L_star[r, c] = smoothness_term + lambda_reg * texture_term

    return L_star


def minimize_energy(image, initial_l, lambda_reg):
    """
    Minimize the energy function using iterative optimization.
    :param image: Input image.
    :param initial_l: Initial estimate of illumination.
    :param lambda_reg: Regularization parameter.
    :return: Optimal illumination estimate.
    """
    pass


def mean_neighbor_log_intensity_differences(image):
    '''
    For each pixel in the image,
    calculates the mean of log intensity differences between the pixel and its neighbors.
    Should be returning a matrix of shape (height, width). (Same as image)
    Assuming image is grayscale
    '''
    h, w = image.shape[0], image.shape[1]
    mean_log_intensity_diffs = np.zeros_like(image)
    log_image = np.log(image + EPSILON)  # avoid log(0)

    for r in range(h):
        for c in range(w):
            # neighborhood with boundary considerations
            top = max(r-1, 0)
            bottom = min(r+1, h-1)
            left = max(c-1, 0)
            right = min(c+1, w-1)
            neighborhood = [
                log_image[top, c],
                log_image[bottom, c],
                log_image[r, left],
                log_image[r, right]
            ]
            mean_log_intensity_diffs[r, c] = np.abs(
                log_image[r, c] - np.mean(neighborhood))

    return mean_log_intensity_diffs


def calculate_weights_u(image, phi_l, phi_p):
    pass


def calculate_weights_v(image, omega_t):
    pass


rgb2gray_weightr = 0.2125
rgb2gray_weightg = 0.7154
rgb2gray_weightb = 0.0721


def find_luminance_chrominance(image):
    luminance = np.average(image, axis=2, weights=[rgb2gray_weightr,
                                                   rgb2gray_weightg,
                                                   rgb2gray_weightb])
    chrom_r, chrom_g, chrom_b = image[:, :, 0] / luminance, \
        image[:, :, 1] / luminance, \
        image[:, :, 2] / luminance
    chrominance = np.dstack([np.clip(chrom_r, 0, 255),
                             np.clip(chrom_g, 0, 255),
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
