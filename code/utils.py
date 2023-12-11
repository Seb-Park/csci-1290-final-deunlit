import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg
from skimage.util import view_as_windows
import cv2

EPSILON = 1e-6  # to avoid log 0

rgb2gray_weightr = 0.2125
rgb2gray_weightg = 0.7154
rgb2gray_weightb = 0.0721


def find_luminance_chrominance(image):
    channel = len(image.shape)
    if channel != 3:  # if gray image, make it 3-channel
        image = np.stack((image,)*3, axis=-1)

    luminance = np.average(image, axis=2, weights=[rgb2gray_weightr,
                                                   rgb2gray_weightg,
                                                   rgb2gray_weightb])
    chrom_r, chrom_g, chrom_b = image[:, :, 0] / (luminance+EPSILON), \
        image[:, :, 1] / (luminance+EPSILON), \
        image[:, :, 2] / (luminance+EPSILON)
    chrominance = np.dstack([np.clip(chrom_r, 0, 255),
                             np.clip(chrom_g, 0, 255),
                             np.clip(chrom_b, 0, 255)]) if channel == 3 else np.clip(chrom_r, 0, 255)
    # plt.imshow(luminance, cmap="gray")
    # plt.show()
    return np.clip(luminance, 0, 255), np.clip(chrominance, 0, 255)

def gaussian_kernel(x, ai, psi, d):
    '''
    Multi-dimensional gaussian kernel according to equation 3
    '''
    normalization_factor = 1 / \
        ((2 * np.pi) ** (d / 2) * np.linalg.det(psi) ** 0.5)
    exponent = 1.0
    if d > 1:
        exponent = -0.5 * \
            np.dot(np.dot((x - ai).T, np.linalg.inv(psi)), (x - ai))
    else:
        exponent = -0.5 * np.dot((x - ai).T, (x - ai) / psi)

    return normalization_factor * np.exp(exponent)


def is_symmetric_and_positive_definite_sparse(A):
    """
    Check if a sparse matrix A is symmetric and positive definite.
    """
    if not sp.isspmatrix(A):
        raise ValueError("The matrix is not a sparse matrix")

    # Check if A is square
    if A.shape[0] != A.shape[1]:
        return False, "Matrix is not square"

    # Check symmetry: A should be equal to its transpose (in terms of non-zero pattern)
    if (A != A.T).nnz == 0:
        # Check positive definiteness using Cholesky decomposition
        try:
            # Convert sparse matrix to dense for Cholesky decomposition
            scipy.linalg.cholesky(A.toarray())
            return True, "Matrix is symmetric and positive definite"
        except scipy.linalg.LinAlgError:
            return False, "Matrix is symmetric but not positive definite"
    else:
        return False, "Matrix is not symmetric"

def get_pixel_neighborhood_data(i_x, i_y, radius, h, w, use_data, data=None):
    neighborhood_coords = []

    # Define the range for x and y based on the radius, handle boundaries using replication
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            # skip the center pixel itself
            if dx == 0 and dy == 0:
                continue
            nx = min(max(i_x + dx, 0), h - 1)
            ny = min(max(i_y + dy, 0), w - 1)

            if use_data:
                neighborhood_coords.append(data[nx, ny])
            else:
                neighborhood_coords.append((nx, ny))
    
    out = np.array(neighborhood_coords)
    return out

def average_variance_of_patches(image, patch_size=5):
    """
    Calculate the average variance of all image patches.

    Parameters:
    image (numpy.ndarray): The input image (should be a 2D grayscale image or a 2D single feature map).
    patch_size (int): The size of the patches (e.g., 3 for 3x3 patches).

    Returns:
    float: The average variance across all patches.
    """
    # Check if the image is a 2D array
    if len(image.shape) != 2:
        raise ValueError("The image must be a 2D array.")
    
    # Create patches using view_as_windows
    patches = view_as_windows(image, (patch_size, patch_size))
    
    # Initialize a list to hold variances of gradient magnitudes for each patch
    gradient_variances = []

    # Iterate over patches and calculate gradient magnitudes
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            gradient_magnitudes = calculate_gradient_magnitudes(patch)
            # Calculate the variance of gradient magnitudes for this patch
            patch_variance = np.var(gradient_magnitudes)
            gradient_variances.append(patch_variance)
    
    # Average variance across all patches
    average_variance = np.mean(gradient_variances)
    
    return average_variance

def calculate_gradient_magnitudes(patch):
    """
    Calculate the gradient magnitudes of an image patch.

    Parameters:
    patch (numpy.ndarray): The input patch (a 2D array).

    Returns:
    numpy.ndarray: An array of gradient magnitudes for the patch.
    """
    # Compute gradients along the x and y axes
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return grad_magnitude