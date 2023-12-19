import numpy as np
from utils import gaussian_kernel, is_symmetric_and_positive_definite_sparse, get_pixel_neighborhood_data, find_luminance_chrominance
import matplotlib.pyplot as plt
import cv2
from scipy import sparse
from tqdm import tqdm
from skimage.filters import gaussian

R = 0 # R=0 -> direct neighbor mode
N = (2*R+1)**2 - 1 if R != 0 else 4
EPSILON = 1e-6  # to avoid log 0


def minimize_energy(image, mask, initial_l, phi_l, phi_p, omega_t, omega_p, lambda_reg=1.0, num_iter=20, maxiter=10000, tol=1e-5, img_name='test_1167_matrix', chrominance=None):
    """
    Minimize the energy function using iterative optimization.
    :return: Optimal illumination estimate (IN LOG DOMAIN!!!!!!!!!!)
    """

    h, w = image.shape[0], image.shape[1]
    num_pixels = h * w

    curr_l = initial_l # ???
    float_img = image.astype(np.float64) / 255
    curr_image = image.astype(np.float64) # linear, [0,255], float64
    mask = mask.reshape((num_pixels, 1))
    for iteration in range(num_iter):
        
        print(f'------------Iteration: {iteration}-------------')
        
        print("Image Type: ", (curr_image).dtype)

        u = calculate_weights_u(curr_image, phi_l, phi_p)
        v = calculate_weights_v(curr_image, omega_t, omega_p)
        print(f"u type: {u.dtype}, v type: {v.dtype}")
        eta_bar = mean_neighbor_log_intensity_differences(
            curr_image).reshape((num_pixels, 1))
        uv_sum = u + (lambda_reg * v)
        A = sparse.diags(np.array(uv_sum.sum(axis=1)).flatten())
        
        b = uv_sum * curr_l + lambda_reg * \
            (sparse.diags(np.array(v.sum(axis=1)).flatten()) - v) * eta_bar

        print(f'eta_bar.shape={eta_bar.shape}')
        print(f'A.shape={A.shape}')
        print(f'b.shape={b.shape}')
        b[mask > 5] = 0

        prev_r = np.log(image + EPSILON).astype(np.float64) - \
            curr_l.reshape(h,w).astype(np.int64)

        # print(f'Image type: {(image + EPSILON).dtype}')
        print(f'Image int max: {np.max((image + EPSILON).astype(np.int64))}')
        print(f'Image int min: {np.min((image + EPSILON).astype(np.int64))}')
        print(f'Image int log max: {np.max(np.log(image + EPSILON).astype(np.int64))}')
        print(f'Image int log min: {np.min(np.log(image + EPSILON).astype(np.int64))}')
        print(f'Image flt log max: {np.max(np.log(image + EPSILON).astype(np.float64))}')
        print(f'Image flt log min: {np.min(np.log(image + EPSILON).astype(np.float64))}')
        print(f'Image flt max: {np.max((image + EPSILON))}')
        print(f'Image flt min: {np.min((image + EPSILON))}')
        
        min_r = np.min(prev_r)
        if(min_r < 0):
            prev_r -= np.min(prev_r)

        print(f'Prev_r min: {np.min(prev_r)}, max: {np.max(prev_r)}')

        curr_l, exit_code = sparse.linalg.cg(A, b, x0=curr_l, maxiter=maxiter, tol=tol)
        curr_l = curr_l.reshape((num_pixels, 1))
        print(f"Min log l* : {np.min(curr_l)}, Max log l*: {np.max(curr_l)}")
        print(f"Min l* : {np.min(np.exp(curr_l))}, Max l*: {np.max(np.exp(curr_l))}")
        print(f'curr_l.shape={curr_l.shape}')
        if exit_code == 0:
            print('CG SOLVER SUCCESS')
        elif exit_code > 0:
            print('CG SOLVER DID NOT CONVERGE')
        else:
            print('CG SOLVER ILLEGAL INPUT')

        curr_image = np.exp(curr_l.reshape((h, w)) + prev_r)
        
        curr_image /= np.max(curr_image)

        if chrominance is None:
            cv2.imwrite(f'../results/{img_name}_R={R}_{iteration}.jpg', (curr_image * 255).astype(np.uint8))
        else:
            cv2.imwrite(f'../results/{img_name}_R={R}_{iteration}.jpg', (curr_image * 255).astype(np.uint8))
        # curr_image = apply_new_illumination(curr_image, curr_l.reshape((h,w)))

    return curr_l, curr_image


def mean_neighbor_log_intensity_differences(image):
    '''
    For each pixel in the image,
    calculates the mean of log intensity differences between the pixel and its neighbors.
    Should be returning a matrix of shape (height, width). (Same as image)
    Assuming image is grayscale
    '''
    h, w = image.shape[0], image.shape[1]
    mean_log_intensity_diffs = np.zeros_like(image).astype(np.float32)
    log_image = np.log(image + EPSILON)  # avoid log(0)

    for r in range(h):
        for c in range(w):
            neighborhood = get_pixel_neighborhood_data(
                r, c, R, h, w, use_data=True, data=log_image)
            mean_log_intensity_diffs[r, c] = np.abs(
                (log_image[r, c] - np.mean(neighborhood)))

    return mean_log_intensity_diffs


def calculate_weights_u(image, phi_l, phi_p):
    '''
    Computes a weight matrix for the inputted image the details information for 
    each pixel, i.e. information on how relevant each of its neighbors are to it
    based on distance from the pixel 

    returns weight matrix
    '''
    h, w = image.shape[0], image.shape[1]
    # necessary info for sparse matrix
    row = []
    col = []
    data = []

    illumination, _ = find_luminance_chrominance(image, EPSILON=EPSILON)

    # loop through each pixel
    num_pixels = image.shape[0] * image.shape[1]
    for i in tqdm(range(num_pixels), desc='Calculating u'):
        curr_row = i // image.shape[1]
        curr_col = i % image.shape[1]

        # Get four neighboring pixels TODO: why only four?
        # TODO: Also, should we include the pixel itself or does
        # that not get weighted?
        # neighbors = [(curr_row - 1, curr_col), (curr_row + 1, curr_col),
        #              (curr_row, curr_col - 1), (curr_row, curr_col + 1)]
        neighbors = get_pixel_neighborhood_data(
            curr_row, curr_col, R, h, w, use_data=False)
        for dir in neighbors:
            if dir[0] < 0 or dir[0] >= image.shape[0] or dir[1] < 0 or dir[1] >= image.shape[1]:
                continue

            # The calculated weight for this value is to be placed in the row
            # corresponding to pixel i and the column corresponding to this
            # neighboring pixel
            row.append(i)
            col.append(dir[0] * image.shape[1] + dir[1])

            # Weights the color difference between two pixels
            illum_gaussian = gaussian_kernel(
                np.array([illumination[dir[0]][dir[1]]]),
                np.array([illumination[curr_row][curr_col]]),
                2*phi_l, 1)

            # Weights the distance between two pixels
            pixel_gaussian = gaussian_kernel(
                np.array([dir[0], dir[1]]),
                np.array([curr_row, curr_col]), 2*phi_p, 2)
            # TODO: I'm confused--aren't these the same for all for pixels?
            # as in, the four neighboring pixels will all be exactly the
            # same distance away from the center pixel, right?

            data.append(illum_gaussian * pixel_gaussian)

    u = sparse.csr_matrix((np.asarray(data)[:, 0], (np.asarray(
        row), np.asarray(col))), shape=(num_pixels, num_pixels))
    print(f'u.shape={u.shape}')
    return u


def calculate_weights_v(image, omega_t, omega_p):
    '''
    Computes a weight matrix that details every pixel's relevance to every other
    pixel based on 
    '''
    h, w = image.shape[0], image.shape[1]
    # necessary info for sparse matrix
    row = []
    col = []
    data = []
    log_image = np.log(image + EPSILON)
    _, reflectance = find_luminance_chrominance(image, EPSILON=EPSILON)

    # loop through each pixel
    num_pixels = image.shape[0] * image.shape[1]
    for i in tqdm(range(num_pixels), desc='Calculating v'):
        curr_row = i // image.shape[1]
        curr_col = i % image.shape[1]
        
        neighbors = get_pixel_neighborhood_data(
            curr_row, curr_col, R, h, w, use_data=False)
        eta_i = get_pixel_neighborhood_data(
            curr_row, curr_col, R, h, w, use_data=True, data=log_image)
        eta_i_bar = np.mean(eta_i)
        t_i = get_pixel_neighborhood_data(
            curr_row, curr_col, R, h, w, use_data=True, data=reflectance)
        for j in neighbors:
            # Check if neighbor is out of bounds
            if j[0] < 0 or j[0] >= image.shape[0] or j[1] < 0 or j[1] >= image.shape[1]:
                continue
            
            eta_j = get_pixel_neighborhood_data(
                j[0], j[1], R, h, w, use_data=True, data=log_image)
            eta_j_bar = np.mean(eta_j)
            t_j = get_pixel_neighborhood_data(
                j[0], j[1], R, h, w, use_data=True, data=reflectance)

            row.append(i)
            col.append(j[0] * image.shape[1] + j[1])

            refl_gaussian = gaussian_kernel(
                t_j, t_i,
                2 * omega_t, N)
            pixel_gaussian = gaussian_kernel(
                np.array([j[0], j[1]]),
                np.array([curr_row, curr_col]),
                omega_p, 2)
            eta_gaussian = gaussian_kernel(
                eta_j - eta_j_bar,
                eta_i - eta_i_bar,
                2 * omega_t,
                N)
            v_ij = refl_gaussian * pixel_gaussian * eta_gaussian
            data.append(v_ij)

    v = sparse.csr_matrix((np.asarray(data), (np.asarray(
        row), np.asarray(col))), shape=(num_pixels, num_pixels))
    print(f'v.shape={v.shape}')
    return v

def apply_new_illumination(image, new_illumination):
    'apply a new illumination on a given image'
    original_luminance, chrominance = find_luminance_chrominance(image, EPSILON=EPSILON)

    # Normalize the new illumination to match the scale of the original luminance
    normalized_new_illumination = new_illumination / \
        np.max(new_illumination) * np.max(original_luminance)

    updated_image = normalized_new_illumination * chrominance

    # Clip values to the valid range and convert to the appropriate datatype
    updated_image = np.clip(updated_image, 0, 255).astype(np.uint8)

    return updated_image


def scale_image(img, scale_factor):
    return gaussian(img, channel_axis=None)[::scale_factor, ::scale_factor].copy()


def create_image_pyramid(img, iterations, rev=False):
    res = [img]
    for i in range(iterations-1):
        img = scale_image(img, 2)
        res.append(img)
    return res[::-1] if rev else res