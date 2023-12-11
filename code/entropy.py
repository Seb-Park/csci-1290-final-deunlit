import numpy as np
from utils import gaussian_kernel, is_symmetric_and_positive_definite_sparse, get_pixel_neighborhood_data, find_luminance_chrominance
import matplotlib.pyplot as plt
import cv2
from scipy import sparse
from skimage.filters import gaussian
from tqdm import tqdm


EPSILON = 1e-6  # to avoid log 0
R = 1  # radius of the neighborhood
N = (2*R+1)**2-1  # size of the neighborhood


def minimize_energy(image, mask, image_name, initial_l, phi_l, phi_p, omega_t, omega_p, lambda_reg=1.0, num_iter=5, maxiter=1000, tol=1e-5):
    """
    Minimize the energy function using iterative optimization.
    :return: Optimal illumination estimate (IN LOG DOMAIN!!!!!!!!!!)
    """
    h, w = image.shape[0], image.shape[1]
    num_pixels = h * w

    curr_l = np.log(initial_l+EPSILON)  # (num_pixel, 1)
    curr_r = np.log(image.reshape((num_pixels, 1))+EPSILON) - \
        curr_l  # (num_pixel, 1)
    curr_image = image
    mask = mask.reshape((num_pixels, 1))
    for iteration in range(num_iter):
        print(f'------------Iteration: {iteration}-------------')
        u = calculate_weights_u(curr_image, phi_l, phi_p)
        v = calculate_weights_v(curr_image, omega_t, omega_p)
        eta_bar = mean_neighbor_log_intensity_differences(
            curr_image).reshape((num_pixels, 1))
        uv_sum = u + (lambda_reg * v)
        A = sparse.diags(np.array(uv_sum.sum(axis=1)).flatten())
        # is_valid_A, message = is_symmetric_and_positive_definite_sparse(A)
        # if not is_valid_A:
        #     print("A doesn't seem right")
        #     print(message)
        #     # print(check_matrix_properties_sparse(A))
        # else:
        #     print("A seems legit")
        b = uv_sum * curr_l + lambda_reg * \
            (sparse.diags(np.array(v.sum(axis=1)).flatten()) - v) * eta_bar
        b[mask != 0] = 0

        print(f'eta_bar.shape={eta_bar.shape}')
        print(f'A.shape={A.shape}')
        print(f'b.shape={b.shape}')

        curr_l, exit_code = sparse.linalg.cg(A, b, x0=curr_l, maxiter=maxiter)

        if exit_code == 0:
            print('CG SOLVER SUCCESS')
        elif exit_code > 0:
            print('CG SOLVER DID NOT CONVERGE')
        else:
            print('CG SOLVER ILLEGAL INPUT')

        curr_l = curr_l.reshape((num_pixels, 1))
        curr_l_reshaped = curr_l.reshape((h, w))
        curr_image = curr_image / np.exp(curr_l_reshaped)
        plt.imshow(curr_l_reshaped)
        plt.show()
        plt.imshow(np.exp(curr_l_reshaped))
        plt.show()
        # curr_r = np.log(image + EPSILON) - curr_l_reshaped
        # curr_image_log = curr_l_reshaped + curr_r # log domain
        # curr_image = np.exp(curr_image_log)
        curr_img_int = np.clip((curr_image * 255).astype(np.uint8), 0, 255)

        # print(curr_l)
        # print(curr_img_int)
        print(f"optimal_l min: {np.min(np.exp(curr_l))}")
        print(f"optimal_l max: {np.max(np.exp(curr_l))}")
        print(f"optimal_r min: {np.min(np.exp(curr_r))}")
        print(f"optimal_r max: {np.max(np.exp(curr_r))}")
        cv2.imwrite(
            f'../results/{image_name}_R={R}_{iteration}.jpg', curr_img_int)

    return curr_l


def minimize_energy_pyramid(image, mask, initial_l, phi_l, phi_p, omega_t, omega_p, lambda_reg=1.0, num_iter=5, maxiter=10000, tol=1e-5, img_name='1171'):
    """
    Minimize the energy function using iterative optimization.
    :return: Optimal illumination estimate (IN LOG DOMAIN!!!!!!!!!!)
    """
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[0], image.shape[1]
    num_pixels = h * w

    curr_l = initial_l
    pyramid = create_image_pyramid(image, num_iter, rev=True)
    curr_image = pyramid[0]
    mask_pyramid = create_image_pyramid(mask, num_iter, rev=True)
    # mask = mask.reshape((num_pixels, 1))
    for iteration in range(num_iter):
        print(f'------------Iteration: {iteration}-------------')
        # curr_image = pyramid[iteration]
        h, w = curr_image.shape[0], curr_image.shape[1]
        num_pixels = h * w
        mask = mask_pyramid[iteration].reshape((num_pixels, 1))
        u = calculate_weights_u(curr_image, phi_l, phi_p)
        v = calculate_weights_v(curr_image, omega_t, omega_p)
        eta_bar = mean_neighbor_log_intensity_differences(
            curr_image).reshape((num_pixels, 1))
        uv_sum = u + (lambda_reg * v)
        A = sparse.diags(np.array(uv_sum.sum(axis=1)).flatten())
        # is_valid_A, message = is_symmetric_and_positive_definite_sparse(A)
        # if not is_valid_A:
        #     print("A doesn't seem right")
        #     print(message)
        #     # print(check_matrix_properties_sparse(A))
        # else:
        #     print("A seems legit")
        b = uv_sum * curr_l + lambda_reg * \
            (sparse.diags(np.array(v.sum(axis=1)).flatten()) - v) * eta_bar
        b[mask != 0] = 0

        print(f'eta_bar.shape={eta_bar.shape}')
        print(f'A.shape={A.shape}')
        print(f'b.shape={b.shape}')
        b[mask == 0] = 0
        curr_l, exit_code = sparse.linalg.cg(
            A, b, x0=curr_l, maxiter=maxiter, tol=tol)
        # print(f'eta_bar.shape={eta_bar.shape}')
        # print(f'A.shape={A.shape}')
        # print(f'b.shape={b.shape}')

        curr_l, exit_code = sparse.linalg.cg(
            A, b, x0=curr_l, maxiter=maxiter, tol=tol)
        curr_l = curr_l.reshape((num_pixels, 1))

        if exit_code == 0:
            print('CG SOLVER SUCCESS')
        elif exit_code > 0:
            print('CG SOLVER DID NOT CONVERGE')
        else:
            print('CG SOLVER ILLEGAL INPUT')

        optimal_r = np.log(image + EPSILON).astype(np.int64) - \
            curr_l.reshape((image.shape[0], image.shape[1])).astype(np.int64)
        optimal_r = scale_up(optimal_r)
        print(pyramid[iteration+1].shape)
        print(optimal_r.shape)
        curr_image = np.multiply(np.exp(curr_l.reshape(
            (pyramid[iteration + 1].shape[0],
             pyramid[iteration + 1].shape[1]))), np.exp(optimal_r)).astype(np.uint8)
        cv2.imwrite(
            f'../results/test_{img_name}_new15_{iteration}.jpg', curr_image)
        # curr_image = apply_new_illumination(curr_image, curr_l.reshape((h,w)))

    return curr_l


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
            # neighborhood with boundary considerations
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

    # necessary info for sparse matrix
    row = []
    col = []
    data = []

    illumination, _ = find_luminance_chrominance(image)
    h, w = image.shape[0], image.shape[1]
    # loop through each pixel
    num_pixels = h * w
    for i in tqdm(range(num_pixels), desc='Calculating u'):
        curr_row = i // w
        curr_col = i % w

        # Get four neighboring pixels TODO: why only four?
        # TODO: Also, should we include the pixel itself or does
        # that not get weighted?
        # neighbors = [(curr_row - 1, curr_col), (curr_row + 1, curr_col),
        #              (curr_row, curr_col - 1), (curr_row, curr_col + 1)]
        neighbors = get_pixel_neighborhood_data(
            curr_row, curr_col, R, h, w, use_data=False)
        for j in neighbors:
            if j[0] < 0 or j[0] >= h or j[1] < 0 or j[1] >= w:
                continue

            # The calculated weight for this value is to be placed in the row
            # corresponding to pixel i and the column corresponding to this
            # neighboring pixel
            row.append(i)
            col.append(j[0] * w + j[1])

            # Weights the color difference between two pixels
            illum_gaussian = gaussian_kernel(
                np.array([illumination[j[0]][j[1]]]),
                np.array([illumination[curr_row][curr_col]]),
                2*phi_l,
                1)

            # Weights the distance between two pixels
            pixel_gaussian = gaussian_kernel(
                np.array([j[0], j[1]]),
                np.array([curr_row, curr_col]),
                2*phi_p,
                2)
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
    _, reflectance = find_luminance_chrominance(image)

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
            eta_j = get_pixel_neighborhood_data(
                j[0], j[1], R, h, w, use_data=True, data=log_image)
            eta_j_bar = np.mean(eta_j)
            t_j = get_pixel_neighborhood_data(
                j[0], j[1], R, h, w, use_data=True, data=reflectance)

            row.append(i)
            col.append(j[0] * image.shape[1] + j[1])

            refl_gaussian = gaussian_kernel(
                t_j,
                t_i,
                2 * omega_t,
                N)

            pixel_gaussian = gaussian_kernel(
                np.array([j[0], j[1]]),
                np.array([curr_row, curr_col]),
                omega_p,
                2)

            eta_gaussian = gaussian_kernel(
                eta_j - eta_j_bar,
                eta_i - eta_i_bar,
                2 * omega_t,
                N)
            data.append(refl_gaussian * pixel_gaussian * eta_gaussian)
    v = sparse.csr_matrix((np.asarray(data), (np.asarray(
        row), np.asarray(col))), shape=(num_pixels, num_pixels))
    print(f'v.shape={v.shape}')
    return v


def apply_new_illumination(image, new_illumination):
    'apply a new illumination on a given image'
    # original_luminance, chrominance = find_luminance_chrominance(image)

    # normalized_new_illumination = new_illumination / \
    #     np.max(new_illumination) * np.max(original_luminance)

    log_intensity = np.log(image + EPSILON).astype(np.int64)
    optimal_l = new_illumination.reshape(
        (image.shape[0], image.shape[1])).astype(np.int64)

    optimal_r = log_intensity - optimal_l

    curr_image = np.multiply(
        np.exp(optimal_l), np.exp(optimal_r)).astype(np.uint8)

    # updated_image = normalized_new_illumination * chrominance
    # updated_image = np.zeros_like(image, dtype=float)
    # for i in range(3):  # Iterate over the RGB channels
    #     updated_image[:, :, i] = normalized_new_illumination * chrominance[:, :, i]

    # updated_image = np.clip(curr_image, 0, 255).astype(np.uint8)

    return curr_image


def scale_image(img, scale_factor):
    return gaussian(img, channel_axis=None)[::scale_factor, ::scale_factor].copy()


def create_image_pyramid(img, iterations, rev=False):
    res = [img]
    for i in range(iterations-1):
        img = scale_image(img, 2)
        res.append(img)
    return res[::-1] if rev else res


def scale_up(img):
    return img.repeat(2, axis=0).repeat(2, axis=1)

# def minimize_energy_with_pyramid(image, initial_l, phi_l, phi_p, omega_t, omega_p, lambda_reg=1.0, num_iter=5):
#     """
#     Minimize the energy function using iterative optimization.
#     :param image: Input image.
#     :param initial_l: Initial estimate of illumination.
#     :param lambda_reg: Regularization parameter.
#     :return: Optimal illumination estimate.
#     """
#     # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     h, w = image.shape[0], image.shape[1]
#     num_pixels = h * w

#     curr_l = initial_l
#     curr_image = image
#     image_pyramid = create_image_pyramid(image, num_iter)
#     mask = cv2.imread('shadow_mask.jpg')
#     mask_pyramid = create_image_pyramid(mask, num_iter)
#     for iteration in range(num_iter):
#         print(f'------------Iteration: {iteration}-------------')
#         curr_image = image_pyramid[iteration]
#         h, w = curr_image.shape[0], curr_image.shape[1]
#         num_pixels = h * w
#         u = calculate_weights_u(curr_image, phi_l, phi_p)
#         v = calculate_weights_v(curr_image, omega_t, omega_p)
#         eta_bar = mean_neighbor_log_intensity_differences(curr_image).reshape((num_pixels, 1))
#         A = u + (lambda_reg * v)
#         b = lambda_reg * v * eta_bar

#         mask_gray = cv2.cvtColor(mask_pyramid[iteration], cv2.COLOR_BGR2GRAY).reshape((num_pixels, 1))
#         for i in range(num_pixels):
#             if mask_gray[i] != 0:
#                 b[i] = 0

#         print(f'u.shape={u.shape}')
#         print(f'v.shape={v.shape}')
#         print(f'eta_bar.shape={eta_bar.shape}')
#         print(f'A.shape={A.shape}')
#         print(f'b.shape={b.shape}')
#         curr_l, exit_code = sparse.linalg.cg(A, b, x0=curr_l, maxiter=5)
#         optimal_r = np.log(image + EPSILON).astype(np.int64) - curr_l.reshape((image.shape[0], image.shape[1])).astype(np.int64)
#         curr_image = np.multiply(np.exp(curr_l.reshape((image.shape[0], image.shape[1]))), np.exp(optimal_r)).astype(np.uint8)
#         # curr_image = apply_new_illumination(curr_image, curr_l.reshape((h,w)))

#     return curr_l
