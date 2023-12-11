import numpy as np
from utils import gaussian_kernel, is_symmetric_and_positive_definite_sparse
import matplotlib.pyplot as plt
import cv2
from scipy import sparse
from skimage.filters import gaussian


EPSILON = 1e-6  # to avoid log 0


# def energy_function(curr_L, image, lambda_reg, phi_l, phi_p, omega_t):
#     """
#     Define the energy function to be minimized according to equation 16.
#     :param curr_L: Current illumination estimate, (h, w)
#     :param image: Input image, (h, w)
#     :param lambda_reg: Regularization parameter, (1,)
#     :param phi_l, phi_p, omega_t: smoothing matrices to calculate weights u and v
#     :return: Energy value for the current illumination estimate.
#     """
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     h, w = image.shape[0], image.shape[1]
#     # u and v should be sparse matrix of (h*w, h*w)
#     u = calculate_weights_u(image, phi_l, phi_p)
#     v = calculate_weights_v(image, omega_t)
#     eta_bar = mean_neighbor_log_intensity_differences(image_gray)

#     L_star = np.zeros((h, w))

#     # using r and c for row and col to avoid confusing with pixel i and neighbor j
#     for r in range(h):
#         for c in range(w):
#             # Handle boundaries
#             top = max(r-1, 0)
#             bottom = min(r+1, h-1)
#             left = max(c-1, 0)
#             right = min(c+1, w-1)

#             # log illumination of current pixel
#             log_L_i = np.log(curr_L[r, c] + EPSILON)
#             l_ijs = [
#                 np.abs(log_L_i - np.log(curr_L[top, c] + EPSILON)),
#                 np.abs(log_L_i - np.log(curr_L[bottom, c] + EPSILON)),
#                 np.abs(log_L_i - np.log(curr_L[r, left] + EPSILON)),
#                 np.abs(log_L_i - np.log(curr_L[r, right] + EPSILON))
#             ]
#             u_ijs = [
#                 u[r*c, top*c],
#                 u[r*c, bottom*c],
#                 u[r*c, r*left],
#                 u[r*c, r*right],
#             ]
#             v_ijs = [
#                 v[r*c, top*c],
#                 v[r*c, bottom*c],
#                 v[r*c, r*left],
#                 v[r*c, r*right],
#             ]

#             smoothness_term = u_ijs * l_ijs**2
#             texture_term = v_ijs * (l_ijs - eta_bar[r, c])**2

#             L_star[r, c] = smoothness_term + lambda_reg * texture_term

#     return L_star


def minimize_energy(image, mask, initial_l, phi_l, phi_p, omega_t, omega_p, lambda_reg=1.0, num_iter=10, maxiter=10000, tol=1e-5, img_name='1171'):
    """
    Minimize the energy function using iterative optimization.
    :return: Optimal illumination estimate (IN LOG DOMAIN!!!!!!!!!!)
    """
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[0], image.shape[1]
    num_pixels = h * w

    curr_l = initial_l
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

        print(f'eta_bar.shape={eta_bar.shape}')
        print(f'A.shape={A.shape}')
        print(f'b.shape={b.shape}')
        b[mask == 0] = 0
        curr_l, exit_code = sparse.linalg.cg(A, b, x0=curr_l, maxiter=maxiter, tol=tol)
        curr_l = curr_l.reshape((num_pixels, 1))
        print(f'curr_l.shape={curr_l.shape}')
        if exit_code == 0:
            print('CG SOLVER SUCCESS')
        elif exit_code > 0:
            print('CG SOLVER DID NOT CONVERGE')
        else:
            print('CG SOLVER ILLEGAL INPUT')

        optimal_r = np.log(image + EPSILON).astype(np.int64) - \
            curr_l.reshape((image.shape[0], image.shape[1])).astype(np.int64)
        curr_image = np.multiply(np.exp(curr_l.reshape(
            (image.shape[0], image.shape[1]))), np.exp(optimal_r)).astype(np.uint8)
        cv2.imwrite(f'../results/test_{img_name}_new15_{iteration}.jpg', curr_image)
        # curr_image = apply_new_illumination(curr_image, curr_l.reshape((h,w)))

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

        print(f'eta_bar.shape={eta_bar.shape}')
        print(f'A.shape={A.shape}')
        print(f'b.shape={b.shape}')
        b[mask == 0] = 0
        curr_l, exit_code = sparse.linalg.cg(A, b, x0=curr_l, maxiter=maxiter, tol=tol)
        curr_l = curr_l.reshape((num_pixels, 1))
        print(f'curr_l.shape={curr_l.shape}')
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
        cv2.imwrite(f'../results/test_{img_name}_new15_{iteration}.jpg', curr_image)
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

    # loop through each pixel
    num_pixels = image.shape[0] * image.shape[1]
    for i in range(num_pixels):
        curr_row = i // image.shape[1]
        curr_col = i % image.shape[1]

        ### Get four neighboring pixels TODO: why only four? 
        ### TODO: Also, should we include the pixel itself or does 
        ###       that not get weighted?
        neighbors = [(curr_row - 1, curr_col), (curr_row + 1, curr_col),
                     (curr_row, curr_col - 1), (curr_row, curr_col + 1)]
        # neighbors = [(curr_row, curr_col),
        #              (curr_row - 1, curr_col), (curr_row + 1, curr_col),
        #              (curr_row, curr_col - 1), (curr_row, curr_col + 1),
        #              (curr_row + 1, curr_col + 1), (curr_row - 1, curr_col + 1),
        #              (curr_row - 1, curr_col - 1), (curr_row + 1, curr_col - 1)]
        for dir in neighbors:
            if dir[0] < 0 or dir[0] >= image.shape[0] or dir[1] < 0 or dir[1] >= image.shape[1]:
                continue

            ### The calculated weight for this value is to be placed in the row 
            ### corresponding to pixel i and the column corresponding to this 
            ### neighboring pixel
            row.append(i)
            col.append(dir[0] * image.shape[1] + dir[1])

            ### Weights the color difference between two pixels
            illum_gaussian = gaussian_kernel(
                np.array([illumination[curr_row][curr_col]]),
                np.array([illumination[dir[0]][dir[1]]]),
                phi_l, 1)

            ### Weights the distance between two pixels
            pixel_gaussian = gaussian_kernel(
                np.array([curr_row, curr_col]),
                np.array([dir[0], dir[1]]), phi_p, 2)
            ### TODO: I'm confused--aren't these the same for all for pixels?
            ###       as in, the four neighboring pixels will all be exactly the
            ###       same distance away from the center pixel, right?

            data.append(illum_gaussian * pixel_gaussian)
    
    u = sparse.csr_matrix((np.asarray(data)[:, 0], (np.asarray(row), np.asarray(col))), shape=(num_pixels, num_pixels))
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
    for i in range(num_pixels):
        curr_row = i // image.shape[1]
        curr_col = i % image.shape[1]

        top = max(curr_row-1, 0)
        bottom = min(curr_row+1, h-1)
        left = max(curr_col-1, 0)
        right = min(curr_col+1, w-1)

        neighbors = [(top, curr_col), (bottom, curr_col),
                     (curr_row, left), (curr_row, right)]
        eta_i = [log_image[neighbors[0]],
                 log_image[neighbors[1]],
                 log_image[neighbors[2]],
                 log_image[neighbors[3]]]
        eta_i_bar = np.mean(eta_i)
        for dir in neighbors:
            # Check if neighbor is out of bounds
            if dir[0] < 0 or dir[0] >= image.shape[0] or dir[1] < 0 or dir[1] >= image.shape[1]:
                continue

            top_dir = max(dir[0]-1, 0)
            bottom_dir = min(dir[0]+1, h-1)
            left_dir = max(dir[1]-1, 0)
            right_dir = min(dir[1]+1, w-1)

            neighbors_dir = [(top_dir, dir[1]), (bottom_dir, dir[1]),
                             (dir[0], left_dir), (dir[0], right_dir)]
            eta_dir = [
                log_image[neighbors_dir[0]],
                log_image[neighbors_dir[1]],
                log_image[neighbors_dir[2]],
                log_image[neighbors_dir[3]],
            ]

            row.append(i)
            col.append(dir[0] * image.shape[1] + dir[1])
            refl_gaussian = gaussian_kernel(
                np.array([reflectance[curr_row][curr_col]]),
                np.array([reflectance[dir[0]][dir[1]]]),
                2 * np.array([[90.2]]) , 1)
            pixel_gaussian = gaussian_kernel(
                np.array([curr_row, curr_col]),
                np.array([dir[0], dir[1]]),
                omega_p, 2)
            eta_gaussian = gaussian_kernel(
                eta_dir - eta_i_bar,
                eta_i - eta_i_bar,
                2 * omega_t, 4)
            data.append(refl_gaussian * pixel_gaussian * eta_gaussian)
    v = sparse.csr_matrix((np.asarray(data)[:, 0], (np.asarray(row), np.asarray(col))), shape=(num_pixels, num_pixels))
    print(f'v.shape={v.shape}')
    return v


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


def apply_new_illumination(image, new_illumination):
    'apply a new illumination on a given image'
    original_luminance, chrominance = find_luminance_chrominance(image)

    # Normalize the new illumination to match the scale of the original luminance
    normalized_new_illumination = new_illumination / \
        np.max(new_illumination) * np.max(original_luminance)

    updated_image = normalized_new_illumination * chrominance
    # updated_image = np.zeros_like(image, dtype=float)
    # for i in range(3):  # Iterate over the RGB channels
    #     updated_image[:, :, i] = normalized_new_illumination * chrominance[:, :, i]

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




# def check_all_eigenvalues(A):
#     # Convert sparse matrix to dense, if not already dense
#     if sparse.issparse(A):
#         A = A.toarray()

#     # Compute all eigenvalues
#     eigenvalues = eigh(A, eigvals_only=True)

#     # Check if all eigenvalues are positive
#     if np.all(eigenvalues > 0):
#         return "All eigenvalues are positive, matrix is positive definite"
#     else:
#         return "Some eigenvalues are non-positive, matrix is not positive definite"