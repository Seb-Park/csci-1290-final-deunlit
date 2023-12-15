import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg


def gaussian_kernel(x, ai, psi, d):
    '''
    Multi-dimensional gaussian kernel according to equation 3
    '''
    # d = x.shape[0]  # Dimensionality of the data
    normalization_factor = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(psi) ** 0.5)
    exponent = 1.0
    if d > 1:
        exponent = -0.5 * np.dot(np.dot((x - ai).T, np.linalg.inv(psi)), (x - ai))
        return normalization_factor * np.exp(exponent)
    else:
        exponent = -0.5 * np.dot((x - ai).T, (x - ai) / psi)
        return normalization_factor * np.exp(exponent)
    

def get_pixel_neighborhood_data(i_x, i_y, radius, h, w, use_data, data=None):
    
    # if radius == 0, return 4 direct neighbors
    if radius == 0:
        top = max(i_x-1, 0)
        bottom = min(i_x+1, h-1)
        left = max(i_y-1, 0)
        right = min(i_y+1, w-1)
        if use_data:
            return np.array([data[top, i_y],data[bottom, i_y], data[i_x, left],data[i_x, right]])
        else: 
            return np.array([(top, i_y), (bottom, i_y), (i_x, left), (i_x, right)])
        

    neighborhood_coords = []
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
        # try:
        #     # Find the smallest eigenvalue
        #     eigenvalues, _ = spla.eigsh(A, k=1, which='SA')
        #     if eigenvalues[0] > 0:
        #         return True, "Matrix is symmetric and positive definite"
        #     else:
        #         return False, "Matrix is symmetric but not positive definite"
        # except spla.ArpackNoConvergence:
        #     return False, "Matrix is symmetric but not positive definite"
    else:
        return False, "Matrix is not symmetric"

rgb2gray_weightr = 0.2125
rgb2gray_weightg = 0.7154
rgb2gray_weightb = 0.0721


def find_luminance_chrominance(image, EPSILON=0):
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
    return np.clip(luminance, 0, 255), np.clip(chrominance, 0, 255)

# def check_matrix_properties_sparse(A):
#     # Check if A is square
#     if A.shape[0] != A.shape[1]:
#         return "Matrix is not square"

#     # Check diagonal elements
#     diags = A.diagonal()
#     if np.any(diags <= 0):
#         return "Matrix has non-positive diagonal elements"

#     # Compute a few eigenvalues (largest in magnitude)
#     try:
#         eigenvalues = eigs(A, k=6, which='LM', return_eigenvectors=False)
#         # Check if any computed eigenvalues are non-positive
#         if np.any(eigenvalues.real <= 0):
#             return "Matrix has non-positive eigenvalues"
#     except RuntimeError as e:
#         return f"Error in computing eigenvalues: {e}"

#     return "Matrix checks passed"