import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg


def gaussian_kernel(x, ai, psi, d):
    '''
    Multi-dimensional gaussian kernel according to equation 3
    '''
    # d = x.shape[0]  # Dimensionality of the data
    # normalization_factor = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(psi) ** 0.5)
    normalization_factor = 1 / ((2 * np.pi) ** (d / 2) * psi ** 0.5)
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