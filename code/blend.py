from scipy import sparse
import numpy as np
from skimage.filters import gaussian
import cv2
from utils import find_luminance_chrominance

from matplotlib import pyplot as plt

def alpha_blending(source, mask, target):
    """
    Performs alpha blending. 
    Source, mask, and target are all numpy arrays of the same shape 
    (this is ensured by the fix_images function called in main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask. Could also be matte.
        target - np.array of target image

    Returns:
        np.array of blended image
    """

    return ((source * mask) + (target * (1 - mask))) / 255

def find_neighbors(index, width, height):
    res = []
    xpos, ypos = index % width, index // width
    max = width * height
    if xpos + 1 < width:
        res.append(index + 1)
    if xpos - 1 >= 0:
        res.append(index - 1)
    if index + width < max:
        res.append(index + width)
    if index - width >= 0:
        res.append(index - width)
    return res

def poisson_blend_channel(source, mask, target, A=None):
    height, width = target.shape ## Get image shape

    flat_src = source.flatten() ## Flatten out source image into 1-D array
    flat_target = target.flatten() ## Flatten out target image into 1-D array
    flat_mask = mask.flatten() ## Flatten mask

    b = np.array(flat_target) ## Set b to be target
    make_a = (A is None) # Only construct A if we don't have existing A
    if(make_a): 
        # Initialize A as a diagonal of ones if we don't have an A already 
        # (because we calculate separate channels but A stays the same)
        # https://docs.scipy.org/doc/scipy/reference/sparse.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye.html
        A = sparse.lil_matrix(sparse.eye(target.shape[0] * target.shape[1], dtype=int))

    # Find all the indices that the mask includes
    mask_indices = np.where(flat_mask == 1)[0]

    for i in mask_indices:
        # Find neighboring pixels. Note: only includes ones in the bounds of the image
        neighbors = find_neighbors(i, width, height)
        if(make_a):
            # Along the diagonal in A, set all pixels that the mask includes to 4
            # As is described in the slides
            A[i, i] = 4
            # Make all the neighboring pixels -1
            A[i, neighbors] = -1
        # Edit b-values within the mask to be 4 times the pixel value minus all
        # neighbor values
        b[i] = 4 * flat_src[i] - sum(flat_src[neighbors])
    if(make_a):
        # Convert A to a csr array to solve linear system with sparse.linalg
        # If A is newly constructed
        A = A.tocsr()
    # Solve linear system
    # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
    x = sparse.linalg.spsolve(A, b)
    
    # Reshape to 2D, return A
    return np.reshape(x, (height, width)), A

def poisson_blend(source, mask, target):
    """
    Performs Poisson blending. Source, mask, and target are all numpy arrays
    of the same shape (this is ensured by the fix_images function called in
    main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask
        target - np.array of target image

    Returns:
        np.array of blended image
    """
    print(source.shape, mask.shape, target.shape)
    # TODO: Implement this function!
    src_r, src_g, src_b = source[:, :, 0], source[:, :, 1], source[:, :, 2]
    mask_r, mask_g, mask_b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    tgt_r, tgt_g, tgt_b = target[:, :, 0], target[:, :, 1], target[:, :, 2]
    # print(sparse.csr_matrix(src_r))

    blend_r, A = poisson_blend_channel(src_r, mask_r, tgt_r)
    blend_g, _ = poisson_blend_channel(src_g, mask_g, tgt_g, A=A)
    blend_b, _ = poisson_blend_channel(src_b, mask_b, tgt_b, A=A)

    return np.stack([blend_r, blend_g, blend_b], axis=2)

def invert_mask(mask, pad=False):
    '''
    Inverts mask to work with poisson blending, pads edges so that program
    doesn't try to blend with the edges of the image.
    '''
    one_d_mask = mask[:, :, 0]
    inverted = mask.copy().astype(np.float32)
    inverted[one_d_mask >= 127] = [0, 0, 0]
    inverted[one_d_mask < 127] = [1., 1., 1.]
    # inverted = gaussian(inverted, sigma=0.5)
    # inverted[inverted < 1] = 0
    if pad:
        h, w, _ = mask.shape
        inverted[:, 0:2, :] = 0
        inverted[:, w-2:w, :] = 0
        inverted[0:2, :, :] = 0
        inverted[h-2:h, :, :] = 0
    return inverted

def blend_gray(im, mask, original):
    blended = poisson_blend(im.astype(np.float32) / 255, 
                            mask.astype(np.float32), 
                            original.astype(np.float32) / 255)
    print (np.max(blended))
    plt.imshow(blended.clip(0, 1))
    plt.show()
    ### CURRENTLY Making white because colliding with the wall I suppose. Need 
    ### to add a black border around mask?
    ### Blend once for grayscale, once for color?

mask = invert_mask(cv2.imread('../data/shadow_mask.jpg'))

original_im = cv2.imread('../data/IMG_1167.jpg')
original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2GRAY)
original_im = np.stack([original_im] * 3, axis=2)

blend_gray(cv2.imread('../results/IMG_1167_R=1_49.jpg'), 
           mask,
           original_im)