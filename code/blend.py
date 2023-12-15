from scipy import sparse
import numpy as np
from skimage.filters import gaussian
import cv2

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

    # TODO: Implement this function!
    src_r, src_g, src_b = source[:, :, 0], source[:, :, 1], source[:, :, 2]
    mask_r, mask_g, mask_b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    tgt_r, tgt_g, tgt_b = target[:, :, 0], target[:, :, 1], target[:, :, 2]
    # print(sparse.csr_matrix(src_r))

    blend_r, A = poisson_blend_channel(src_r, mask_r, tgt_r)
    blend_g, _ = poisson_blend_channel(src_g, mask_g, tgt_g, A=A)
    blend_b, _ = poisson_blend_channel(src_b, mask_b, tgt_b, A=A)

    return np.stack([blend_r, blend_g, blend_b], axis=2) / 255

def blend_one(im, mask):
    one_d_mask = mask[:, :, 0]
    # masked_im = im.copy()
    # masked_im[one_d_mask == 0] = 0
    # masked_im_2 = im.copy()
    # masked_im_2[one_d_mask == 255] = 0
    # plt.imshow(one_d_mask)
    # plt.show()
    # plt.imshow(masked_im)
    # plt.show()
    # plt.imshow(masked_im_2)
    # plt.show()
    inverted = mask.copy().astype(np.float32)
    inverted[one_d_mask == 255] = [0, 0, 0]
    inverted[one_d_mask < 255] = [255, 255, 255]
    h, w, _ = mask.shape
    inverted[:, 0:2, :] = 0
    inverted[:, w-2:w, :] = 0
    inverted[0:2, :, :] = 0
    inverted[h-2:h, :, :] = 0
    plt.imshow(inverted)
    plt.show()
    plt.imshow(poisson_blend(im, inverted.astype(np.float32), im))
    plt.show()
    ### CURRENTLY Making white because colliding with the wall I suppose. Need 
    ### to add a black border around mask?
    ### Blend once for grayscale, once for color?

blend_one(cv2.imread('../results/IMG_1167_R=1_49.jpg'), 
          cv2.imread('../data/shadow_mask.jpg'))

# def blend_images(a, b):
#     '''
#     Performs poisson blending given a warped image a and a base image b. 
#     Generates a mask for a, including all pixels in a that are not completely 
#     black. Applies a gaussian blur to the mask to effectively shrink the mask 
#     (algorithm includes only completely white values). Underlaps a below b, then
#     poisson blends a onto b overlapped onto a.
#     '''
#     # composites a onto b
#     # mask = np.sum(a, axis=2)
#     # mask[mask > 0] = 1
#     # mask = np.stack
#     mask_mask = (a == 0).all(axis=2)
#     mask = np.zeros(a.shape)
#     mask[mask_mask] = [0, 0, 0]
#     mask[~mask_mask] = [1, 1, 1]
#     mask = gaussian(mask, sigma=1.2)
#     # plt.imshow(mask)
#     # plt.show()
#     # return 
#     # print(a/255)
#     # return poisson_blend(a / 255, mask.astype(dtype=np.float32), b / 255) * 255
#     return poisson_blend(a / 255, mask.astype(dtype=np.float32), underlap(a, b) / 255) * 255

# def underlap(a, b):
#     # overlaps b onto a
#     out = b.copy()
#     out[out <= 0] = a[out <= 0]
#     return out