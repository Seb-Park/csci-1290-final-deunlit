from scipy import sparse
import numpy as np
from skimage import color
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
        source - np.array of source image between 0 and 1
        mask   - np.array of binary mask between 0 and 1
        target - np.array of target image between 0 and 1

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
    ### TODO: IS blending too many times.
    """
    Im must be between 0 and 1
    Mask must be between 0 and 1
    Original is between 0 and 255
    """
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    original = np.stack([original] * 3, axis=2)
    blended = poisson_blend(im.astype(np.float32), 
                            mask.astype(np.float32), 
                            original.astype(np.float32) / 255)
    return blended.clip(0, 1)

def blend_color(im, mask):
    blended = poisson_blend(im.astype(np.float32), 
                            mask.astype(np.float32), 
                            im.astype(np.float32))
    plt.imshow(np.flip(blended.clip(0, 1), axis=2)-np.flip(im, axis=2))
    plt.show()
    print(np.max(np.abs(np.flip(blended.clip(0, 1), axis=2)-np.flip(im, axis=2))))
    return blended.clip(0, 1)

def blend_color_one_image(img, mask, original=None):
    '''
    Takes the image to blend in RGB [0, 255]
    and an "inverted" mask, where shadowed pixels are positive, and non-shadowed
    pixels are [0, 0, 0]. 
    
    Adjusts the shadowed pixels to match the color of the non-shadowed pixels
    '''
    bool_mask = np.full(mask.shape, True)
    bool_mask[mask == 0] = False
    img = color.rgb2lab(img / 255)
    non_shadowed_part = np.ma.masked_array(img, mask=bool_mask)
    shadowed_part = np.ma.masked_array(img, mask=~bool_mask)
    shadow_l, shadow_a, shadow_b = align_three_channels(shadowed_part, non_shadowed_part)
    new_lab_shadow = np.dstack((np.clip(shadow_l, 0, 100),
                         np.clip(shadow_a, -100, 100),
                         np.clip(shadow_b, -100, 100)))
    combined = np.zeros(img.shape)
    combined = np.ma.array(new_lab_shadow.filled(1) * non_shadowed_part.filled(1))
    combined = color.lab2rgb(new_lab_shadow) * 255
    return combined.astype(np.uint8)

def align_channels(ch1, ch2):
    '''
    Aligns ch1 to be with ch2
    '''
    return ((ch1 - np.mean(ch1)) / np.std(ch1)) * np.std(ch2) + np.mean(ch2)

def align_three_channels(im1, im2):
    im1_x, im1_y, im1_z = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
    im2_x, im2_y, im2_z = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]
    return align_channels(im1_x, im2_x), \
        align_channels(im1_y, im2_y), \
        align_channels(im1_z, im2_z)

# mask = invert_mask(cv2.imread('../data/shadow_mask.jpg'))

# original_im = cv2.imread('../data/IMG_1167.jpg')

# processed = cv2.imread('../results/IMG_1167_R=1_49.jpg')

# colored = cv2.imread('../test.png')

# print(processed.dtype, mask.dtype, original_im.dtype)

# blent = blend_gray(processed.astype(np.float32) / 255,
#                    mask,
#                    original_im)

# blend_color(colored.astype(np.float32) / 255,
#             mask)