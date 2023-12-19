import cv2
from utils import find_luminance_chrominance
import numpy as np
import matplotlib.pyplot as plt
from blend import invert_mask
from skimage import color

def combine_luma_chroma(luma_img, chroma_img):
    luma_img = luma_img.astype(np.float32) / 255
    chroma_img = chroma_img.astype(np.float32)
    return (np.clip(luma_img * chroma_img, 0, 1) * 255).astype(np.uint8)

# lum = cv2.imread('../extra_post/2411/final_luma.jpg')

# orig_luma, chrom = find_luminance_chrominance(cv2.imread('../extra_post/2411/original.jpg').astype(np.float32))

# # print(chrom.dtype)
# # print(orig_luma)
# print(chrom)
# print(np.max(chrom[:, :, 2]))

# max_b, max_g, max_r = np.max(chrom[:, :, 0]), np.max(chrom[:, :, 1]), np.max(chrom[:, :, 2])

# chrom[:, :, 0] = chrom[:, :, 0] / max_b
# chrom[:, :, 1] = chrom[:, :, 1] / max_g
# chrom[:, :, 2] = chrom[:, :, 2] / max_r

# print(np.max(chrom))

# plt.imshow(np.flip(chrom / np.max(chrom), axis=2))
# plt.show()

# plt.imshow(cv2.cvtColor(combine_luma_chroma(lum, chrom), cv2.COLOR_BGR2RGB))
# plt.show()

################################################################################

def blend_color_one_image(img, mask):
    '''
    Takes the image to blend
    and an "inverted" mask, where shadowed pixels are positive, and non-shadowed
    pixels are [0, 0, 0]. 
    
    Adjusts the shadowed pixels to match the color of the non-shadowed pixels
    '''
    # one_d_mask = mask[:, :, 0]
    # bool_mask = np.full((mask.shape[0], mask.shape[1]), True)
    # bool_mask[one_d_mask == 0] = False
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
    one_d_bool_mask = bool_mask[0, :, :]
    combined = np.ma.array(new_lab_shadow.filled(1) * non_shadowed_part.filled(1))
    combined = color.lab2rgb(new_lab_shadow) * 255
    plt.imshow(np.flip(combined.astype(np.uint8), axis=2))
    plt.show()

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

recolored = cv2.imread('../extra_post/2254/recolored.jpg')
mask = invert_mask(cv2.imread('../extra_post/2254/mask.jpg'))

blend_color_one_image(recolored, mask)