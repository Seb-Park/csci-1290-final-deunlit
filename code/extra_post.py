import cv2
from utils import find_luminance_chrominance
import numpy as np
import matplotlib.pyplot as plt
from blend import invert_mask, poisson_blend, blend_color_one_image
from skimage import color
from skimage.filters import gaussian

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

recolored = cv2.imread('../extra_post/2411/recolored.jpg')
mask = invert_mask(cv2.imread('../extra_post/2411/mask.jpg'))
original = cv2.imread('../extra_post/2411/original.jpg')

blend_color_one_image(recolored, mask)