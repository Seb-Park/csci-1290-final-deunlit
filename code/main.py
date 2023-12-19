import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from entropy import minimize_energy, N, R
from utils import find_luminance_chrominance
from blend import blend_gray, invert_mask

EPSILON = 1e-6 

def main():
    image_path = '../data/'
    src_name = 'IMG_2411'
    mask_name = 'mask_2411'
    jpg = '.jpg'
    png='.png'

    image_color = cv2.imread( image_path + src_name + jpg )
    lum, chrom = find_luminance_chrominance( image_color, EPSILON )
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) 

    # print(chrom)

    image = image_gray
    # plt.imshow(image_gray - lum.astype(np.int32))
    # plt.show()
    # cv2.imshow("i0", image_gray)
    mask = cv2.imread( image_path + mask_name + jpg )
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # np.set_printoptions(threshold=sys.maxsize)
    # print(mask_gray)

    initial_l = np.ones((image.shape[0]*image.shape[1], 1)) # in log domain
    phi_l = np.array([[0.2]]) 
    phi_p = np.array([[0.2, 0.0], [0.0, 0.2]])
    omega_t = np.diag([90]*N)
    omega_p = np.array([[90.2, 0.0], [0.0, 90.2]])
    lambda_reg = 1.0 
    num_iter = 50

    optimal_l, optimal_img = minimize_energy(image, mask_gray, initial_l, \
                                phi_l=phi_l, phi_p=phi_p, \
                                omega_t=omega_t, omega_p=omega_p, \
                                img_name=src_name, num_iter=num_iter,
                                chrominance=chrom)
    optimal_l = optimal_l.reshape((image.shape[0], image.shape[1]))
    image_gray = image_gray.astype(np.int64)
    optimal_r = np.log(image_gray + EPSILON).astype(np.float64) - optimal_l.astype(np.int64)
    # optimal_img = np.multiply(np.exp(optimal_l), np.exp(optimal_r)).astype(np.uint8)
    plt.imshow(optimal_img, cmap='gray')
    plt.show()
    max_l_star = np.max(optimal_img)
    # plt.imshow(np.flip(chrom, axis=2))
    plt.imshow(np.multiply(np.flip(chrom, axis=2), np.stack([optimal_img] * 3, axis=2) / max_l_star))
    plt.show()
    # plt.imshow(np.stack([optimal_img] * 3, axis=2))
    # plt.show()
    # plt.imshow(image_color)
    # plt.show()
    # print(np.stack([optimal_img] * 3, axis=2), optimal_img.shape)
    # print(mask)
    # print(f"asdfasdf {np.stack([optimal_img] * 3, axis=2).dtype} {mask.dtype} {image_color.dtype}")
    inverted_mask = invert_mask(mask)
    blent = blend_gray(np.stack([optimal_img] * 3, axis=2), inverted_mask, image_color)
    plt.imshow(blent)
    plt.show()
    rechromed = np.multiply(chrom, blent)
    rechromed = rechromed.clip(0, 1)
    print(rechromed.dtype)
    print(np.max(rechromed))
    print(np.min(rechromed))
    plt.imshow(np.flip(rechromed, axis=2))
    plt.show()
    cv2.imwrite(f'../results/{src_name}_color_{num_iter}_iters_R={R}.jpg', (rechromed * 255).astype(np.uint8))
    # cv2.imshow("i", optimal_img)
    # cv2.waitKey(0)
    print("--------------OPTIMAL--------------")
    print(f"optimal_l min: {np.min(np.exp(optimal_l))}")
    print(f"optimal_l max: {np.max(np.exp(optimal_l))}")
    print(f"optimal_r min: {np.min(np.exp(optimal_r))}")
    print(f"optimal_r max: {np.max(np.exp(optimal_r))}")

if __name__ == "__main__":
    main()
