import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from entropy import minimize_energy, N, find_luminance_chrominance

EPSILON = 1e-6 

def main():
    image_path = '../data/'
    src_name = 'IMG_1167'
    mask_name = 'shadow_mask'
    jpg = '.jpg'
    png='.png'

    image_color = cv2.imread( image_path + src_name + jpg )
    _, chrom = find_luminance_chrominance( image_color )
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) 
    image = image_gray
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

    optimal_l = minimize_energy(image, mask_gray, initial_l, \
                                phi_l=phi_l, phi_p=phi_p, \
                                omega_t=omega_t, omega_p=omega_p, \
                                img_name=src_name, num_iter=30,
                                chrominance=chrom).reshape((image.shape[0], image.shape[1]))
    image_gray = image_gray.astype(np.int64)
    optimal_r = np.log(image_gray + EPSILON).astype(np.int64) - optimal_l.astype(np.int64)
    optimal_img = np.multiply(np.exp(optimal_l), np.exp(optimal_r)).astype(np.uint8)
    plt.imshow(optimal_img, cmap='gray')
    plt.show()
    max_l_star = np.max(optimal_img)
    # plt.imshow(np.flip(chrom, axis=2))
    plt.imshow(np.multiply(np.flip(chrom, axis=2), np.stack([optimal_img] * 3, axis=2) / max_l_star))
    plt.show()
    # cv2.imshow("i", optimal_img)
    # cv2.waitKey(0)
    print("--------------OPTIMAL--------------")
    print(f"optimal_l min: {np.min(np.exp(optimal_l))}")
    print(f"optimal_l max: {np.max(np.exp(optimal_l))}")
    print(f"optimal_r min: {np.min(np.exp(optimal_r))}")
    print(f"optimal_r max: {np.max(np.exp(optimal_r))}")




if __name__ == "__main__":
    main()
