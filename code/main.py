import numpy as np
import cv2
import matplotlib.pyplot as plt
from entropy import minimize_energy, EPSILON, N
from utils import average_variance_of_patches

def main():
    img_name = "IMG_1167.jpg"
    mask_name = "mask_1167.png"

    image = cv2.imread(f'../data/{img_name}')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = image_gray

    # cv2.imshow("i0", image_gray)
    mask = cv2.imread(f'../data/{mask_name}')
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    initial_l = np.ones((image.shape[0]*image.shape[1], 1)) # in log domain
    average_variance_patch = average_variance_of_patches(image)
    phi_l = np.array([[0.2]]) 
    phi_p = np.array([[0.1, 0.0], [0.0, 0.1]])
    omega_t = np.diag([average_variance_patch]*N)
    omega_p = np.array([[30, 0.0], [0.0, 30]])
    lambda_reg = 1.0 

    optimal_l = minimize_energy(image, mask_gray, img_name, initial_l, phi_l=phi_l, phi_p=phi_p, omega_t=omega_t, omega_p=omega_p).reshape((image.shape[0], image.shape[1]))
    image_gray = image_gray.astype(np.int64)
    optimal_r = np.log(image_gray + EPSILON).astype(np.int64) - optimal_l.astype(np.int64)
    optimal_img = np.multiply(np.exp(optimal_l), np.exp(optimal_r)).astype(np.uint8)
    # cv2.imshow("i", optimal_img)
    # cv2.waitKey(0)
    print("--------------OPTIMAL--------------")
    print(f"optimal_l min: {np.min(np.exp(optimal_l))}")
    print(f"optimal_l max: {np.max(np.exp(optimal_l))}")
    print(f"optimal_r min: {np.min(np.exp(optimal_r))}")
    print(f"optimal_r max: {np.max(np.exp(optimal_r))}")




if __name__ == "__main__":
    main()
