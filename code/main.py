import numpy as np
import cv2
import matplotlib.pyplot as plt
from entropy import minimize_energy
from skimage import img_as_float32

EPSILON = 1e-6 

def main():
    image = cv2.imread('../data/IMG_1167.jpg')
    # image = cv2.imread('test_shadow.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = image_gray
    cv2.imshow("i0", image_gray)

    initial_l = np.zeros((image.shape[0]*image.shape[1], 1))
    phi_l = np.array([0.2]) 
    phi_p = np.array([0.2])
    omega_t =  np.array([0.4])
    omega_p = np.array([0.5])
    lambda_reg = 1.0 

    optimal_l = minimize_energy(image, initial_l, phi_l=phi_l, phi_p=phi_p, omega_t=omega_t, omega_p=omega_p).reshape((image.shape[0], image.shape[1]))
    image_gray = image_gray.astype(np.int64)
    optimal_r = np.log(image_gray + EPSILON).astype(np.int64) - optimal_l.astype(np.int64)
    optimal_img = np.multiply(np.exp(optimal_l), np.exp(optimal_r)).astype(np.uint8)
    cv2.imshow("i", optimal_img)
    cv2.waitKey(0)
    print("OPTIMAL")
    print(np.min(np.exp(optimal_l)))
    print(np.max(np.exp(optimal_l)))
    print(np.min(np.exp(optimal_r)))
    print(np.max(np.exp(optimal_r)))




if __name__ == "__main__":
    main()
