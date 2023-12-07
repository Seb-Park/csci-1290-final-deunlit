import numpy as np
import cv2
import matplotlib.pyplot as plt
from entropy import minimize_energy


def main():
    image = cv2.imread('IMG_1167.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    cv2.imshow("i0", image_gray)
    # image = 0.2989 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,0] 
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    initial_l = np.zeros_like(image)
    lambda_reg = 1.0 

    optimal_l = minimize_energy(image, initial_l, lambda_reg).reshape((image.shape[0], image.shape[1]))
    optimal_r = image_gray - optimal_l
    optimal_img = optimal_l * optimal_r
    cv2.imshow("i", optimal_img)
    cv2.waitKey(0)
    print("OPTIMAL")
    print(optimal_l)
    print(np.max(optimal_l))
    print(np.min(optimal_l))




if __name__ == "__main__":
    main()
