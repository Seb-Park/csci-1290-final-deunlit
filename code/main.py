import numpy as np
import cv2
import matplotlib.pyplot as plt
from entropy import minimize_energy


def main():
    image = cv2.imread('path_to_image.jpg')
    initial_l = np.zeros_like(image)
    lambda_reg = 1.0 

    optimal_l = minimize_energy(image, initial_l, lambda_reg)



if __name__ == "__main__":
    main()
