import cv2
import numpy as np

def gaussian_filter_19x19(image):
    ksize = 19
    sigma = 3

    gaussian_kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

    blurred_image = cv2.filter2D(image, -1, gaussian_kernel_2d)

    return blurred_image

def median_filter_7x7(image):
    ksize = 7
    blurred_image = cv2.medianBlur(image, ksize)
    return blurred_image

def predictive_encode(image):

    h, w = image.shape
    residuals = np.zeros((h, w), dtype=np.int16)

    for i in range(h):
        for j in range(w):
            if j == 0:
                pred = 0
            else:
                pred = image[i, j-1]
            residuals[i, j] = int(image[i, j]) - int(pred)

    return residuals

def predictive_decode(residuals):
    
    h, w = residuals.shape
    image_rec = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if j == 0:
                pred = 0
            else:
                pred = image_rec[i, j-1]
            val = residuals[i, j] + pred
            val = np.clip(val, 0, 255)
            image_rec[i, j] = val

    return image_rec