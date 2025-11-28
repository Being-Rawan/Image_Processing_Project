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

def predictive_encode(data: bytes) -> bytes:
    
    arr = np.frombuffer(data, dtype=np.uint8)
    
    
    residuals = np.zeros_like(arr, dtype=np.int16)
    residuals[0] = arr[0]
    residuals[1:] = arr[1:] - arr[:-1]
    
    
    residuals_bytes = (residuals + 256) % 256
    return residuals_bytes.astype(np.uint8).tobytes()


def predictive_decode(comp: bytes) -> bytes:
    
    residuals_bytes = np.frombuffer(comp, dtype=np.uint8)
    
    
    residuals = residuals_bytes.astype(np.int16)
    residuals[residuals > 127] -= 256
    
    
    arr = np.zeros_like(residuals, dtype=np.int16)
    arr[0] = residuals[0]
    for i in range(1, len(residuals)):
        arr[i] = arr[i-1] + residuals[i]
    
    
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.tobytes()