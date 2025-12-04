import cv2
import numpy as np


# ============================================================
# 1) LOW-PASS FILTERS
# ============================================================

def gaussian_filter_19x19(image):
    """
    Apply a 19x19 Gaussian blur with sigma = 3.

    Works for both grayscale and color images.
    """
    # OpenCV's GaussianBlur is highly optimized (C/C++) and
    # handles borders and multi-channel images internally.
    return cv2.GaussianBlur(image, (19, 19), 3)


def median_filter_7x7(image):
    """
    Apply a 7x7 median filter.

    Works for both grayscale and color images.
    """
    # Median filter is also implemented efficiently in OpenCV.
    return cv2.medianBlur(image, 7)


# ============================================================
# 2) PREDICTIVE CODING (bytes <-> bytes)
# ============================================================

def predictive_encode(data: bytes) -> bytes:
    """
    First-order predictive coding on a flat byte stream.

    For each position i in the original byte array x:
        res[0] = x[0]
        res[i] = (x[i] - x[i-1]) mod 256   for i >= 1

    The result (res) has the same length as the input and is
    returned as bytes.

    This is NOT a full compression method by itself; it transforms
    the data into residuals that are usually more suitable for
    entropy coders (e.g., Huffman, Arithmetic).
    """
    if not data:
        return b""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("predictive_encode expects a bytes-like object.")

    # 1D view of the bytes
    arr = np.frombuffer(data, dtype=np.uint8)

    # Allocate residual array (same length)
    residuals = np.empty_like(arr, dtype=np.uint8)

    # First element is stored as-is
    residuals[0] = arr[0]

    # For uint8 in NumPy, subtraction already wraps modulo 256,
    # so this directly gives us (x[i] - x[i-1]) mod 256
    if arr.size > 1:
        residuals[1:] = arr[1:] - arr[:-1]

    return residuals.tobytes()


def predictive_decode(comp: bytes) -> bytes:
    """
    Inverse of predictive_encode (perfectly lossless).

    Given residuals res (same format as produced by predictive_encode):
        x[0] = res[0]
        x[i] = (x[i-1] + res[i]) mod 256   for i >= 1

    We use a cumulative sum with modulo 256 to reconstruct exactly
    the original byte stream.
    """
    if not comp:
        return b""

    if not isinstance(comp, (bytes, bytearray, memoryview)):
        raise TypeError("predictive_decode expects a bytes-like object.")

    residuals = np.frombuffer(comp, dtype=np.uint8)

    # Use uint16 for the cumulative sum to avoid overflow in NumPy,
    # then wrap back to 0..255 using bitwise AND with 0xFF.
    cumsum = np.cumsum(residuals.astype(np.uint16), dtype=np.uint16)
    reconstructed = (cumsum & 0xFF).astype(np.uint8)

    return reconstructed.tobytes()
