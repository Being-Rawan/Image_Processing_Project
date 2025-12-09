import cv2
import numpy as np
from PIL import Image
from member1_io_rle import rle_encode, rle_decode

from itertools import groupby

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

def predictive_encode(data: Image) -> bytes:
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
    
    # 1D view of the bytes
    arr = np.frombuffer(np.array(data, dtype=np.uint8), dtype=np.uint8)

    # Allocate residual array (same length)
    residuals = np.empty_like(arr, dtype=np.uint8)

    # First element is stored as-is
    residuals[0] = arr[0]

    # For uint8 in NumPy, subtraction already wraps modulo 256,
    # so this directly gives us (x[i] - x[i-1]) mod 256
    if arr.size > 1:
        residuals[1:] = arr[1:] - arr[:-1]

    #RLE encode
    # return rle_encode(residuals)

    # Use memoryview to avoid extra copies if data is a large bytes object
    mv = memoryview(residuals.tobytes())
    encoded = bytearray()
    append = encoded.append  # local binding for speed

    # groupby groups consecutive identical bytes efficiently in C
    for value, group in groupby(mv):
        # We need the run length; iterate once over the group
        run_length = 0
        for _ in group:
            run_length += 1

        # Split long runs into chunks of at most 255
        while run_length > 255:
            append(255)
            append(value)
            run_length -= 255

        # Remainder (1..255)
        append(run_length)
        append(value)

    return bytes(encoded)

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
    
    #RLE decode
    out= rle_decode(comp)

    # mv = memoryview(comp)
    # out = bytearray()
    # extend = out.extend  # local binding for speed

    # # Iterate over pairs: (count, value)
    # # Using step=2 avoids manual indexing logic
    # for i in range(0, len(mv), 2):
    #     count = mv[i]
    #     value = mv[i + 1]

    #     if count <= 0:
    #         # Defensive check; not strictly necessary if encoder is correct
    #         continue

    #     # Use list repetition + extend (fast in C)
    #     extend([value] * count)

    residuals = np.frombuffer(out, dtype=np.uint8)

    # Use uint16 for the cumulative sum to avoid overflow in NumPy,
    # then wrap back to 0..255 using bitwise AND with 0xFF.
    cumsum = np.cumsum(residuals.astype(np.uint16), dtype=np.uint16)
    reconstructed = (cumsum & 0xFF).astype(np.uint8)

    return reconstructed.tobytes()

