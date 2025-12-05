"""
member6_histogram_bitplane.py
------------------------------------
Functions using OpenCV:

  compute_histogram(gray_image) → list[256]
  bitplane_encode(data: bytes)  → dict of bit-planes
  bitplane_decode(comp)         → bytes

"""

import cv2
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# 1) Histogram using cv2.calcHist
# -----------------------------------------------------------
def compute_histogram(gray_array):
    """
    gray_array: list or numpy array of grayscale values 0–255
    Returns list of 256 histogram counts.
    """
    arr = np.array(gray_array, dtype=np.uint8)

    hist = cv2.calcHist([arr], [0], None, [256], [0, 256])
    # Flatten to python list of ints
    return [int(x) for x in hist.flatten()]


# -----------------------------------------------------------
# 2) Bit-Plane Encoding
# -----------------------------------------------------------
def bitplane_encode(data: Image):
    arr = np.array(data).astype(np.uint8)
    planes = {}

    for bit in range(8):
        # 1. Isolate the bit
        # This results in values of either 0 or (2^bit)
        mask_val = 1 << bit
        isolated = cv2.bitwise_and(arr, mask_val)

        # 2. Compare to get a binary mask
        # cv2.compare returns 255 for True, 0 for False
        cmp_result = cv2.compare(isolated, 0, cv2.CMP_GT)

        #  Convert 255 to 1
        # We divide by 255 so the plane stores strictly 0 or 1
        plane_0_1 = cv2.divide(cmp_result, 255)

        planes[f"plane{bit}"] = plane_0_1.tobytes()

    return planes

# -----------------------------------------------------------
# 3) Bit-Plane Decoding
# -----------------------------------------------------------
def bitplane_decode(comp):
    length = len(comp["plane0"])
    out = np.zeros(length, dtype=np.uint8)

    for bit in range(8):
        # Load the plane (which is now strictly 0 or 1)
        plane = np.frombuffer(comp[f"plane{bit}"], dtype=np.uint8)

        # Multiply 1 by the bit weight (e.g., 1 * 128 = 128)
        # Since plane is uint8, we must ensure we don't overflow logic here,
        # but since values are 0 or 1, multiplying by 128 fits in uint8.
        shifted = cv2.multiply(plane, (1 << bit))

        # Add to the accumulator
        out = cv2.add(out, shifted)

    return out.tobytes()

