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
# 2) Bit-Plane Encoding using cv2 bit operations
# -----------------------------------------------------------
def bitplane_encode(data: bytes):
    """
    Converts bytes → numpy uint8 → 8 bit-planes (each 0/1).
    Returns dict of 8 planes: plane0..plane7
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    planes = {}

    for bit in range(8):
        # Extract one bit using cv2 bitwise operations
        plane = cv2.bitwise_and(arr, 1 << bit)
        plane = cv2.compare(plane, 0, cv2.CMP_NE)  # convert to 0/1
        plane = plane.astype(np.uint8)
        planes[f"plane{bit}"] = plane.tobytes()

    return planes


# -----------------------------------------------------------
# 3) Bit-Plane Decoding using cv2.add and shifting
# -----------------------------------------------------------
def bitplane_decode(comp):
    """
    Reconstruct bytes from bit-planes {plane0..plane7}
    Each plane is stored as bytes containing 0 or 1.
    """
    length = len(comp["plane0"])
    out = np.zeros(length, dtype=np.uint8)

    for bit in range(8):
        plane = np.frombuffer(comp[f"plane{bit}"], dtype=np.uint8)
        # value = plane * (1 << bit)
        shifted = cv2.multiply(plane, (1 << bit))
        out = cv2.add(out, shifted)

    return out.tobytes()
