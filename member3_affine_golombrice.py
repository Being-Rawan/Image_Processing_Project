import numpy as np
import math
from PIL import Image

# ==========================================
# PART 1: Affine Transformations (White Background)
# ==========================================

def get_pixel(img, x, y):
    """Helper to get pixel value with boundary checking."""
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return img[int(y), int(x)]
    # التعديل هنا: إرجاع 255 (أبيض) بدلاً من 0 (أسود) عند الخروج عن الحدود
    return 255

def translate(img_array, tx, ty):
    h, w = img_array.shape[:2]
    # التعديل: ملء المصفوفة بـ 255
    output = np.full_like(img_array, 255)

    for y_out in range(h):
        for x_out in range(w):
            x_in = x_out - tx
            y_in = y_out - ty
            output[y_out, x_out] = get_pixel(img_array, x_in, y_in)

    return output

def scale(img_array, sx, sy):
    h, w = img_array.shape[:2]
    new_w = int(w * sx)
    new_h = int(h * sy)

    # التعديل: ملء المصفوفة بـ 255 حسب نوع الصورة (ملونة أو رمادية)
    if len(img_array.shape) == 3:
        output = np.full((new_h, new_w, img_array.shape[2]), 255, dtype=img_array.dtype)
    else:
        output = np.full((new_h, new_w), 255, dtype=img_array.dtype)

    for y_out in range(new_h):
        for x_out in range(new_w):
            x_in = int(x_out / sx)
            y_in = int(y_out / sy)
            output[y_out, x_out] = get_pixel(img_array, x_in, y_in)

    return output

def rotate(img_array, angle_deg):
    h, w = img_array.shape[:2]
    cx, cy = w // 2, h // 2

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # التعديل: ملء المصفوفة بـ 255
    output = np.full_like(img_array, 255)

    for y_out in range(h):
        for x_out in range(w):
            x_shifted = x_out - cx
            y_shifted = y_out - cy

            x_in_shifted = x_shifted * cos_a + y_shifted * sin_a
            y_in_shifted = -x_shifted * sin_a + y_shifted * cos_a

            x_in = int(x_in_shifted + cx)
            y_in = int(y_in_shifted + cy)

            output[y_out, x_out] = get_pixel(img_array, x_in, y_in)

    return output

def shear_x(img_array, kx):
    h, w = img_array.shape[:2]
    new_w = int(w + h * abs(kx))

    # التعديل: ملء المصفوفة بـ 255
    if len(img_array.shape) == 3:
        output = np.full((h, new_w, img_array.shape[2]), 255, dtype=img_array.dtype)
    else:
        output = np.full((h, new_w), 255, dtype=img_array.dtype)

    x_offset = int(h * abs(kx)) if kx < 0 else 0

    for y_out in range(h):
        for x_out in range(new_w):
            x_in = int((x_out - x_offset) - (kx * y_out))
            y_in = y_out
            output[y_out, x_out] = get_pixel(img_array, x_in, y_in)

    return output

def shear_y(img_array, ky):
    h, w = img_array.shape[:2]
    new_h = int(h + w * abs(ky))

    # التعديل: ملء المصفوفة بـ 255
    if len(img_array.shape) == 3:
        output = np.full((new_h, w, img_array.shape[2]), 255, dtype=img_array.dtype)
    else:
        output = np.full((new_h, w), 255, dtype=img_array.dtype)

    y_offset = int(w * abs(ky)) if ky < 0 else 0

    for y_out in range(new_h):
        for x_out in range(w):
            x_in = x_out
            y_in = int((y_out - y_offset) - (ky * x_out))
            output[y_out, x_out] = get_pixel(img_array, x_in, y_in)

    return output


def golomb_rice_encode(data:Image)->bytes:
    """
    Optimized: Uses a list builder to avoid slow string concatenation.
    """
    data= np.array(data).tobytes()
    k = 4
    m = 1 << k

    # Use a list to store chunks. This is much faster than string +=
    parts = []

    # Pre-calculate formatting string to avoid f-string parsing in loop
    fmt = f'0{k}b'

    for x in data:
        if x < 0: x = abs(x)

        q = x >> k
        r = x & (m - 1)

        # Append parts to list
        # 1. Unary part ("1" repeated q times + "0")
        parts.append("1" * q + "0")

        # 2. Binary part
        parts.append(format(r, fmt))

    # Join all parts at once
    return "".join(parts)

def golomb_rice_decode(bitstream):
    """
    Optimized decoding that returns BYTES instead of a list.
    """
    k = 4
    data = []
    i = 0
    total_len = len(bitstream)

    while i < total_len:
        # 1. Read Unary (Count 1s) fast
        # Use .find() to jump to the next '0'
        next_zero_index = bitstream.find('0', i)

        if next_zero_index == -1:
            break # End of stream or malformed

        q = next_zero_index - i
        i = next_zero_index + 1 # Skip the '0' delimiter

        # 2. Read Binary
        if i + k > total_len:
            break

        # Slicing is fast
        binary_part = bitstream[i : i+k]
        r = int(binary_part, 2)

        i += k

        # 3. Reconstruct
        value = (q << k) + r
        data.append(value)

    # --- FIX IS HERE ---
    # Convert the list of integers [10, 255, 0...] into bytes b'\x0a\xff\x00...'
    return bytes(data)

