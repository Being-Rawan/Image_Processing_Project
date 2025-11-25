import numpy as np
import math

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

# ==========================================
# PART 2: Golomb-Rice Coding (No changes needed here)
# ==========================================

def golomb_rice_encode(data, k=4):
    m = 2 ** k
    bitstream = ""
    for number in data:
        if number < 0: raise ValueError("Requires non-negative integers.")
        q = number // m
        r = number % m
        unary = "1" * q + "0"
        binary = format(r, f'0{k}b')
        bitstream += unary + binary
    return bitstream

def golomb_rice_decode(bitstream, k=4):
    m = 2 ** k
    data = []
    i = 0
    n = len(bitstream)
    while i < n:
        q = 0
        while i < n and bitstream[i] == '1':
            q += 1
            i += 1
        if i < n and bitstream[i] == '0':
            i += 1
        else: break
        if i + k > n: break
        r = int(bitstream[i : i+k], 2)
        i += k
        data.append(q * m + r)
    return data