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


def golomb_rice_encode(data: Image) -> bytes:
    """
    Properly packs bits into bytes instead of ASCII representation.
    """
    data = np.array(data).tobytes()
    k = 4
    m = 1 << k
    
    # Collect bits as a string first (for simplicity)
    bit_string = []
    
    for x in data:
        if x < 0: x = abs(x)
        q = x >> k
        r = x & (m - 1)
        
        # Unary part
        bit_string.append("1" * q + "0")
        # Binary part
        bit_string.append(format(r, f'0{k}b'))
    
    # Join all bits
    bits = ''.join(bit_string)
    
    # Pack bits into bytes
    return pack_bits_to_bytes(bits)

def pack_bits_to_bytes(bit_string: str) -> bytes:
    """
    Converts a string of '0's and '1's into packed bytes.
    Pads with zeros if necessary.
    """
    # Pad to multiple of 8
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding
    
    # Convert every 8 bits to a byte
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = int(bit_string[i:i+8], 2)
        byte_array.append(byte)
    
    # Store padding info in first byte (or use a header)
    result = bytes([padding]) + bytes(byte_array)
    return result

def golomb_rice_decode(bitstream: bytes) -> bytes:
    """
    Unpacks bytes back to bits, then decodes.
    """
    # First byte is padding info
    padding = bitstream[0]
    packed_data = bitstream[1:]
    
    # Unpack bytes to bit string
    bit_string = ''.join(format(byte, '08b') for byte in packed_data)
    
    # Remove padding
    if padding > 0:
        bit_string = bit_string[:-padding]
    
    # Now decode as before
    k = 4
    data = []
    i = 0
    total_len = len(bit_string)
    
    while i < total_len:
        # Read unary
        next_zero_index = bit_string.find('0', i)
        if next_zero_index == -1:
            break
        q = next_zero_index - i
        i = next_zero_index + 1
        
        # Read binary
        if i + k > total_len:
            break
        r = int(bit_string[i:i+k], 2)
        i += k
        
        # Reconstruct value
        value = (q << k) + r
        data.append(value)
    
    return bytes(data)