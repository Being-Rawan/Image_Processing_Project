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
# GOLOMB-RICE (Manual Implementation)
# ==========================================

def manual_int_to_binary_string(number, k):
    """
    Converts a number to a binary string of fixed length k
    using bitwise operators, without using format() or bin().
    """
    binary_string = ""
    # We iterate from the most significant bit (k-1) down to 0
    for i in range(k - 1, -1, -1):
        # Shift the number right by i positions and check the last bit
        # Example: if number is 101 (5) and i is 2:
        # 101 >> 2 becomes 001. 001 & 1 is 1.
        bit = (number >> i) & 1
        
        if bit == 1:
            binary_string += "1"
        else:
            binary_string += "0"
            
    return binary_string

def manual_binary_string_to_int(bin_str):
    """
    Converts a binary string segment to an integer
    without using int(x, 2).
    """
    value = 0
    length = len(bin_str)
    
    for i in range(length):
        bit = bin_str[i]
        # If the character is '1', we add 2^(power) to the value
        # The power is (length - 1 - i)
        if bit == '1':
            power = length - 1 - i
            # (1 << power) is the same as 2^power
            value += (1 << power)
            
    return value

def golomb_rice_encode(data):
    """
    Encodes list of integers.
    Uses default K=4 (M=16) as per common assignments.
    """
    k = 4
    m = 1 << k  # Bitwise way to do 2^k
    bitstream = ""
    
    for x in data:
        # Standard Golomb requires non-negative integers
        if x < 0:
            x = abs(x) # Simple handling for this example
            
        # 1. Calculate Quotient (Unary part)
        # Shift right by k is equivalent to integer division by 2^k
        q = x >> k 
        
        # 2. Calculate Remainder (Binary part)
        # Bitwise AND with (m-1) is equivalent to modulo m
        r = x & (m - 1)
        
        # 3. Construct Unary Code (q ones followed by a zero)
        # We use a loop to avoid string multiplication built-ins if needed,
        # but string mult is usually acceptable. Let's be manual:
        for _ in range(q):
            bitstream += "1"
        bitstream += "0"
        
        # 4. Construct Binary Code manually
        bitstream += manual_int_to_binary_string(r, k)
        
    return bitstream

def golomb_rice_decode(bitstream):
    """
    Decodes bitstring back to integers.
    """
    k = 4
    data = []
    i = 0
    total_len = len(bitstream)
    
    while i < total_len:
        # 1. Read Unary
        q = 0
        while i < total_len:
            char = bitstream[i]
            i += 1
            if char == '0':
                break 
            elif char == '1':
                q += 1
        
        # 2. Read Binary
        if i + k > total_len:
            break
            
        remainder_bits = ""
        for _ in range(k):
            remainder_bits += bitstream[i]
            i += 1
            
        r = manual_binary_string_to_int(remainder_bits)
        
        # 3. Reconstruct
        value = (q << k) + r
        data.append(value)
        
    # ==========================================
    # التعديل هنا (IMPORTANT FIX)
    # ==========================================
    # Main app expects 'bytes' to feed into np.frombuffer.
    # We must convert the list of integers back to bytes.
    return bytes(data)
# ==========================================
# REGISTRATION
# ==========================================
# Assuming the register function exists in your main scope:
# register_compression_method("Golomb-Rice", golomb_rice_encode, golomb_rice_decode)

# ==========================================
# TEST BLOCK: CHECK COMPRESSION
# ==========================================
if __name__ == "__main__":
    import random

    # 1. تجهيز بيانات اختبار (Golomb-Rice يحب الأرقام الصغيرة)
    # سنحاكي بيانات "الفرق بين البكسلات" لأنها عادة تكون صغيرة
    # إذا جربت أرقام كبيرة جداً (مثل 200 و 255) قد يزيد الحجم بدلاً من أن ينقص
    test_data = [2, 5, 0, 1, 8, 3, 0, 12, 4, 1, 0, 1, 15] 
    
    print(f"--- Testing Golomb-Rice ---")
    print(f"Original Data: {test_data}")

    # 2. عملية التشفير
    encoded_bits = golomb_rice_encode(test_data)
    print(f"\nEncoded Bitstream: {encoded_bits}")
    
    # 3. حساب الأحجام للمقارنة
    # الحجم الأصلي: عدد الأرقام * 8 (بافتراض أن كل رقم يأخذ 8 بت - بايت واحد)
    original_size_bits = len(test_data) * 8
    compressed_size_bits = len(encoded_bits)
    
    print(f"\n--- Size Comparison ---")
    print(f"Original Size:   {original_size_bits} bits")
    print(f"Compressed Size: {compressed_size_bits} bits")
    
    if compressed_size_bits < original_size_bits:
        saved = original_size_bits - compressed_size_bits
        ratio = (saved / original_size_bits) * 100
        print(f"✅ COMPRESSION WORKING! Saved {saved} bits ({ratio:.2f}% smaller).")
    else:
        print(f"⚠️ EXPANSION DETECTED! The output is larger.")
        print("Note: Golomb-Rice expands large numbers. It works best on small differences.")

    # 4. عملية فك التشفير والتأكد من الصحة
    decoded_data = golomb_rice_decode(encoded_bits)
    
    print(f"\n--- Integrity Check ---")
    if test_data == decoded_data:
        print("✅ SUCCESS: Decoded data matches original exactly.")
    else:
        print("❌ FAILED: Data mismatch!")
        print(f"Decoded: {decoded_data}")