import cv2
import numpy as np
from PIL import Image
from collections import Counter
import struct
import pickle




def resize_nearest(img_array, new_w, new_h):
    return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def resize_bilinear(img_array, new_w, new_h):
    return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def resize_bicubic(img_array, new_w, new_h):
    return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)





def ensure_grayscale(image):
    arr = np.array(image)
    if arr.ndim == 3:
        
        arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    return arr.astype(np.uint8)




def arithmetic_encode(image):
    """
    Integer-based arithmetic coding with rescaling
    RETURNS:
        compressed_bytes (pickled dict)
    """
    gray = ensure_grayscale(image)
    flat = gray.flatten()
    
    # Calculate probabilities
    freq = Counter(flat)
    total = len(flat)
    probs = {s: freq[s] / total for s in freq}
    
    # Build cumulative distribution with integer precision
    PRECISION = 2**32  # Use 32-bit precision
    cumulative = {}
    cum_count = 0
    
    for s in sorted(probs.keys()):
        count = int(probs[s] * PRECISION)
        cumulative[s] = (cum_count, cum_count + count)
        cum_count += count
    
    # Ensure last symbol reaches PRECISION
    last_sym = max(cumulative.keys())
    cumulative[last_sym] = (cumulative[last_sym][0], PRECISION)
    
    # Arithmetic coding with rescaling
    low = 0
    high = PRECISION - 1
    output_bits = []
    pending_bits = 0
    
    HALF = PRECISION // 2
    QUARTER = PRECISION // 4
    THREE_QUARTERS = 3 * QUARTER
    
    def output_bit(bit):
        nonlocal pending_bits
        output_bits.append(bit)
        # Output pending bits (opposite of current bit)
        for _ in range(pending_bits):
            output_bits.append(1 - bit)
        pending_bits = 0
    
    for pixel in flat:
        # Narrow the range
        range_size = high - low + 1
        c_low, c_high = cumulative[pixel]
        
        high = low + (range_size * c_high) // PRECISION - 1
        low = low + (range_size * c_low) // PRECISION
        
        # Rescaling to prevent underflow
        while True:
            if high < HALF:
                # Both in lower half
                output_bit(0)
                low = 2 * low
                high = 2 * high + 1
            elif low >= HALF:
                # Both in upper half
                output_bit(1)
                low = 2 * (low - HALF)
                high = 2 * (high - HALF) + 1
            elif low >= QUARTER and high < THREE_QUARTERS:
                # Straddle middle - E3 scaling
                pending_bits += 1
                low = 2 * (low - QUARTER)
                high = 2 * (high - QUARTER) + 1
            else:
                break
    
    # Output final bits
    pending_bits += 1
    if low < QUARTER:
        output_bit(0)
    else:
        output_bit(1)
    
    # Convert bits to bytes
    bit_string = ''.join(map(str, output_bits))
    # Pad to byte boundary
    while len(bit_string) % 8 != 0:
        bit_string += '0'
    
    compressed_bytes = bytes([int(bit_string[i:i+8], 2) 
                              for i in range(0, len(bit_string), 8)])
    
    data_dict = {
        "width": gray.shape[1],
        "height": gray.shape[0],
        "compressed": compressed_bytes,
        "num_bits": len(output_bits),
        "probs": probs
    }
    return pickle.dumps(data_dict)


def arithmetic_decode(data_bytes):
    """
    Decode integer-based arithmetic coding
    RETURNS:
        original_bytes (flattened image bytes)
    """
    data = pickle.loads(data_bytes)
    W = data["width"]
    H = data["height"]
    compressed = data["compressed"]
    num_bits = data["num_bits"]
    probs = data["probs"]
    total_pixels = W * H
    
    # Rebuild cumulative distribution
    PRECISION = 2**32
    cumulative = {}
    cum_count = 0
    
    for s in sorted(probs.keys()):
        count = int(probs[s] * PRECISION)
        cumulative[s] = (cum_count, cum_count + count)
        cum_count += count
    
    last_sym = max(cumulative.keys())
    cumulative[last_sym] = (cumulative[last_sym][0], PRECISION)
    
    # Convert bytes back to bits
    bit_string = ''.join(format(byte, '08b') for byte in compressed)
    bit_string = bit_string[:num_bits]  # Trim padding
    
    # Initialize decoder
    HALF = PRECISION // 2
    QUARTER = PRECISION // 4
    THREE_QUARTERS = 3 * QUARTER
    
    low = 0
    high = PRECISION - 1
    value = 0
    
    # Read first 32 bits into value
    for i in range(min(32, len(bit_string))):
        value = (value << 1) | int(bit_string[i])
    
    bit_index = 32
    decoded = []
    
    for _ in range(total_pixels):
        # Find symbol
        range_size = high - low + 1
        
        # Clamp scaled_value to valid range
        if range_size <= 0:
            range_size = 1
        
        scaled_value = ((value - low + 1) * PRECISION - 1) // range_size
        scaled_value = max(0, min(PRECISION - 1, scaled_value))
        
        # Binary search for symbol (faster and more robust)
        symbol = None
        symbols_list = sorted(cumulative.keys())
        
        for s in symbols_list:
            c_low, c_high = cumulative[s]
            if c_low <= scaled_value < c_high:
                symbol = s
                break
        
        # Fallback: if no match found, use last symbol
        if symbol is None:
            symbol = symbols_list[-1]
        
        decoded.append(symbol)
        
        # Update range
        c_low, c_high = cumulative[symbol]
        
        # Ensure valid range calculations
        new_high = low + (range_size * c_high) // PRECISION - 1
        new_low = low + (range_size * c_low) // PRECISION
        
        # Prevent range collapse
        if new_high <= new_low:
            new_high = new_low + 1
        
        high = new_high
        low = new_low
        
        # Rescaling
        rescale_count = 0
        while rescale_count < 100:  # Safety limit to prevent infinite loops
            if high < HALF:
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value
            elif low >= HALF:
                low = 2 * (low - HALF)
                high = 2 * (high - HALF) + 1
                value = 2 * (value - HALF)
            elif low >= QUARTER and high < THREE_QUARTERS:
                low = 2 * (low - QUARTER)
                high = 2 * (high - QUARTER) + 1
                value = 2 * (value - QUARTER)
            else:
                break
            
            # Read next bit
            if bit_index < len(bit_string):
                value += int(bit_string[bit_index])
                bit_index += 1
            
            value &= (PRECISION - 1)  # Keep in range
            rescale_count += 1
    
    dec_arr = np.array(decoded, dtype=np.uint8)
    return dec_arr.reshape((H, W)).tobytes()