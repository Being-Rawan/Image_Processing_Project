import numpy as np
import pywt
import pickle
import cv2
import math

# --- HELPER FUNCTION ---
def get_best_shape(total_pixels):
    """
    Guesses image dimensions (H, W) from a total pixel count.
    Prioritizes standard aspect ratios like 4:3 and 16:9.
    """
    sqrt_val = int(math.sqrt(total_pixels))
    best_shape = (sqrt_val, sqrt_val) # Default to square
    best_ratio_diff = float('inf')
    
    # Check widths starting from square root
    for w in range(sqrt_val, total_pixels + 1):
        if total_pixels % w == 0:
            h = total_pixels // w
            ratio = w / h
            
            # Look for ratios between 1.0 (square) and 2.5 (ultrawide)
            if 1.0 <= ratio <= 2.5:
                # Compare to common standards (4:3=1.33, 16:9=1.77)
                diff_4_3 = abs(ratio - (4/3))
                diff_16_9 = abs(ratio - (16/9))
                diff = min(diff_4_3, diff_16_9)
                
                if diff < best_ratio_diff:
                    best_ratio_diff = diff
                    best_shape = (h, w)
            
            # Optimization: Stop searching if we go too extreme
            if w > sqrt_val * 3:
                break
                
    return best_shape


# --- ENCODE FUNCTION ---
def wavelet_encode(data: bytes):
    image_array = None
    
   
    try:
        file_bytes = np.frombuffer(data, np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    except Exception:
        pass

    if image_array is None:
        total_bytes = len(data)
        
    
        if total_bytes % 3 == 0:
            total_pixels = total_bytes // 3
            h, w = get_best_shape(total_pixels)
            try:
    
                image_array = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
            except ValueError:
        
                s = int(math.sqrt(total_pixels))
                image_array = np.frombuffer(data, dtype=np.uint8)[:s*s*3].reshape((s, s, 3))
        else:
    
            h, w = get_best_shape(total_bytes)
            image_array = np.frombuffer(data, dtype=np.uint8).reshape((h, w))

    if image_array is None:
         raise ValueError(f"Could not decode image (Size: {len(data)}).")


    shape = image_array.shape
    if len(shape) == 2:  # Grayscale
        coeffs = pywt.wavedec2(image_array, 'haar', level=1)
        shape_info = ('gray', shape)
    elif len(shape) == 3:  # Color
        coeffs = []
        channels = shape[2]
        for i in range(channels):
            channel = image_array[:, :, i]
            c = pywt.wavedec2(channel, 'haar', level=1)
            coeffs.append(c)
        shape_info = ('color', shape)
    
    return pickle.dumps((shape_info, coeffs))


# --- DECODE FUNCTION ---
def wavelet_decode(compressed_data: bytes) -> bytes:
    if not compressed_data:
        raise ValueError("Input compressed data is empty.")

    try:
        shape_info, coeffs = pickle.loads(compressed_data)
    except Exception:
        raise ValueError("Failed to load compressed data.")
        
    mode, original_shape = shape_info
    

    if mode == 'gray':
        reconstructed = pywt.waverec2(coeffs, 'haar')
    elif mode == 'color':
        channels_data = []
        for c in coeffs:
            c_recon = pywt.waverec2(c, 'haar')
            channels_data.append(c_recon)
        reconstructed = np.dstack(channels_data)
        
    h, w = original_shape[:2]
    reconstructed = reconstructed[:h, :w]
    
    # Normalize pixel values
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    return reconstructed.tobytes()