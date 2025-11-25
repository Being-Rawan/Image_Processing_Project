import numpy as np
import pywt
import pickle
import cv2

def wavelet_encode(data: bytes) -> bytes:
    """
    Input: Raw image bytes (e.g., from reading a PNG/JPG file).
    Output: Compressed bytes (pickled wavelet coefficients).
    """
    # 1. Decode raw bytes into a NumPy array using OpenCV
    # np.frombuffer converts bytes string to a 1D array
    file_bytes = np.frombuffer(data, np.uint8)
    # cv2.imdecode turns that 1D array into an image matrix (H, W, C)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Could not decode image bytes. Ensure input is a valid image format.")

    # 2. Prepare for Wavelet Transform
    # OpenCV loads color as BGR by default. We process channels independently, 
    # so order doesn't strictly matter for compression, but we maintain consistency.
    wavelet_type = 'haar'
    level = 1
    
    coeffs = {}
    coeffs['shape'] = image.shape
    
    # Handle Grayscale (2D) vs Color (3D)
    if len(image.shape) == 2:
        coeffs['mode'] = 'gray'
        coeffs['data'] = pywt.wavedec2(image, wavelet_type, level=level)
        
    elif len(image.shape) == 3:
        coeffs['mode'] = 'color'
        # Split channels and transform each independently
        channels = cv2.split(image)
        coeffs['data'] = [pywt.wavedec2(c, wavelet_type, level=level) for c in channels]
        
    return pickle.dumps(coeffs)

def wavelet_decode(comp: bytes) -> bytes:
    """
    Input: Compressed bytes (from wavelet_encode).
    Output: Reconstructed raw image bytes (encoded as PNG).
    """
    try:
        packet = pickle.loads(comp)
    except Exception as e:
        raise ValueError(f"Failed to load compressed data: {e}")
        
    mode = packet['mode']
    wavelet_type = 'haar'
    
    reconstructed_image = None

    if mode == 'gray':
        # Inverse Transform
        rec = pywt.waverec2(packet['data'], wavelet_type)
        reconstructed_image = rec
        
    elif mode == 'color':
        # Inverse Transform for each channel
        rec_channels = [pywt.waverec2(c, wavelet_type) for c in packet['data']]
        # Merge channels back
        reconstructed_image = cv2.merge(rec_channels)

    # Post-processing: Clip to valid pixel range and cast to uint8
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    
    # Encode back to bytes (PNG format is lossless and safe)
    success, encoded_image = cv2.imencode('.png', reconstructed_image)
    
    if not success:
        raise ValueError("Could not encode reconstructed image to bytes.")
        
    return encoded_image.tobytes()