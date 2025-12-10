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
    RETURNS:
        compressed_bytes  (this is what main_app expects)
    """

    gray = ensure_grayscale(image)
    flat = gray.flatten()

  
    freq = Counter(flat)
    total = len(flat)
    probs = {s: freq[s] / total for s in freq}


    
    cumulative = {}
    csum = 0.0
    for s in sorted(probs.keys()):
        cumulative[s] = (csum, csum + probs[s])
        csum += probs[s]


    
    low = 0.0
    high = 1.0

    for pixel in flat:
        r = high - low

        
        if r < 1e-30:
            r = 1e-30

        c_low, c_high = cumulative[pixel]
        high = low + r * c_high
        low  = low + r * c_low

       
        if high - low < 1e-20:
            high = low + 1e-20

    encoded_value = (low + high) / 2.0


    
    data_dict = {
        "width": gray.shape[1],
        "height": gray.shape[0],
        "encoded": encoded_value,
        "probs": probs
    }

    return pickle.dumps(data_dict)




def arithmetic_decode(data_bytes):
    """
    RETURNS:
        original_bytes  (flattened image bytes)
    """

    data = pickle.loads(data_bytes)

    W = data["width"]
    H = data["height"]
    encoded_value = data["encoded"]
    probs = data["probs"]

    total_pixels = W * H


    
    cumulative = {}
    csum = 0.0
    for s in sorted(probs.keys()):
        cumulative[s] = (csum, csum + probs[s])
        csum += probs[s]

    symbols_sorted = list(sorted(cumulative.keys()))

    
    decoded = []
    value = encoded_value
    low = 0.0
    high = 1.0

    for _ in range(total_pixels):

        r = high - low

        if r < 1e-30:
            r = 1e-30

        scaled = (value - low) / r

        if scaled < 0.0: scaled = 0.0
        if scaled >= 1.0: scaled = 0.999999999999

        for sym in symbols_sorted:
            c_low, c_high = cumulative[sym]
            if c_low <= scaled < c_high:
                decoded.append(sym)

                high = low + r * c_high
                low  = low + r * c_low

                # Avoid collapse
                if high - low < 1e-20:
                    high = low + 1e-20

                break

   
    dec_arr = np.array(decoded, dtype=np.uint8)
    return dec_arr.reshape((H, W)).tobytes()
