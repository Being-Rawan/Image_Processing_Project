
import sys
import numpy as np
import cv2
from PIL import Image
import pickle
import traceback

sys.path.append("d:/project image branch")

try:
    from member6_histogram_bitplane import bitplane_encode, bitplane_decode
except ImportError:
    print("Could not import member6_histogram_bitplane.")
    sys.exit(1)

def verify():
    print("Starting verification...")
    # Create a synthetic RGB image if we can't load the file
    width, height = 100, 100
    # Random RGB
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode='RGB')
    
    print("Encoding RGB image...")
    try:
        encoded = bitplane_encode(img)
        print(f"Encoded size: {len(encoded)} bytes")
    except Exception as e:
        print(f"FAILED to encode: {e}")
        traceback.print_exc()
        return

    print("Decoding...")
    try:
        decoded = bitplane_decode(encoded)
        if decoded is None:
            print("FAILED: decoded is None")
            return
            
        print(f"Decoded mode: {decoded.mode}")
        if decoded.mode != 'L':
            print(f"FAILED: Expected Grayscale (L), got {decoded.mode}")
            return
            
        dec_arr = np.array(decoded)
        # For comparison, we will convert the input 'arr' (RGB) to grayscale 'L' to compare
        gray_input = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        if dec_arr.shape != gray_input.shape:
             print(f"FAILED: Shape mismatch {dec_arr.shape} vs {gray_input.shape}")
             return
             
        # Check equality 
        # Lossy compression: we expect difference!
        if np.array_equal(gray_input, dec_arr):
            print("WARNING: Perfect reconstruction? We expected lossy compression!")
        else:
            diff = np.abs(gray_input.astype(int) - dec_arr.astype(int))
            max_diff = np.max(diff)
            print(f"SUCCESS: Lossy reconstruction confirmed. Max diff: {max_diff}")
            
            # Since we dropped 3 bits (val 1, 2, 4 -> max 7)
            if max_diff < 15: # slightly higher tolerance for RGB->Gray conversion diffs
                print("Difference is within expected range.")
            else:
                print(f"FAILED: Difference too large {max_diff}, expected <8.")
                
    except Exception as e:
        print(f"FAILED to decode: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    verify()
