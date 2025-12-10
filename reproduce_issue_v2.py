
import sys
import numpy as np
import cv2
from PIL import Image
import pickle
import traceback

# Import the module functions directly or copy them if needed. 
# Since I can import from the file system, I will try to add the path.
sys.path.append("d:/project image branch")

try:
    from member6_histogram_bitplane import bitplane_encode, bitplane_decode
except ImportError:
    print("Could not import member6_histogram_bitplane. Make sure the path is correct.")
    # Fallback: copy paste relevant parts for standalone reproduction if import fails
    # But for now, let's assume it works since we are in the same env (hopefully)

def test_on_image(image_path):
    print(f"Testing on {image_path}")
    try:
        img = Image.open(image_path)
        encoded_data = bitplane_encode(img)
        print(f"Encoded size: {len(encoded_data)} bytes")
        
        decoded_img = bitplane_decode(encoded_data)
        if decoded_img is None:
            print("Decoding failed (returned None)")
        else:
            print("Decoding successful")
            decoded_img.save("decoded_output.png")
            
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    # stored image path from metadata
    img_path = r"C:/Users/LENOVO/.gemini/antigravity/brain/91e414e8-85cc-4343-9bc5-315b631dc7de/uploaded_image_1765391039033.jpg"
    test_on_image(img_path)
