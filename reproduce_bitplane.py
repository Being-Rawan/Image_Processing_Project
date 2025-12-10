
import numpy as np
from PIL import Image
import member6_histogram_bitplane as m6
import io

def test_bitplane():
    # Create a dummy image
    img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    try:
        encoded = m6.bitplane_encode(img)
        print("Encoding successful, size:", len(encoded))
    except Exception as e:
        print("Encoding failed:", e)

if __name__ == "__main__":
    test_bitplane()
