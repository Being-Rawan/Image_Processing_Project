# member5_crop_lzw.py
import numpy as np
from PIL import Image

def crop_image(img_array, x1, y1, x2, y2):
    """
    Crop an image array to the specified coordinates.

    Args:
        img_array: numpy array of the image (grayscale or RGB)
        x1, y1: top-left coordinates
        x2, y2: bottom-right coordinates

    Returns:
        Cropped image array
    """
    # Ensure coordinates are within bounds and properly ordered
    height, width = img_array.shape[:2]

    # Validate and clamp coordinates
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    # Perform cropping
    if len(img_array.shape) == 3:  # RGB image
        cropped = img_array[y1:y2, x1:x2, :]
    else:  # Grayscale image
        cropped = img_array[y1:y2, x1:x2]

    return cropped

def lzw_encode(data:Image):
    """
    LZW compression for image data.

    Args:
        data: bytes object containing image data

    Returns:
        Compressed data as list of integers (codes)
    """
    data= np.array(data).tobytes()
    if isinstance(data, (bytes, bytearray)):
        # Convert bytes to string for LZW encoding
        data_str = data.decode('latin-1')
    else:
        data_str = str(data)

    # Initialize dictionary with all possible single characters
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []

    for c in data_str:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # Output the code for w
            result.append(dictionary[w])
            # Add wc to the dictionary
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w
    if w:
        result.append(dictionary[w])

    return result

def lzw_decode(compressed_data):
    """
    LZW decompression for image data.

    Args:
        compressed_data: list of integers (LZW codes)

    Returns:
        Decompressed data as bytes
    """
    if not compressed_data:
        return b""

    # Initialize dictionary with all possible single characters
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}

    # Convert first code
    if isinstance(compressed_data, list) and compressed_data:
        w = chr(compressed_data[0])
        result = [w]

        for k in compressed_data[1:]:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[0]
            else:
                raise ValueError("Bad compressed k: %d" % k)

            result.append(entry)

            # Add w+entry[0] to the dictionary
            dictionary[dict_size] = w + entry[0]
            dict_size += 1

            w = entry
    else:
        # Handle case where compressed_data might already be decompressed
        return compressed_data if isinstance(compressed_data, bytes) else b""

    # Convert list of strings to single string, then to bytes
    decompressed_str = "".join(result)
    return decompressed_str.encode('latin-1')