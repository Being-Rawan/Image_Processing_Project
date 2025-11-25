"""
member1_io_rle.py

Member 1 responsibilities:
- Load a color image from disk.
- Extract basic image info (width, height, file size, type).
- Implement Run-Length Encoding (RLE) compression and decompression for bytes.

This file is used by main_app.py via:
    from member1_io_rle import load_image, get_image_info, rle_encode, rle_decode
"""

import os
from typing import Tuple
from PIL import Image


# ============================================================
# IMAGE I/O
# ============================================================

def load_image(path: str) -> Image.Image:
    """
    Load an image from disk and return it as a PIL Image in RGB mode.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    PIL.Image.Image
        Image object in RGB mode.
    """
    img = Image.open(path).convert("RGB")
    return img


def get_image_info(path: str) -> Tuple[int, int, str, str]:
    """
    Get basic information about the image.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    (width, height, size_str, img_type)
        width    : int  - image width in pixels
        height   : int  - image height in pixels
        size_str : str  - file size as a human-readable string, e.g. "123.4 KB"
        img_type : str  - image format string, e.g. "JPEG", "PNG"
    """
    img = Image.open(path)
    width, height = img.size
    img_type = img.format if img.format is not None else "Unknown"

    try:
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024.0
        size_str = f"{size_kb:.1f} KB"
    except OSError:
        size_str = "Unknown size"

    return width, height, size_str, img_type


# ============================================================
# RLE COMPRESSION (bytes <-> bytes)
# ============================================================

def rle_encode(data: bytes) -> bytes:
    """
    Run-Length Encode a byte sequence.

    Scheme:
        Input  : [v, v, v, w, w, x, x, x, x, ...]
        Output : [count, v, count, w, count, x, ...]
    Where 'count' is stored in a single byte (1..255).

    Parameters
    ----------
    data : bytes
        Original data to be compressed.

    Returns
    -------
    bytes
        RLE-compressed data.
    """
    if not data:
        return b""

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rle_encode expects a bytes-like object.")

    encoded = bytearray()
    prev = data[0]
    count = 1

    for b in data[1:]:
        if b == prev and count < 255:
            count += 1
        else:
            encoded.append(count)
            encoded.append(prev)
            prev = b
            count = 1

    # flush last run
    encoded.append(count)
    encoded.append(prev)

    return bytes(encoded)


def rle_decode(comp: bytes) -> bytes:
    """
    Decode RLE-compressed data produced by rle_encode().

    Expects data in (count, value) pairs:
        [count, v, count, w, count, x, ...]

    Parameters
    ----------
    comp : bytes
        RLE-compressed data.

    Returns
    -------
    bytes
        Decompressed original data.
    """
    if not comp:
        return b""

    if not isinstance(comp, (bytes, bytearray)):
        raise TypeError("rle_decode expects a bytes-like object.")

    if len(comp) % 2 != 0:
        raise ValueError("Invalid RLE data length (must be even).")

    out = bytearray()

    # Iterate over pairs: (count, value)
    for i in range(0, len(comp), 2):
        count = comp[i]
        value = comp[i + 1]
        out.extend([value] * count)

    return bytes(out)

