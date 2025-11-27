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
from itertools import groupby
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
    # Use Pillow's optimized loader + convert once to RGB
    img = Image.open(path).convert("RGB")
    return img


def _format_file_size(size_bytes: int) -> str:
    """
    Convert file size in bytes to a human-readable string.
    (B, KB, MB, GB, ...)

    This is just for display in the GUI's Image Info.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_idx = 0

    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1

    if unit_idx == 0:
        return f"{int(size)} {units[unit_idx]}"
    else:
        return f"{size:.1f} {units[unit_idx]}"


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
        size_str : str  - file size as a human-readable string
        img_type : str  - image format string, e.g. "JPEG", "PNG"
    """
    img = Image.open(path)
    width, height = img.size
    img_type = img.format if img.format is not None else "Unknown"

    try:
        size_bytes = os.path.getsize(path)
        size_str = _format_file_size(size_bytes)
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

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("rle_encode expects a bytes-like object.")

    # Use memoryview to avoid extra copies if data is a large bytes object
    mv = memoryview(data)
    encoded = bytearray()
    append = encoded.append  # local binding for speed

    # groupby groups consecutive identical bytes efficiently in C
    for value, group in groupby(mv):
        # We need the run length; iterate once over the group
        run_length = 0
        for _ in group:
            run_length += 1

        # Split long runs into chunks of at most 255
        while run_length > 255:
            append(255)
            append(value)
            run_length -= 255

        # Remainder (1..255)
        append(run_length)
        append(value)

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

    if not isinstance(comp, (bytes, bytearray, memoryview)):
        raise TypeError("rle_decode expects a bytes-like object.")

    if len(comp) % 2 != 0:
        raise ValueError("Invalid RLE data length (must be even).")

    mv = memoryview(comp)
    out = bytearray()
    extend = out.extend  # local binding for speed

    # Iterate over pairs: (count, value)
    # Using step=2 avoids manual indexing logic
    for i in range(0, len(mv), 2):
        count = mv[i]
        value = mv[i + 1]

        if count <= 0:
            # Defensive check; not strictly necessary if encoder is correct
            continue

        # Use list repetition + extend (fast in C)
        extend([value] * count)

    return bytes(out)
