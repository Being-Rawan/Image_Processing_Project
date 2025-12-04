# member9_highpass_dct.py
"""
Member 9: High-pass filters + DCT-block coding (compatible with main_app.py)

- Exposes:
    laplacian_filter(img_array)
    sobel_x(img_array)
    sobel_y(img_array)
    gradient_magnitude(img_array)
    dct_block_encode(data: bytes) -> bytes
    dct_block_decode(compressed_bytes: bytes) -> bytes

- dct_block_encode accepts raw image bytes (file bytes or arr.tobytes()).
  It attempts to decode standard image files (png/jpg/bmp) first; if that fails
  it will try to interpret the bytes as a raw buffer (RGB/RGBA/grayscale) by simple heuristics.
- The compressed packet stores the original shape and quality so decode can reconstruct
  image bytes with the exact byte count expected by main_app._bytes_to_image_array.

Tuning:
- DEFAULT_QUALITY: 10..100 (lower -> stronger loss). Default is chosen to be a
  little stronger than the middle value so you see noticeable but not extreme artifacts.
- BLOCK_SIZE: 8 (standard JPEG-style block).
"""

import cv2
import numpy as np
import pickle
import zlib
import math
from typing import Tuple, Dict, Any, Optional

# -------------------------
# CONFIG: change these values to tune visible loss
# - DEFAULT_QUALITY: integer 10..100. Lower -> more loss.
#   Recommended options:
#     80..100 -> nearly lossless / mild changes
#     60..79  -> mild noticeable loss
#     40..59  -> medium (middle) loss
#     30..39  -> medium+ (little higher than middle)  <-- recommended by user
#     10..29  -> strong artifacts
# - BLOCK_SIZE: 8 (keep)
# -------------------------
DEFAULT_QUALITY = 40   # set to 40 for "middle + a bit higher" visible change
BLOCK_SIZE = 8

# -------------------------
# Helper filters (these are fine)
# -------------------------
def laplacian_filter(img_array: np.ndarray) -> np.ndarray:
    """Return Laplacian edges (grayscale)."""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    lap = cv2.filter2D(gray, cv2.CV_32F, kernel)
    lap = np.absolute(lap)
    lap8 = np.clip(lap, 0, 255).astype(np.uint8)
    return lap8


def sobel_x(img_array: np.ndarray) -> np.ndarray:
    """Return Sobel X (horizontal) magnitude (grayscale)."""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob = np.absolute(sob)
    return np.clip(sob, 0, 255).astype(np.uint8)


def sobel_y(img_array: np.ndarray) -> np.ndarray:
    """Return Sobel Y (vertical) magnitude (grayscale)."""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    sob = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.absolute(sob)
    return np.clip(sob, 0, 255).astype(np.uint8)


def gradient_magnitude(img_array: np.ndarray) -> np.ndarray:
    """Return gradient magnitude sqrt(Sx^2 + Sy^2)."""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)


# -------------------------
# DCT block compression helpers
# -------------------------
def _safe_imdecode(data: bytes) -> Optional[np.ndarray]:
    """Try decoding file bytes via OpenCV."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def _guess_array_from_raw(data: bytes) -> np.ndarray:
    """
    Heuristic to try to interpret raw bytes as RGB/RGBA/grayscale array.
    If ambiguous, prefer RGB when length divisible by 3.
    """
    total = len(data)
    # prefer RGBA if divisible by 4 and yields a plausible rectangle
    for channels in (3, 4, 1):
        if channels == 1:
            pixels = total
        else:
            if total % channels != 0:
                continue
            pixels = total // channels

        s = int(math.isqrt(pixels))
        # if perfect square, use it; otherwise try to find divisor pair
        if s * s == pixels:
            h, w = s, s
        else:
            # try to find a reasonable divisor pair close to square
            h = s
            # find a divisor <= sqrt
            while h > 1 and pixels % h != 0:
                h -= 1
            if h == 1:
                continue
            w = pixels // h

        arr = np.frombuffer(data, dtype=np.uint8)[: (h * w * channels)]
        if channels == 1:
            arr = arr.reshape((h, w))
        else:
            arr = arr.reshape((h, w, channels))
        return arr

    # last resort: treat as 1D grayscale square by cropping to s*s
    s = int(math.isqrt(total))
    arr = np.frombuffer(data, dtype=np.uint8)[: (s * s)]
    return arr.reshape((s, s))


def _create_quant_matrix(block_size: int) -> np.ndarray:
    """Return standard 8x8 quant matrix (or simple generalization)."""
    if block_size == 8:
        std = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68,109,103, 77],
            [24, 35, 55, 64, 81,104,113, 92],
            [49, 64, 78, 87,103,121,120,101],
            [72, 92, 95, 98,112,100,103, 99]
        ], dtype=np.float64)
    else:
        std = np.ones((block_size, block_size), dtype=np.float64) * 16.0
        for i in range(block_size):
            for j in range(block_size):
                std[i,j] = 1 + (i + j) * 2.0
    return std


def _quality_scale(quality: int) -> float:
    """
    Map quality (10..100) to a scale factor for quant matrix.
    Lower quality -> larger scale -> more aggressive quantization.

    We'll use a smooth mapping:
      if quality < 50: scale = 50 / quality (stronger)
      else: scale = 1 + (100 - quality) / 50 (mild)
    """
    q = max(1, min(100, int(quality)))
    if q < 50:
        return max(1.0, 50.0 / q)
    else:
        return 1.0 + (100.0 - q) / 50.0


# -------------------------
# Encode / Decode (interfaces expected by main_app.py)
# -------------------------
def dct_block_encode(data: bytes) -> bytes:
    """
    Accepts bytes (either image file bytes or raw arr.tobytes()).
    Produces a compressed packet (zlib-compressed pickle) that includes:
      - 'color' flag (True if 3+ channels)
      - 'original_shape' (tuple)
      - 'block_size'
      - 'quality'
      - compressed blocks (list or per-channel lists)
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("dct_block_encode expects bytes-like input")

    # Try file decode first (PNG/JPG/BMP)
    img = _safe_imdecode(data)
    if img is None:
        # try to interpret as raw buffer
        img = _guess_array_from_raw(data)

    arr = np.asarray(img)
    original_shape = arr.shape

    quality = DEFAULT_QUALITY
    block_size = BLOCK_SIZE
    quant_base = _create_quant_matrix(block_size)
    qscale = _quality_scale(quality)
    quant_matrix = quant_base * qscale

    # helper to process 8x8 blocks for a single-channel array
    def process_channel(chan: np.ndarray):
        h, w = chan.shape
        pad_h = (block_size - (h % block_size)) % block_size
        pad_w = (block_size - (w % block_size)) % block_size
        chan_padded = np.pad(chan, ((0, pad_h), (0, pad_w)), mode='edge')
        ph, pw = chan_padded.shape
        blocks_y = ph // block_size
        blocks_x = pw // block_size

        blocks = []
        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = chan_padded[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size].astype(np.float64)
                block_shifted = block - 128.0
                dct_block = cv2.dct(block_shifted)
                # quantize
                qblock = np.round(dct_block / quant_matrix).astype(np.int16)
                # keep only top-left coefficients proportionally to quality
                coeffs_to_keep = _coeffs_to_keep_by_quality(quality, block_size*block_size)
                flat = qblock.flatten()
                kept = flat[:coeffs_to_keep].astype(np.int16)
                blocks.append(kept.tobytes())
        return {
            'blocks_y': blocks_y,
            'blocks_x': blocks_x,
            'padded_shape': chan_padded.shape,
            'blocks': blocks
        }

    if arr.ndim == 2:
        # grayscale
        channel_packet = process_channel(arr)
        packet = {
            'color': False,
            'original_shape': original_shape,
            'block_size': block_size,
            'quality': quality,
            'quant_matrix': quant_matrix,
            'channel_data': channel_packet
        }
    elif arr.ndim == 3:
        # color image: process each channel separately (assume BGR as cv2.decode returns)
        channels = []
        for c in range(arr.shape[2]):
            channels.append(process_channel(arr[..., c]))
        packet = {
            'color': True,
            'original_shape': original_shape,
            'block_size': block_size,
            'quality': quality,
            'quant_matrix': quant_matrix,
            'channel_data': channels
        }
    else:
        raise ValueError("Unsupported array rank for DCT encoding")

    serialized = pickle.dumps(packet)
    compressed = zlib.compress(serialized, level=6)  # balanced
    return compressed


def _coeffs_to_keep_by_quality(quality: int, total_coeffs: int) -> int:
    """Return how many coefficients to keep, based on quality."""
    q = max(1, min(100, int(quality)))
    if q >= 90:
        return total_coeffs
    if q >= 75:
        return total_coeffs * 3 // 4
    if q >= 60:
        return total_coeffs * 2 // 3
    if q >= 50:
        return total_coeffs // 2
    if q >= 40:
        return total_coeffs * 3 // 8
    if q >= 30:
        return total_coeffs // 4
    if q >= 20:
        return total_coeffs // 6
    return max(1, total_coeffs // 8)


def dct_block_decode(compressed_bytes: bytes) -> bytes:
    """
    Decompress the packet and reconstruct image bytes (raw row-major).
    Return value: bytes that match original image element count so main_app may reshape by current_array.shape.
    """
    try:
        serialized = zlib.decompress(compressed_bytes)
        packet = pickle.loads(serialized)
    except Exception as e:
        raise ValueError(f"Failed to decompress/deserialize DCT data: {e}")

    color = packet['color']
    original_shape = tuple(packet['original_shape'])
    block_size = int(packet['block_size'])
    quality = int(packet['quality'])
    quant_matrix = packet.get('quant_matrix')  # stored matrix (float)
    qscale = _quality_scale(quality)

    def rebuild_channel(ch_pkt):
        blocks_y = ch_pkt['blocks_y']
        blocks_x = ch_pkt['blocks_x']
        ph, pw = ch_pkt['padded_shape']
        reconstructed = np.zeros((ph, pw), dtype=np.float64)
        idx = 0
        for by in range(blocks_y):
            for bx in range(blocks_x):
                bbytes = ch_pkt['blocks'][idx]
                important = np.frombuffer(bbytes, dtype=np.int16).astype(np.float64)
                # fill into block (row-major). truncated entries are zeros
                full = np.zeros(block_size * block_size, dtype=np.float64)
                full[:len(important)] = important
                qblock = full.reshape((block_size, block_size))
                # dequantize
                # quant_matrix may be stored as ndarray; if not, rebuild
                qmat = quant_matrix if isinstance(quant_matrix, np.ndarray) else _create_quant_matrix(block_size) * qscale
                deq = qblock * qmat
                idct = cv2.idct(deq)
                block_rec = idct + 128.0
                y0, x0 = by*block_size, bx*block_size
                reconstructed[y0:y0+block_size, x0:x0+block_size] = block_rec
                idx += 1
        return reconstructed

    if not color:
        ch_pkt = packet['channel_data']
        rec_chan = rebuild_channel(ch_pkt)
        h, w = original_shape
        rec_crop = rec_chan[:h, :w]
        rec_uint8 = np.clip(rec_crop, 0, 255).astype(np.uint8)
        return rec_uint8.tobytes()
    else:
        channels_pkt = packet['channel_data']
        channels_rec = []
        for ch_pkt in channels_pkt:
            channels_rec.append(rebuild_channel(ch_pkt))
        # stack channels (they were processed independently)
        min_h = min(ch.shape[0] for ch in channels_rec)
        min_w = min(ch.shape[1] for ch in channels_rec)
        stacked = np.dstack([ch[:min_h, :min_w] for ch in channels_rec])
        # crop to original shape
        h, w, c = original_shape
        result = stacked[:h, :w, :c]
        rec_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        return rec_uint8.tobytes()
