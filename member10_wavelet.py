# member10_wavelet.py
"""
Moderate lossy wavelet encoder/decoder for the project (drop-in).
This version uses moderate quantization so you'll see a noticeable but not
harsh degradation after decompressing from the GUI.

How to tune (you don't need to change GUI):
- DEFAULT_QUANT_STEP = 12.0  -> moderate quantization (middle effect)
- DEFAULT_DETAIL_MULTIPLIER = 2.5 -> slightly stronger on detail bands
Smaller quant_step -> less loss; larger -> more loss.
"""

import pickle
import gzip
from typing import Any
from PIL import Image
import numpy as np
import pywt
import gzip

DEFAULT_QUANT_STEP = 16.0
DEFAULT_DETAIL_MULTIPLIER = 3.0
DEFAULT_WAVELET = "haar"
DEFAULT_LEVEL = 3  # Increased for better compression

def _quantize_coeffs(coeffs: Any, q: float, detail_mult: float) -> Any:
    if isinstance(coeffs, list) and coeffs and isinstance(coeffs[0], list):
        return [_quantize_coeffs(ch, q, detail_mult) for ch in coeffs]

    qcoeffs = []
    for item in coeffs:
        if isinstance(item, tuple):
            qd = q * detail_mult
            # Use int16 instead of int32 to save space
            qtuple = tuple(np.round(band / qd).astype(np.int16) for band in item)
            qcoeffs.append(qtuple)
        else:
            qcoeffs.append(np.round(item / q).astype(np.int16))
    return qcoeffs
def _dequantize_coeffs(qcoeffs: Any, q: float, detail_mult: float) -> Any:
    """
    Dequantize integer coeffs back to float arrays by multiplying with q (and q*detail_mult for details).
    """
    # Color case
    if isinstance(qcoeffs, list) and qcoeffs and isinstance(qcoeffs[0], list):
        return [_dequantize_coeffs(ch, q, detail_mult) for ch in qcoeffs]

    coeffs = []
    for item in qcoeffs:
        if isinstance(item, tuple):
            qd = q * detail_mult
            # Convert int16 back to float32
            coeffs.append(tuple(band.astype(np.float32) * qd for band in item))
        else:
            # Convert int16 back to float32
            coeffs.append(item.astype(np.float32) * q)
    return coeffs
def wavelet_encode(
    data: Image,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> bytes:
    arr = np.array(data)
    original_shape = arr.shape

    if arr.ndim == 2:
        mode = "gray"
        coeffs = pywt.wavedec2(arr.astype(np.float32), wavelet, level=level)
    elif arr.ndim == 3:
        mode = "color"
        coeffs = [pywt.wavedec2(arr[..., c].astype(np.float32), wavelet, level=level)
                  for c in range(arr.shape[2])]
    else:
        raise ValueError(f"Unsupported array rank: {arr.ndim}")

    q = float(DEFAULT_QUANT_STEP)
    dm = float(DEFAULT_DETAIL_MULTIPLIER)

    qcoeffs = _quantize_coeffs(coeffs, q, dm)

    # CRITICAL: Add gzip compression to exploit sparsity
    pickled = pickle.dumps(((mode, original_shape), qcoeffs, {
        "wavelet": wavelet, 
        "level": level, 
        "quant_step": q, 
        "detail_multiplier": dm
    }))
    
    return gzip.compress(pickled, compresslevel=9)

def wavelet_decode(compressed_data: bytes) -> bytes:
    """
    Decode packet produced by wavelet_encode and return raw image bytes (row-major).
    """
    if not compressed_data:
        raise ValueError("Empty compressed data")

    try:
        # CRITICAL: Decompress the gzip data FIRST
        pickled = gzip.decompress(compressed_data)
        shape_info, qcoeffs, meta = pickle.loads(pickled)
    except Exception as e:
        raise ValueError(f"Failed to decompress/unpickle: {e}")

    mode, original_shape = shape_info
    wavelet = meta.get("wavelet", DEFAULT_WAVELET)
    q = float(meta.get("quant_step", DEFAULT_QUANT_STEP))
    detail_mult = float(meta.get("detail_multiplier", DEFAULT_DETAIL_MULTIPLIER))

    # Dequantize (update to handle int16 instead of int32)
    coeffs = _dequantize_coeffs(qcoeffs, q, detail_mult)

    # Inverse wavelet reconstruction
    if mode == "gray":
        rec = pywt.waverec2(coeffs, wavelet)
        rec = np.asarray(rec)
    elif mode == "color":
        channels_recon = []
        for c_coeff in coeffs:
            c_rec = pywt.waverec2(c_coeff, wavelet)
            channels_recon.append(np.asarray(c_rec))
        min_h = min(ch.shape[0] for ch in channels_recon)
        min_w = min(ch.shape[1] for ch in channels_recon)
        channels_recon = [ch[:min_h, :min_w] for ch in channels_recon]
        rec = np.dstack(channels_recon)
    else:
        raise ValueError("Invalid mode in compressed packet")

    desired_h, desired_w = original_shape[0], original_shape[1]
    if rec.ndim == 2:
        rec = rec[:desired_h, :desired_w]
    else:
        rec = rec[:desired_h, :desired_w, :original_shape[2]]

    rec = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
    return rec.tobytes()