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
from typing import Any

from PIL import Image
import numpy as np
import pywt

# -----------------------
# MODERATE DEFAULTS (middle effect)
# -----------------------
DEFAULT_QUANT_STEP = 48.0        # moderate: 1=almost lossless, 12=moderate, 48+=strong
DEFAULT_DETAIL_MULTIPLIER = 3.0  # amplify quantization on detail bands (mildly)
DEFAULT_WAVELET = "haar"
DEFAULT_LEVEL = 1

# -----------------------
# QUANTIZATION HELPERS
# -----------------------
def _quantize_coeffs(coeffs: Any, q: float, detail_mult: float) -> Any:
    """
    Quantize pywt coeffs structure:
      coeffs: [cA, (cH,cV,cD), (cH,cV,cD), ...]
    For color: coeffs is a list of per-channel coeff lists.
    q: quant step for approximation
    detail_mult: multiplier applied to q for detail bands
    """
    # Color case (list of per-channel coeff lists)
    if isinstance(coeffs, list) and coeffs and isinstance(coeffs[0], list):
        return [_quantize_coeffs(ch, q, detail_mult) for ch in coeffs]

    qcoeffs = []
    for item in coeffs:
        if isinstance(item, tuple):
            # detail tuple (cH, cV, cD)
            qd = q * detail_mult
            qtuple = tuple(np.round(band / qd).astype(np.int32) for band in item)
            qcoeffs.append(qtuple)
        else:
            # approximation coefficients (2D array)
            qcoeffs.append(np.round(item / q).astype(np.int32))
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
            coeffs.append(tuple(band.astype(np.float32) * qd for band in item))
        else:
            coeffs.append(item.astype(np.float32) * q)
    return coeffs


# -----------------------
# ENCODE (LOSSY)
# -----------------------
def wavelet_encode(
    data: Image,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> bytes:
    """
    Lossy wavelet encode with moderate defaults.
    Input: image file bytes (png/jpg/bmp) or raw buffer.
    Output: pickled packet (shape_info, quantized_coeffs, meta)
    """
    arr = np.array(data)
    original_shape = arr.shape

    # Compute wavelet coefficients per channel or grayscale
    if arr.ndim == 2:
        mode = "gray"
        coeffs = pywt.wavedec2(arr.astype(np.float32), wavelet, level=level)
    elif arr.ndim == 3:
        mode = "color"
        coeffs = [pywt.wavedec2(arr[..., c].astype(np.float32), wavelet, level=level)
                  for c in range(arr.shape[2])]
    else:
        raise ValueError(f"Unsupported array rank: {arr.ndim}")

    # Use moderate quantization defaults
    q = float(DEFAULT_QUANT_STEP)
    dm = float(DEFAULT_DETAIL_MULTIPLIER)

    qcoeffs = _quantize_coeffs(coeffs, q, dm)

    packet = (
        (mode, original_shape),
        qcoeffs,
        {"wavelet": wavelet, "level": level, "quant_step": q, "detail_multiplier": dm},
    )
    return pickle.dumps(packet)


# -----------------------
# DECODE
# -----------------------
def wavelet_decode(compressed_data: bytes) -> bytes:
    """
    Decode packet produced by wavelet_encode and return raw image bytes (row-major).
    """
    if not compressed_data:
        raise ValueError("Empty compressed data")

    try:
        shape_info, qcoeffs, meta = pickle.loads(compressed_data)
    except Exception as e:
        raise ValueError(f"Failed to unpickle compressed data: {e}")

    mode, original_shape = shape_info
    wavelet = meta.get("wavelet", DEFAULT_WAVELET)
    q = float(meta.get("quant_step", DEFAULT_QUANT_STEP))
    detail_mult = float(meta.get("detail_multiplier", DEFAULT_DETAIL_MULTIPLIER))

    # Dequantize
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
        # align channel shapes and stack
        min_h = min(ch.shape[0] for ch in channels_recon)
        min_w = min(ch.shape[1] for ch in channels_recon)
        channels_recon = [ch[:min_h, :min_w] for ch in channels_recon]
        rec = np.dstack(channels_recon)
    else:
        raise ValueError("Invalid mode in compressed packet")

    # Crop/pad to original shape (important)
    desired_h, desired_w = original_shape[0], original_shape[1]
    if rec.ndim == 2:
        rec = rec[:desired_h, :desired_w]
    else:
        rec = rec[:desired_h, :desired_w, :original_shape[2]]

    rec = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
    return rec.tobytes()
