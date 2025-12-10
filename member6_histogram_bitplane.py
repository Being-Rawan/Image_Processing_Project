import cv2
import numpy as np
from PIL import Image
import pickle

# -----------------------------------------------------------
# Helper: Run-Length Encoding Logic
# -----------------------------------------------------------
def _rle_encode_helper(binary_mask):
    """Internal helper to compress using RLE"""
    flat = binary_mask.flatten()
    if len(flat) == 0:
        return 0, np.array([], dtype=np.uint32)
    
    # Find changes
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    
    # Calculate run lengths
    indices = np.concatenate(([0], change_indices, [len(flat)]))
    runs = np.diff(indices).astype(np.uint32)
    
    return int(flat[0]), runs

def _rle_decode_helper(start_val, runs, shape):
    """Internal helper to decompress RLE"""
    # Create alternating values [start, !start, start...]
    vals = np.array([(start_val + i) % 2 for i in range(len(runs))], dtype=np.uint8)
    # Repeat values by run length
    flat = np.repeat(vals, runs)
    return flat.reshape(shape)

# -----------------------------------------------------------
# 1) Histogram using cv2.calcHist
# -----------------------------------------------------------
def compute_histogram(gray_array):
    """
    gray_array: list or numpy array of grayscale values 0–255
    Returns list of 256 histogram counts.
    """
    arr = np.array(gray_array, dtype=np.uint8)
    hist = cv2.calcHist([arr], [0], None, [256], [0, 256])
    return [int(x) for x in hist.flatten()]

# -----------------------------------------------------------
# 2) Bit-Plane Encoding (Smart Hybrid Compression)
# -----------------------------------------------------------

def _bitplane_encode_channel(channel_array):
    """
    Helper: Encodes a single 2D grayscale/channel array (0-255).
    Returns a dictionary of planes for this channel.
    """
    height, width = channel_array.shape
    planes_data = {}
    
    # DISCARD lowest 3 bits (0, 1, 2) for lossy compression / noise
    for bit in range(3, 8):
        mask_val = 1 << bit
        isolated = cv2.bitwise_and(channel_array, mask_val)
        
        # Determine 0 or 1. optimized: (val > 0) -> 1
        # cv2.compare returns 255 for match, 0 for mismatch
        cmp_result = cv2.compare(isolated, 0, cv2.CMP_GT)
        plane_0_1 = (cmp_result // 255).astype(np.uint8)

        # Strategy A: RLE
        start_val, runs = _rle_encode_helper(plane_0_1)
        rle_size = runs.nbytes + 1 # +1 for start_val byte approximation
        
        # Strategy B: Packed Bits
        packed_bits = np.packbits(plane_0_1)
        packed_size = packed_bits.nbytes
        
        if rle_size < packed_size:
            planes_data[bit] = ("RLE", (start_val, runs))
        else:
            planes_data[bit] = ("PACKED", packed_bits)
            
    return planes_data

def _bitplane_decode_channel(planes_data, shape):
    """
    Helper: Decodes a single channel from planes data.
    """
    height, width = shape
    reconstructed = np.zeros(shape, dtype=np.uint8)
    
    for bit in range(8):
        if bit not in planes_data:
            continue
            
        mode, content = planes_data[bit]
        
        if mode == "RLE":
            start_val, runs = content
            # FIX: Force start_val to standard Python int to avoid numpy overflow warning on creation
            # and ensure it's valid 0/1 for checks, though _rle_decode_helper expects uint8 logic.
            # We pass it as int, helper will use it in list/array creation.
            start_val = int(start_val) & 1 # Ensure strictly 0 or 1
            
            plane_0_1 = _rle_decode_helper(start_val, runs, shape)
        
        elif mode == "PACKED":
            packed_bits = content
            unpacked = np.unpackbits(packed_bits)
            unpacked = unpacked[:height * width]
            plane_0_1 = unpacked.reshape(shape)
        else:
            continue
            
        # Add bit contribution
        shift_val = 1 << bit
        # plane_0_1 is 0/1. Multiply by shift_val (e.g. 1, 2, 4...)
        # We can use bitwise_or or simple addition since bits don't overlap
        plane_layer = ((plane_0_1 > 0).astype(np.uint8) * shift_val)
        reconstructed = cv2.bitwise_or(reconstructed, plane_layer)
        
    return reconstructed

def bitplane_encode(data: Image) -> bytes:
    # Check image mode
    img_mode = data.mode
    
    # We will support 'L' (grayscale) and 'RGB'
    # Convert others to RGB for simplicity if not L
    if img_mode != 'L' and img_mode != 'RGB':
        data = data.convert('RGB')
        img_mode = 'RGB'
        
    width, height = data.size
    
    compressed_structure = {
        "mode": img_mode,
        "size": (width, height), # Note: PIL size is (width, height)
        "channels": []
    }
    
    if img_mode == 'L':
        # Single channel
        arr = np.array(data, dtype=np.uint8)
        encoded_channel = _bitplane_encode_channel(arr)
        compressed_structure["channels"].append(encoded_channel)
    else:
        # RGB -> Split channels
        # array shape is (height, width, 3)
        arr = np.array(data, dtype=np.uint8)
        # Split into R, G, B planes
        # cv2.split or numpy slicing
        channels = [arr[:,:,0], arr[:,:,1], arr[:,:,2]]
        
        for ch_arr in channels:
            encoded_channel = _bitplane_encode_channel(ch_arr)
            compressed_structure["channels"].append(encoded_channel)

    return pickle.dumps(compressed_structure)

def bitplane_decode(data: bytes) -> Image:
    """
    Reconstruct image from bit-plane compressed data.
    """
    try:
        structure = pickle.loads(data)
    except Exception:
        return None
        
    # Check for legacy format (keys: "shape", "planes" at top level)
    if "planes" in structure and "shape" in structure:
        # Legacy grayscale fallback
        shape = structure["shape"]
        planes = structure["planes"]
        # Wrap as if it were a single channel decode
        arr = _bitplane_decode_channel(planes, shape)
        return Image.fromarray(arr)
        
    # New format
    img_mode = structure.get("mode", "L")
    width, height = structure.get("size", (0, 0))
    channels_data = structure.get("channels", [])
    
    shape = (height, width) # numpy shape
    
    decoded_channels = []
    for ch_data in channels_data:
        decoded_arr = _bitplane_decode_channel(ch_data, shape)
        decoded_channels.append(decoded_arr)
        
    if img_mode == 'L':
        if not decoded_channels:
            return None
        return Image.fromarray(decoded_channels[0])
    
    elif img_mode == 'RGB':
        if len(decoded_channels) < 3:
            return None
        # Merge R, G, B
        merged = np.dstack((decoded_channels[0], decoded_channels[1], decoded_channels[2]))
        # User requested GRAYSCALE return even if RGB
        return Image.fromarray(merged, mode='RGB').convert("L")
        
    return None
