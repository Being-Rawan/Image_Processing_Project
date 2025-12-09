# member2_gray_binary_huffman.py
# Member 2 - Grayscale + Binary + Huffman Coding

import numpy as np
import cv2
from collections import Counter
import struct
import heapq
from typing import Tuple
from PIL import Image

# ------------------------------------------------------------
# 1. Grayscale conversion (using cv2 - super simple)
# ------------------------------------------------------------
def to_grayscale(img_array: np.ndarray) -> np.ndarray:
    if len(img_array.shape)==2:#check if already grayscale
        return img_array
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) #Y=0.299R+0.587G+0.114B
    return gray  # shape (H, W), uint8

# ------------------------------------------------------------
# 2. Binary with mean threshold + comment
# ------------------------------------------------------------
def to_binary_with_mean_threshold(img_array: np.ndarray, maxval:float) -> Tuple[np.ndarray, str]:
    gray = to_grayscale(img_array)
    mean_val = int(np.mean(gray))
    _, binary = cv2.threshold(gray, mean_val, maxval, cv2.THRESH_BINARY)

    # Simple quality comment (same logic as before, students love it)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peak_ratio = np.max(hist) / gray.size

    if peak_ratio > 0.25:
        comment = "good (strong peak around threshold)"
    elif peak_ratio > 0.15:
        comment = "acceptable (moderate separation)"
    else:
        comment = "poor (flat distribution, threshold may be arbitrary)"

    return binary, f"Threshold = {mean_val}; {comment}"

# ------------------------------------------------------------
# 3. Huffman coding (same reliable code, unchanged)
# ------------------------------------------------------------
class _HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encode(data: Image) -> bytes:
    if not data:
        return b''

    data= np.array(data).tobytes()
    # EX ==> data = [100, 100, 100, 50] ===>  freqs = {100: 3, 50: 1}

    freqs = Counter(data)
    # Priority queue of leaf nodes
    pq = [_HuffmanNode(freq, sym) for sym, freq in freqs.items()]
    heapq.heapify(pq)

    # Build tree (handle single-symbol implicitly)
    if len(pq) == 1:
        # Keep the single node as root (leaf)
        root = pq[0]
    else:
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            heapq.heappush(pq, _HuffmanNode(left.freq + right.freq, left=left, right=right))
        root = pq[0]

    # Generate codes
    codes = {}
    def generate(node, code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code or "0"
            return
        generate(node.left, code + "0")
        generate(node.right, code + "1")
    generate(root)

    # Encode bitstring
    # EX ==> If data = A A B C  ===> And codes = {A:"0", B:"10", C:"11"} == > Then: A A B C  →  "0 0 10 11" → "001011"
    bitstring = ''.join(codes[b] for b in data)
    padding = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + ('0' * padding)
    # Convert bitstring to real bytes ===> Every 8 bits → convert to integer 0–255:
    payload = bytearray(int(bitstring_padded[i:i+8], 2) for i in range(0, len(bitstring_padded), 8))

    # Header: number of unique symbols, (sym, freq)*, total_symbols, padding
    header = bytearray()
    header.extend(struct.pack('<I', len(freqs)))
    for sym, freq in freqs.items():
        header.append(sym)                 # sym is 0..255
        header.extend(struct.pack('<I', freq))
    total_symbols = sum(freqs.values())
    header.extend(struct.pack('<I', total_symbols))
    header.extend(struct.pack('<I', padding))

    return bytes(header + payload)

def huffman_decode(comp: bytes) -> bytes:
    if not comp:
        return b''

    pos = 0
    n = struct.unpack_from('<I', comp, pos)[0]
    pos += 4
    freqs = Counter()
    for _ in range(n):
        sym = comp[pos]
        pos += 1
        freq = struct.unpack_from('<I', comp, pos)[0]
        pos += 4
        freqs[sym] = freq

    total_symbols = struct.unpack_from('<I', comp, pos)[0]
    pos += 4
    padding = struct.unpack_from('<I', comp, pos)[0]
    pos += 4

    payload = comp[pos:]
    # Rebuild tree
    pq = [_HuffmanNode(freq, sym) for sym, freq in freqs.items()]
    heapq.heapify(pq)
    if len(pq) == 1:
        root = pq[0]
    else:
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            heapq.heappush(pq, _HuffmanNode(left.freq + right.freq, left=left, right=right))
        root = pq[0]

    # Convert payload to bits and remove padding
    bitstream = ''.join(f'{b:08b}' for b in payload)
    if padding:
        bitstream = bitstream[:-padding]

    # Decode until we've produced total_symbols bytes
    result = bytearray()
    node = root

    # Special-case: single-symbol tree (root is leaf)
    if root.symbol is not None:
        # Just repeat that symbol total_symbols times
        return bytes([root.symbol] * total_symbols)

    for bit in bitstream:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            if len(result) >= total_symbols:
                break
            node = root

    return bytes(result)

