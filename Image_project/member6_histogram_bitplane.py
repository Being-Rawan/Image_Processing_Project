
from PIL import Image


# -----------------------------------------------------------
# 1) Manual Histogram Calculation
# -----------------------------------------------------------
def compute_histogram(gray_array):
    """
    Computes a histogram manually (256 bins)
    gray_array: iterable of integers [0..255]
    Returns: list length 256 with counts.
    """
    hist = [0] * 256

    for v in gray_array:
        if 0 <= v <= 255:
            hist[v] += 1
        else:
            raise ValueError("Input contains non-grayscale value.")

    return hist


# -----------------------------------------------------------
# 2) Bit-plane Encoding
# -----------------------------------------------------------
def bitplane_encode(data: bytes):
    """
    Splits each byte into 8 bit-planes.
    Returns a dictionary with keys plane0..plane7
    (plane7 = MSB, plane0 = LSB)
    Each plane is stored as a bytes object of 0/1 values.
    """
    n = len(data)
    planes = {f"plane{i}": bytearray(n) for i in range(8)}

    for idx, byte_val in enumerate(data):
        for bit in range(8):
            bit_value = (byte_val >> bit) & 1
            planes[f"plane{bit}"][idx] = bit_value

    # convert planes to immutable bytes
    return {k: bytes(v) for k, v in planes.items()}


# -----------------------------------------------------------
# 3) Bit-plane Decoding
# -----------------------------------------------------------
def bitplane_decode(comp):
    """
    Reconstructs original bytes from bit-plane dictionary.
    comp must contain 8 planes: plane0..plane7
    Returns: bytes
    """
    # Ensure valid planes
    for bit in range(8):
        if f"plane{bit}" not in comp:
            raise ValueError("Missing required bit-planes.")

    length = len(comp["plane0"])
    output = bytearray(length)

    for i in range(length):
        value = 0
        for bit in range(8):
            bit_val = comp[f"plane{bit}"][i] & 1
            value |= (bit_val << bit)
        output[i] = value

    return bytes(output)

'''
# -----------------------------------------------------------
# 4) Main Test Section
# -----------------------------------------------------------
def main():
    print("=== Bit-Plane + Histogram Test ===")

    # Load grayscale image
    img_path = "C:/Users/LENOVO/Downloads/IMG_5566 - Shahd Elganzoury.jpeg"
    print(f"Loading: {img_path}")

    img = Image.open(img_path).convert("L")
    gray_array = list(img.getdata())

    # --- histogram ---
    print("Computing histogram...")
    hist = compute_histogram(gray_array)
    print("Histogram[0..20] =", hist[:20], " ...")

    # --- bit-plane coding ---
    print("Encoding bit-planes...")
    data_bytes = bytes(gray_array)
    comp = bitplane_encode(data_bytes)

    print("Decoding...")
    restored = bitplane_decode(comp)

    # Verify reconstruction
    ok = (restored == data_bytes)
    print("Reconstruction OK:", ok)

    # Rebuild image and save
    if ok:
        out_img = Image.new("L", img.size)
        out_img.putdata(list(restored))
        out_img.save("reconstructed.png")
        print("Saved reconstructed image as: reconstructed.png")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
'''