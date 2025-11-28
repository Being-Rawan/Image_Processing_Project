import cv2
import numpy as np
import pickle
import zlib



def laplacian_filter(img_array: np.ndarray):
    """
    Apply Laplacian filter for edge detection.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        Filtered image with detected edges
    """
    # grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Laplacian kernel
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)
    
    # 2d filter
    laplacian = cv2.filter2D(gray, -1, laplacian_kernel)
    
    # abs values for clear visualization
    laplacian_abs = np.absolute(laplacian)
    laplacian_8u = np.uint8(laplacian_abs)
    
    return laplacian_8u


def sobel_x(img_array: np.ndarray):
    """
    Apply Sobel X filter for horizontal gradient detection.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        Sobel X filtered image (horizontal edges)
    """
    # grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    #  sobel x kernel
    sobel_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    # sobel x filter
    sobel_x = cv2.filter2D(gray, cv2.CV_32F, sobel_x_kernel)
    
    # abs values for clear visualization
    sobel_x_abs = np.absolute(sobel_x)
    sobel_x_8u = np.uint8(sobel_x_abs)
    
    return sobel_x_8u


def sobel_y(img_array: np.ndarray):
    """
    Apply Sobel Y filter for vertical gradient detection.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        Sobel Y filtered image (vertical edges)
    """
    # grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # sobel y kernel
    sobel_y_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    # sobel y filter
    sobel_y = cv2.filter2D(gray, cv2.CV_32F, sobel_y_kernel)
    
    # abs values for clear visualization
    sobel_y_abs = np.absolute(sobel_y)
    sobel_y_8u = np.uint8(sobel_y_abs)
    
    return sobel_y_8u


def gradient_magnitude(img_array: np.ndarray):
    """
    Calculate gradient magnitude using Sobel filters.
    Magnitude = sqrt(Sx² + Sy²)
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        Gradient magnitude image
    """
    # grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # sobel kernels
    sobel_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    sobel_y_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    # sobel filters
    sobel_x = cv2.filter2D(gray, cv2.CV_32F, sobel_x_kernel)
    sobel_y = cv2.filter2D(gray, cv2.CV_32F, sobel_y_kernel)
    
    # gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize and visualize
    magnitude = np.clip(magnitude, 0, 255)
    magnitude_8u = np.uint8(magnitude)
    
    return magnitude_8u



def dct_block_encode(data: bytes, original_shape: tuple, block_size: int = 8, quality: int = 50) -> bytes:
    """
    Implement block transform coding using DCT for encoding.
    
    Args:
        data: Image bytes
        original_shape: Original image shape
        block_size: Size of blocks (default: 8×8)
        quality: Compression quality (1-100)
    
    Returns:
        bytes: Compressed DCT coefficients as bytes
    """
    # array conversion through bytes with the image's original shape
    image_array = np.frombuffer(data, dtype=np.uint8).reshape(original_shape)
    
    # grayscale and color images conditions
    if len(original_shape) == 2:
        compressed_data = _dct_encode_grayscale_compressed(image_array, original_shape, block_size, quality)
    elif len(original_shape) == 3:
        compressed_data = _dct_encode_color_compressed(image_array, original_shape, block_size, quality)
    else:
        raise ValueError(f"Unsupported image shape: {original_shape}")
    
    # serialzation and compression
    serialized = pickle.dumps(compressed_data)
    return zlib.compress(serialized, level=9)

def _dct_encode_grayscale_compressed(image_array: np.ndarray, original_shape: tuple, block_size: int, quality: int) -> dict:
    height, width = original_shape
    
    # padding to have the image be divisible by block size
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    
    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width)), mode='edge')
    else:
        padded_image = image_array
    
    padded_height, padded_width = padded_image.shape
    blocks_y = padded_height // block_size
    blocks_x = padded_width // block_size
    
    # quantization matrix
    quant_matrix = _create_quantization_matrix(block_size, quality)
    
    # applying compression to each block
    compressed_blocks = []
    
    for i in range(blocks_y):
        for j in range(blocks_x):
            # extracted block
            block = padded_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_float = block.astype(np.float64) - 128
            
            # discrete cosine transform 
            dct_block = cv2.dct(block_float)
            
            # compression
            compressed_block = _compress_dct_block(dct_block, quant_matrix, quality)
            compressed_blocks.append(compressed_block)
    

    return {
        'compressed_blocks': compressed_blocks,
        'original_shape': original_shape,
        'padded_shape': padded_image.shape,
        'block_size': block_size,
        'quality': quality,
        'blocks_y': blocks_y,
        'blocks_x': blocks_x,
        'color': False
    }

def _dct_encode_color_compressed(image_array: np.ndarray, original_shape: tuple, block_size: int, quality: int) -> dict:
    height, width, channels = original_shape
    
    # padding to have the image be divisible by block size
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    
    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')
    else:
        padded_image = image_array
    
    padded_height, padded_width, channels = padded_image.shape
    blocks_y = padded_height // block_size
    blocks_x = padded_width // block_size
    
    # quantization matrix (use low quality for chroma channels)
    quant_matrix_luma = _create_quantization_matrix(block_size, quality)
    quant_matrix_chroma = _create_quantization_matrix(block_size, max(quality - 20, 10))
    
    # another looping block added in relevance to the grayscale because of the increase number of channels
    channel_blocks = []
    for channel in range(channels):
        channel_compressed_blocks = []
        # Use chroma quantization for color channels (1 and 2) and luma for channel 0
        quant_matrix = quant_matrix_chroma if channel > 0 else quant_matrix_luma
        
        for i in range(blocks_y):
            for j in range(blocks_x):
                # extracted block
                block = padded_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, channel]
                block_float = block.astype(np.float64) - 128
                
                # discrete cosine transform 
                dct_block = cv2.dct(block_float)
                
                # compression
                compressed_block = _compress_dct_block(dct_block, quant_matrix, quality)
                channel_compressed_blocks.append(compressed_block)
        
        channel_blocks.append(channel_compressed_blocks)
    
    return {
        'channel_blocks': channel_blocks,
        'original_shape': original_shape,
        'padded_shape': padded_image.shape,
        'block_size': block_size,
        'quality': quality,
        'blocks_y': blocks_y,
        'blocks_x': blocks_x,
        'channels': channels,
        'color': True
    }

def _compress_dct_block(dct_block: np.ndarray, quant_matrix: np.ndarray, quality: int) -> bytes:
    # quantization coefficients
    quantized = np.round(dct_block / quant_matrix)
    
    # coefficients to keep based on quality
    coeffs_to_keep = _get_coefficients_to_keep(quality, quantized.size)
    
    flattened = quantized.flatten()
    important_coeffs = flattened[:coeffs_to_keep]
    
    return important_coeffs.astype(np.int16).tobytes()

def _get_coefficients_to_keep(quality: int, total_coeffs: int) -> int:
    if quality >= 90:
        return total_coeffs
    elif quality >= 75:
        return total_coeffs * 3 // 4
    elif quality >= 50:
        return total_coeffs // 2
    elif quality >= 25:
        return total_coeffs // 4
    else:
        return total_coeffs // 8


def dct_block_decode(compressed_bytes: bytes) -> bytes:
    """
    Implement block transform coding using DCT for decoding.
    
    Args:
        compressed_bytes: Compressed DCT coefficients as bytes
    
    Returns:
        image_bytes
    """
    try:
        # decompress zlib then unpickle
        serialized_data = zlib.decompress(compressed_bytes)
        compressed_data = pickle.loads(serialized_data)
        
        if compressed_data['color']:
            image_array = _dct_decode_color_compressed(compressed_data)
        else:
            image_array = _dct_decode_grayscale_compressed(compressed_data)
        
        return image_array.tobytes()
        
    except (zlib.error, pickle.UnpicklingError) as e:
        raise ValueError(f"Decompression failed: {str(e)}")

def _dct_decode_grayscale_compressed(compressed_data: dict) -> np.ndarray:
    compressed_blocks = compressed_data['compressed_blocks']
    original_shape = compressed_data['original_shape']
    padded_shape = compressed_data['padded_shape']
    block_size = compressed_data['block_size']
    quality = compressed_data['quality']
    blocks_y = compressed_data['blocks_y']
    blocks_x = compressed_data['blocks_x']
    
    quant_matrix = _create_quantization_matrix(block_size, quality)
    reconstructed_image = np.zeros(padded_shape, dtype=np.float64)
    
    block_idx = 0
    for i in range(blocks_y):
        for j in range(blocks_x):
            compressed_block = compressed_blocks[block_idx]
            dequantized_block = _decompress_dct_block(compressed_block, block_size, quant_matrix)
            
            # inverse discrete cosine transform 
            idct_block = cv2.idct(dequantized_block)
            reconstructed_block = idct_block + 128
            
            reconstructed_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = reconstructed_block
            block_idx += 1
    
    # crop to original size
    height, width = original_shape
    reconstructed_cropped = reconstructed_image[:height, :width]
    return np.clip(reconstructed_cropped, 0, 255).astype(np.uint8)

def _dct_decode_color_compressed(compressed_data: dict) -> np.ndarray:
    channel_blocks = compressed_data['channel_blocks']
    original_shape = compressed_data['original_shape']
    padded_shape = compressed_data['padded_shape']
    block_size = compressed_data['block_size']
    quality = compressed_data['quality']
    blocks_y = compressed_data['blocks_y']
    blocks_x = compressed_data['blocks_x']
    channels = compressed_data['channels']
    
    quant_matrix_luma = _create_quantization_matrix(block_size, quality)
    quant_matrix_chroma = _create_quantization_matrix(block_size, max(quality - 20, 10))
    
    reconstructed_image = np.zeros(padded_shape, dtype=np.float64)
    
    for channel in range(channels):
        quant_matrix = quant_matrix_chroma if channel > 0 else quant_matrix_luma
        channel_compressed_blocks = channel_blocks[channel]
        block_idx = 0
        
        for i in range(blocks_y):
            for j in range(blocks_x):
                compressed_block = channel_compressed_blocks[block_idx]
                dequantized_block = _decompress_dct_block(compressed_block, block_size, quant_matrix)
                
                idct_block = cv2.idct(dequantized_block)
                reconstructed_block = idct_block + 128
                
                reconstructed_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, channel] = reconstructed_block
                block_idx += 1
    
    height, width, _ = original_shape
    reconstructed_cropped = reconstructed_image[:height, :width, :]
    return np.clip(reconstructed_cropped, 0, 255).astype(np.uint8)

def _decompress_dct_block(compressed_block: bytes, block_size: int, quant_matrix: np.ndarray) -> np.ndarray:
    # bytes to int16 array
    important_coeffs = np.frombuffer(compressed_block, dtype=np.int16).astype(np.float64)
    
    # full block with zeros for truncated coefficients
    full_coeffs = np.zeros(block_size * block_size, dtype=np.float64)
    full_coeffs[:len(important_coeffs)] = important_coeffs
    quantized_block = full_coeffs.reshape(block_size, block_size)
    
    # dequantization
    dequantized = quantized_block * quant_matrix
    
    return dequantized

def _create_quantization_matrix(block_size: int, quality: int) -> np.ndarray:
    if block_size == 8:
        std_quant_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)
    else:
        std_quant_matrix = np.ones((block_size, block_size), dtype=np.float64)
        for i in range(block_size):
            for j in range(block_size):
                std_quant_matrix[i, j] = 1 + (i + j) * 2
    
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100
    
    quant_matrix = std_quant_matrix
    return np.maximum(quant_matrix, 0.1)
