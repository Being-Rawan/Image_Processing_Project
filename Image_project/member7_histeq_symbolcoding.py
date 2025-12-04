import numpy as np
from PIL import Image
import hashlib
import pickle

def histogram_equalization(gray):
    '''Applies histogram equalization to a grayscale image'''

    gray_float= gray.astype(np.float32)#prevent overflow during computations
    hist, bins= np.histogram(gray_float.flatten(), bins=256, range=[0, 256])

    cdf= hist.cumsum()
    cdf_normalized= cdf*255/cdf[-1]
    equalized_image= np.interp(gray_float.flatten(), bins[:-1], cdf_normalized)#map intensity levels based on CDF
    equalized_image= equalized_image.reshape(gray.shape)
    equalized_image= np.clip(equalized_image, 0, 255).astype(np.uint8)
    return equalized_image

def symbol_encode(image:Image, tile_size=16)->bytes:
    """Divides the image into non-overlapping tiles, storing only unique tiles."""
    img_width, img_height = image.size
    tiles = {}
    locations = []

    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            box = (x, y, min(x+tile_size, img_width), min(y+tile_size, img_height))
            tile = image.crop(box)
            tile_array= np.array(tile)

            # Create a unique hash for the tile array
            tile_hash= hashlib.md5(tile_array.tobytes()).hexdigest()

            if tile_hash not in tiles:
                tiles[tile_hash] = tile_array
            locations.append((tile_hash, (x, y)))

    return pickle.dumps((tiles, locations, image.size))

def symbol_decode(data:bytes)->Image:
    """Reconstructs the image from unique tiles."""

    tiles_dict, locations, original_size= pickle.loads(data)
    new_image = Image.new('RGB', original_size)

    for tile_hash, (x, y) in locations:
        tile = tiles_dict[tile_hash]
        new_image.paste(Image.fromarray(tile), (x, y))

    return new_image

