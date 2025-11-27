import numpy as np

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

#TODO(Wagih): find some actual algorithm to be implemented for "Symbol-Based Coding".
# all I could find was some indians describing something with the same name:
#https://www.youtube.com/watch?v=DciaZHFRX_A
#https://www.ripublication.com/irph/ijert19/ijertv12n4_05.pdf
# AI insists that Symbol-Based is the same as RLE and doesn't seem to generate any other useful code otherwise...
def symbol_encode():
    pass

def symbol_decode():
    pass

