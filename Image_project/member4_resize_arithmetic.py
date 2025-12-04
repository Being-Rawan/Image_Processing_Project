import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def resize_nearest(img, nw, nh):
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)

def resize_bilinear(img, nw, nh):
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def resize_bicubic(img, nw, nh):
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)


def arithmetic_encode(data):
    freqs = {x:data.count(x)/len(data) for x in set(data)}
    low = 0.0
    high = 1.0
    for symbol in data:
        range_ = high - low
        high = low + range_ * sum(freqs[s] for s in freqs if s <= symbol)
        low = low + range_ * sum(freqs[s] for s in freqs if s < symbol)
    return (low + high)/2, freqs


def arithmetic_decode(code, freqs, length):
    sorted_symbols = sorted(freqs.keys())
    result = []
    low, high = 0.0, 1.0
    for _ in range(length):
        range_ = high - low
        for s in sorted_symbols:
            high_s = low + range_ * sum(freqs[x] for x in freqs if x <= s)
            low_s = low + range_ * sum(freqs[x] for x in freqs if x < s)
            if low_s <= code < high_s:
                result.append(s)
                low, high = low_s, high_s
                break
    return result

