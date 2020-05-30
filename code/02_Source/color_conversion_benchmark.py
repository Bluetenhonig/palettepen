import time

import numpy as np
import cv2

from colormath.color_objects import sRGBColor, LCHabColor
from colormath.color_conversions import convert_color

N = 256 ** 2

print("N-Pixels = ", N)
rgb_to_convert = np.random.random(size=(N, 3)).astype(np.float32)

# COLOR MATH
t = time.time()
for i in range(N):
    r, g, b = rgb_to_convert[i].tolist()
    lch = convert_color(sRGBColor(r,g,b), LCHabColor, target_illuminant='d50')
print("Colormath takes:", np.round(time.time() - t, 4), " seconds")


# OPENCV and Numpy

def lab2lch(lab, h_as_degree = False):
    """

    :param lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    :return: lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}
    """
    if not isinstance(lab, np.ndarray):
        lab = np.array(lab, dtype=np.float32)

    lch = np.zeros_like(lab, dtype=np.float32)
    lch[..., 0] = lab[..., 0]
    lch[..., 1] = np.linalg.norm(lab[..., 1:3], axis=len(lab.shape) - 1)
    lch[..., 2] = np.arctan2(lab[..., 2], lab[..., 1])

    lch[..., 2] += np.where(lch[..., 2] < 0., 2 * np.pi, 0)

    if h_as_degree:
        lch[..., 2] = (lch[..., 2] / (2*np.pi)) * 360
    return lch

t = time.time()
lab = cv2.cvtColor(np.array([rgb_to_convert],dtype=np.float32), cv2.COLOR_RGB2LAB)
lch = lab2lch(lab)

print("OpenCV and Numpy takes: ", round(time.time() - t, 4), " seconds")