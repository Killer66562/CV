import cv2
import numpy as np
import math


def fat(image: cv2.typing.MatLike, rad: int = 1, threshold: int = 63):
    height, width = image.shape

    new_image = np.array([[image[i][j] for j in range(width)] for i in range(height)])

    for i in range(height):
        for j in range(width):
            if image[i][j] <= threshold:
                for k in range(i - rad, i + rad + 1):
                    for l in range(j - rad, j + rad + 1):
                        if k < 0 or k >= height or l < 0 or l >= width:
                            continue
                        elif math.sqrt((i - k) ** 2 + (j - l) ** 2) <= rad:
                            new_image[k][l] = 0
            elif new_image[i][j] > 0:
                new_image[i][j] = 255
    
    new_image = (new_image ^ image) ^ 255
    
    return new_image
    