import cv2
import numpy as np
import multiprocessing

from concurrent.futures import ThreadPoolExecutor
#Local threshold
def niblack(image: cv2.typing.MatLike, rad: int = 1, K: float = -0.2) -> cv2.typing.MatLike:
    height, width = image.shape
    pixels_in_window = (rad * 2 + 1) ** 2

    new_image = np.array([[image[i][j] for j in range(width)] for i in range(height)])
    pixels = np.array([0 for _ in range(pixels_in_window)], dtype=int)

    for i in range(height):
        for j in range(width):
            idx = 0
            for k in range(i - rad, i + rad + 1):
                for l in range(j - rad, j + rad + 1):
                    if k < 0 or k >= height or l < 0 or l >= width:
                        pixels[idx] = 0
                    else:
                        pixels[idx] = image[k][l]
                    idx += 1
            local_mean = np.mean(pixels)
            standard_deviation = np.std(pixels)
            threshold = local_mean + K * standard_deviation

            if new_image[i][j] >= threshold:
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0

    return new_image

#Local threshold
def sauvola(image: cv2.typing.MatLike, rad: int = 1, K: float = -0.2, R: int = 128) -> cv2.typing.MatLike:
    height, width = image.shape
    pixels_in_window = (rad * 2 + 1) ** 2

    new_image = np.array([[image[i][j] for j in range(width)] for i in range(height)])
    pixels = np.array([0 for _ in range(pixels_in_window)], dtype=int)

    for i in range(height):
        for j in range(width):
            idx = 0
            for k in range(i - rad, i + rad + 1):
                for l in range(j - rad, j + rad + 1):
                    if k < 0 or k >= height or l < 0 or l >= width:
                        pixels[idx] = 0
                    else:
                        pixels[idx] = image[k][l]
                    idx += 1
            local_mean = np.mean(pixels)
            standard_deviation = np.std(pixels)
            threshold = local_mean * (1 + K * (standard_deviation / (R - 1)))

            if new_image[i][j] >= threshold:
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0

    return new_image

#Global threshold
def isodata_thresholding(image: cv2.typing.MatLike, threshold: int, difference: float = 0.5) -> cv2.typing.MatLike:
    height, width = image.shape
    new_image = np.array([[image[i][j] for j in range(width)] for i in range(height)])

    while True:
        prev_threshold = threshold

        pixels_below_threshold = image[image <= threshold]
        pixels_above_threshold = image[image > threshold]

        pixels_below_threshold_avr = pixels_below_threshold.mean() if pixels_below_threshold.size > 0 else 0
        pixels_above_threshold_avr = pixels_above_threshold.mean() if pixels_above_threshold.size > 0 else 0

        threshold = (pixels_below_threshold_avr + pixels_above_threshold_avr) / 2.0
        if abs(prev_threshold - threshold) < difference:
            break

    for i in range(height):
        for j in range(width):
            new_image[i][j] = 0 if new_image[i][j] <= threshold else 255

    return new_image

#Globol threshold
def ostu(image: cv2.typing.MatLike):
    height, width = image.shape
    n = height * width

    new_image = np.array([[image[i][j] for j in range(width)] for i in range(height)])

    p_arr = np.array([(new_image == i).sum() / n for i in range(256)])
    mg = np.sum(np.arange(256) * p_arr)

    diff_max = 0
    best_threshold = 0

    for threshold in range(1, 256):
        p1 = np.sum(p_arr[:threshold], dtype=float)
        p2 = 1 - p1

        if p1 == 0 or p2 == 0:
            continue

        m1 = np.sum(np.arange(0, threshold) * p_arr[:threshold]) / p1
        m2 = np.sum(np.arange(threshold, 256) * p_arr[threshold:]) / p2

        diff = p1 * ((m1 - mg) ** 2) + p2 * ((m2 - mg) ** 2)

        if diff > diff_max:
            diff_max = diff
            best_threshold = threshold

    for i in range(height):
        for j in range(width):
            if new_image[i][j] >= best_threshold:
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0

    return new_image

def mean_filter(image: cv2.typing.MatLike, rad: int = 1):
    height, width, channels = image.shape

    new_image = np.array([[[image[i][j][c] for c in range(channels)] for j in range(width)] for i in range(height)])

    pixels_in_window = (rad * 2 + 1) ** 2
    pixels = np.array([0 for _ in range(pixels_in_window)], dtype=int)

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                pixel = image[i][j][c]
                idx = 0
                for k in range(i - rad, i + rad + 1):
                    for l in range(j - rad, j + rad + 1):
                        if k < 0 or k >= height or l < 0 or l >= width:
                            pixels[idx] = pixel
                        else:
                            pixels[idx] = image[k][l][c]
                        idx += 1
                new_image[i][j][c] = int(np.mean(pixels))

    return new_image

def gas_filter(image: cv2.typing.MatLike, rad: int = 1):
    height, width, channels = image.shape

    new_image = np.array([[[image[i][j][c] for c in range(channels)] for j in range(width)] for i in range(height)])

    pixels_in_window = (rad * 2 + 1) ** 2
    pixels = np.array([0 for _ in range(pixels_in_window)], dtype=int)
    gases = np.array([0 for _ in range(pixels_in_window)], dtype=float)
    v_arr = np.array([0 for _ in range(channels)], dtype=float)

    for c in range(channels):
        v_arr[c] = np.std(np.array([[image[i][j][c] for j in range(width)] for i in range(height)], dtype=float)) ** 2
        idx = 0

        for i in range(-rad, rad + 1):
            for j in range(-rad, rad + 1):
                gas = (1 / (2 * np.pi * v_arr[c])) * np.pow(np.e, ((i ** 2 + j ** 2) / (2 * v_arr[c])) * -1)
                gases[idx] = gas
                idx += 1
        
        sum = np.sum(gases)
        gases = gases / np.array([sum for _ in range(pixels_in_window)])

        for i in range(height):
            for j in range(width):
                idx = 0
                for k in range(i - rad, i + rad + 1):
                    for l in range(j - rad, j + rad + 1):
                        if k < 0 or k >= height or l < 0 or l >= width:
                            pixels[idx] = image[i][j][c] * gases[idx]
                        else:
                            pixels[idx] = image[k][l][c] * gases[idx]
                        idx += 1

                new_image[i][j][c] = int(np.sum(pixels))

    return new_image

def main():

    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    '''

    lena_niblack = niblack(lena)
    cv2.imwrite('lena_niblack.jpg', lena_niblack)

    lena_sauvola = sauvola(lena)
    cv2.imwrite('lena_sauvola.jpg', lena_sauvola)

    lena_ostu = ostu(lena)
    cv2.imwrite('lena_ostu.jpg', lena_ostu)
    '''
    lena_isodata_thresholding = isodata_thresholding(lena, 255)
    cv2.imwrite('lena_isodata_thresholding.jpg', lena_isodata_thresholding)

    noise = cv2.imread('noise.bmp')

    '''
    noise_mean_filter = mean_filter(noise, rad=1)
    cv2.imwrite('noise_mean_filter.jpg', noise_mean_filter)

    noise_gas_filter = gas_filter(noise, rad=1)
    cv2.imwrite('noise_gas_filter.jpg', noise_gas_filter)
    '''

if __name__ == '__main__':
    main()