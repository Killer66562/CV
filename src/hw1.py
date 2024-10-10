import cv2
import numpy as np


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


def main():
    image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    image_niblack = niblack(image)
    image_sauvola = sauvola(image)
    cv2.imwrite('lena_niblack.jpg', image_niblack)
    cv2.imwrite('lena_sauvola.jpg', image_sauvola)

    image_ostu = ostu(image)
    cv2.imwrite('lena_ostu.jpg', image_ostu)

if __name__ == '__main__':
    main()