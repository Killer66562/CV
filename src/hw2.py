import cv2
import numpy as np


sobel_x = np.array([
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1], 
    [0, 0, 0], 
    [1, 2, 1]
], dtype=np.float32)

prewitt_x = np.array([
    [1, 1, 1], 
    [0, 0, 0], 
    [-1, -1, -1]
], dtype=np.float32)

prewitt_y = np.array([
    [1, 0, -1], 
    [1, 0, -1], 
    [1, 0, -1]
], dtype=np.float32)

def gaussian_filter(image: cv2.typing.MatLike, rad: int = 1):
    height, width = image.shape

    new_image = np.array([[image[i][j] for j in range(width)] for i in range(height)])

    pixels_in_window = (rad * 2 + 1) ** 2
    pixels = np.array([0 for _ in range(pixels_in_window)], dtype=int)
    gases = np.array([0 for _ in range(pixels_in_window)], dtype=float)
    v = np.std(np.array([[image[i][j] for j in range(width)] for i in range(height)], dtype=float)) ** 2

    idx = 0

    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            gas = (1 / (2 * np.pi * v)) * np.pow(np.e, ((i ** 2 + j ** 2) / (2 * v)) * -1)
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
                        pixels[idx] = image[i][j] * gases[idx]
                    else:
                        pixels[idx] = image[k][l] * gases[idx]
                    idx += 1

            new_image[i][j] = int(np.sum(pixels))

    return new_image

def unnormalize(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    0-1 -> 0-255
    '''
    max_value = np.max(image)
    min_value = np.min(image)

    delta = max_value - min_value

    if delta == 0:
        new_image = np.zeros_like(image)
    else:
        new_image = ((image - min_value) / delta * 255).astype(np.uint8)

    return new_image

def normalize(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    0-255 -> 0-1
    '''
    min_value = np.min(image)
    max_value = np.max(image)

    delta = max_value - min_value

    if delta == 0:
        new_image = np.zeros_like(image)
    else:
        new_image = image.astype(np.float32) / delta

    return new_image

def get_gradients(image: cv2.typing.MatLike, operator_x, operator_y) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    '''
    Return normalized Gx, Gy
    '''
    height, width = image.shape
    k_size = 3

    normalized_image = normalize(image)

    pixels_x = np.zeros((k_size, k_size), dtype=np.float32)
    pixels_y = np.zeros((k_size, k_size), dtype=np.float32)

    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            for k in range(k_size):
                real_k = i - k -1
                for l in range(k_size):
                    real_l = j - l - 1
                    if real_k < 0 or real_k >= height or real_l < 0 or real_l >= width:
                        pixels_x[k][l] = normalized_image[i][j]
                        pixels_y[k][l] = normalized_image[i][j]
                    else:
                        pixels_x[k][l] = normalized_image[real_k][real_l]
                        pixels_y[k][l] = normalized_image[real_k][real_l]

            gx[i][j] = np.sum(pixels_x * operator_x)
            gy[i][j] = np.sum(pixels_y * operator_y)

    return (gx, gy)


def sobel(image: cv2.typing.MatLike, fast: bool = False) -> cv2.typing.MatLike:
    gx, gy = get_gradients(image, sobel_x, sobel_y)

    if fast:
        new_image = np.abs(gx) + np.abs(gy)
    else:
        new_image = np.sqrt(gx ** 2 + gy ** 2)

    new_image = unnormalize(new_image)
    
    return new_image

def prewitt(image: cv2.typing.MatLike, fast: bool = False) -> cv2.typing.MatLike:
    gx, gy = get_gradients(image, prewitt_x, prewitt_y)

    if fast:
        new_image = np.abs(gx) + np.abs(gy)
    else:
        new_image = np.sqrt(gx ** 2 + gy ** 2)

    new_image = unnormalize(new_image)
    
    return new_image

def canny(image: cv2.typing.MatLike, filter_rad: int = 1, fast: bool = False, thres_low: float = 0.2, thres_high: float = 0.4) -> cv2.typing.MatLike:
    gaussian_filtered_image = gaussian_filter(image, filter_rad)
    gaussian_filtered_image = normalize(gaussian_filtered_image)

    gx, gy = get_gradients(gaussian_filtered_image, sobel_x, sobel_y)

    if fast:
        g = np.abs(gx) + np.abs(gy)
    else:
        g = np.sqrt(gx ** 2 + gy ** 2)

    degs = np.rad2deg(np.arctan(gy / (gx + 10e-9)))

    gh, gw = g.shape

    g_cp = np.array([[g[i][j] for j in range(gw)] for i in range(gh)], dtype=np.float32)
    kernel = np.zeros((3, 3))

    for i in range(gh):
        for j in range(gw):
            for k in range(3):
                real_k = i + k - 1
                for l in range(3):
                    real_l = j + l - 1
                    if real_k < 0 or real_k >= gh or real_l < 0 or real_l >= gw:
                        kernel[k][l] = 0
                    else:
                        kernel[k][l] = g[real_k][real_l]

            pairs = None
            if degs[i][j] > 67.5 or degs[i][j] <= -67.5: #90deg
                pairs = ((0, 1), (1, 1), (2, 1)) 
            elif degs[i][j] > 22.5 and degs[i][j] <= 67.5: #45deg
                pairs = ((0, 2), (1, 1), (2, 0))
            elif degs[i][j] > -22.5 and degs[i][j] <= 22.5: #0deg
                pairs = ((1, 0), (1, 1), (1, 2))
            elif degs[i][j] > -67.5 and degs[i][j] <= 22.5: #-45deg
                pairs = ((0, 0), (1, 1), (2, 2))

            if pairs:
                k_max = np.max(np.array([kernel[a][b] for a, b in pairs]))
                if kernel[1][1] != k_max:
                    g_cp[i][j] = 0
                    continue
            else:
                continue

            if np.abs(g_cp[i][j]) >= thres_high:
                g_cp[i][j] = 1
                continue
            elif np.abs(g_cp[i][j]) <= thres_low:
                g_cp[i][j] = 0
                continue

    for i in range(gh):
        for j in range(gw):
            if g_cp[i][j] != 1:
                continue
            for k in range(3):
                real_k = i + k - 1
                for l in range(3):
                    real_l = j + l - 1
                    if real_k < 0 or real_k >= gh or real_l < 0 or real_l >= gw:
                        continue
                    elif g_cp[real_k][real_l] != 0:
                        g_cp[real_k][real_l] = 1

    for i in range(gh):
        for j in range(gw):
            if g_cp[i][j] != 1:
                g_cp[i][j] = 0

    new_image = unnormalize(g_cp)
    return new_image

def erode(image: cv2.typing.MatLike, rad: int = 1) -> cv2.typing.MatLike:
    height, width = image.shape

    k_size = (rad * 2 + 1) ** 2

    new_image = np.zeros_like(image, dtype=np.uint8)
    pixels = np.zeros((k_size, ), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            idx = 0
            for k in range(i - rad, i + rad + 1):
                for l in range(j - rad, j + rad + 1):
                    if k < 0 or k >= height or l < 0 or l >= width:
                        pixels[idx] = image[i][j]
                    else:
                        pixels[idx] = image[k][l]
                    idx = idx + 1

            for pixel in pixels:
                if pixel == 0:
                    new_image[i][j] = 0
                    break
            else:
                new_image[i][j] = 255

    return new_image

def dilate(image: cv2.typing.MatLike, rad: int = 1) -> cv2.typing.MatLike:
    height, width = image.shape

    k_size = (rad * 2 + 1) ** 2

    new_image = np.zeros_like(image, dtype=np.uint8)
    pixels = np.zeros((k_size, ), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            idx = 0
            for k in range(i - rad, i + rad + 1):
                for l in range(j - rad, j + rad + 1):
                    if k < 0 or k >= height or l < 0 or l >= width:
                        pixels[idx] = image[i][j]
                    else:
                        pixels[idx] = image[k][l]
                    idx = idx + 1

            for pixel in pixels:
                if pixel == 255:
                    new_image[i][j] = 255
                    break
            else:
                new_image[i][j] = 0

    return new_image

def opening(image: cv2.typing.MatLike, rad_erode: int = 1, rad_dilate: int = 1) -> cv2.typing.MatLike:
    return dilate(erode(image, rad_erode), rad_dilate)

def closing(image: cv2.typing.MatLike, rad_erode: int = 1, rad_dilate: int = 1) -> cv2.typing.MatLike:
    return erode(dilate(image, rad_dilate), rad_erode)


def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    lena_sobel = sobel(lena)
    cv2.imwrite('lena_sobel.png', lena_sobel)

    lena_prewitt = prewitt(lena)
    cv2.imwrite('lena_prewitt.png', lena_prewitt)

    lena_canny = canny(lena)
    cv2.imwrite('lena_canny.png', lena_canny)

    lena_binary = cv2.imread('binary.png', cv2.THRESH_BINARY)

    lena_erosion = erode(lena_binary, 2)
    cv2.imwrite('lena_erosion.png', lena_erosion)

    lena_dilation = dilate(lena_binary, 2)
    cv2.imwrite('lena_dilation.png', lena_dilation)

    lena_opening = opening(lena_binary, 1, 1)
    cv2.imwrite('lena_opening.png', lena_opening)

    lena_closing = closing(lena_binary, 1, 1)
    cv2.imwrite('lena_closing.png', lena_closing)

if __name__ == "__main__":
    main()