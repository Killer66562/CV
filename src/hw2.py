import cv2
import numpy as np


def get_sobel_x(rad: int = 1):
    k_size = rad * 2 + 1
    kernel = np.zeros((k_size, k_size), dtype=np.float32)

    for i in range(k_size):
        if i == rad:
            kernel[i] = np.array(list(range(-2 * rad, 2 * rad + 1, 2)))
        else:
            kernel[i] = np.array(list(range(-1 * rad, 1 * rad + 1, 1)))
    
    return kernel

def get_sobel_y(rad: int = 1):
    k_size = rad * 2 + 1
    kernel = np.zeros((k_size, k_size), dtype=np.float32)

    for i in range(k_size):
        if i == rad:
            kernel[:k_size, i] = np.array(list(range(-2 * rad, 2 * rad + 1, 2)))
        else:
            kernel[:k_size, i] = np.array(list(range(-1 * rad, 1 * rad + 1, 1)))
    
    return kernel

def unnormalize(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    0-1 -> 0-255
    '''
    max_value = np.max(image)
    min_value = np.min(image)

    delta = max_value - min_value

    if delta == 0:
        delta = 255

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

def get_gradients(image: cv2.typing.MatLike, rad: int = 1) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    '''
    Return normalized Gx, Gy
    '''
    height, width = image.shape
    k_size = rad * 2 + 1

    normalized_image = normalize(image)

    sobel_x = get_sobel_x(rad)
    sobel_y = get_sobel_y(rad)
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

            gx[i][j] = np.sum(pixels_x * sobel_x)
            gy[i][j] = np.sum(pixels_y * sobel_y)

    return (gx, gy)


def sobel(image: cv2.typing.MatLike, rad: int = 1) -> cv2.typing.MatLike:
    gx, gy = get_gradients(image, rad)

    new_image = np.sqrt(gx ** 2 + gy ** 2)

    #Faster method
    #new_image = np.abs(gx) + np.abs(gy)

    new_image = unnormalize(new_image)
    
    return new_image

def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    lena_sobel = sobel(lena, 1)
    cv2.imwrite('lena_sobel.png', lena_sobel)

if __name__ == "__main__":
    main()