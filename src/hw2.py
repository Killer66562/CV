import cv2
import numpy as np


def normalize(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    max_value = np.max(image)
    min_value = np.min(image)

    delta = max_value - min_value

    if delta == 0:
        delta = 255

    new_image = ((image - min_value) / delta * 255).astype(np.uint8)
    return new_image
        

def sobel(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    height, width = image.shape

    sobel_x = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1], 
        [0, 0, 0], 
        [1, 2, 1]
    ])

    pixels_x = np.array([
        [0, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0]
    ], dtype=np.float32)

    pixels_y = np.array([
        [0, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0]
    ], dtype=np.float32)

    gx = np.array([[image[i][j] for j in range(width)] for i in range(height)], dtype=np.float32)
    gy = np.array([[image[i][j] for j in range(width)] for i in range(height)], dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            for k in range(3):
                real_k = i - k -1
                for l in range(3):
                    real_l = j - l - 1
                    if real_k < 0 or real_k >= height or real_l < 0 or real_l >= width:
                        pixels_x[k][l] = image[i][j]
                        pixels_y[k][l] = image[i][j]
                    else:
                        pixels_x[k][l] = image[real_k][real_l]
                        pixels_y[k][l] = image[real_k][real_l]

            gx[i][j] = np.sum(pixels_x * sobel_x)
            gy[i][j] = np.sum(pixels_y * sobel_y)

    new_image = np.sqrt(gx ** 2 + gy ** 2)
    new_image = normalize(new_image)
    
    return new_image

def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    lena_sobel = sobel(lena)
    cv2.imwrite('lena_sobel.png', lena_sobel)

if __name__ == "__main__":
    main()