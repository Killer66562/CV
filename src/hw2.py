import cv2
import numpy as np


def sobel(image: cv2.typing.MatLike, threshold: int = 63) -> cv2.typing.MatLike:
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
    ])

    pixels_y = np.array([
        [0, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0]
    ])

    new_image_x = np.array([[image[i][j] for j in range(width)] for i in range(height)])
    new_image_y = np.array([[image[i][j] for j in range(width)] for i in range(height)])
    
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

            dot_x = np.abs(np.sum(pixels_x * sobel_x))
            dot_y = np.abs(np.sum(pixels_y * sobel_y))

            print(dot_x)
            print(dot_y)

            new_image_x[i][j] = 0 if dot_x <= threshold else 255
            new_image_y[i][j] = 0 if dot_y <= threshold else 255
    
    new_image = new_image_x | new_image_y
    return new_image

def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    lena_sobel = sobel(lena)
    cv2.imwrite('lena_sobel.png', lena_sobel)

if __name__ == "__main__":
    main()