import cv2
import numpy as np
import math

import matplotlib.pyplot as plt


def get_gradients(image: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    '''
    Return normalized Gx, Gy
    '''

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

    height, width = image.shape
    k_size = 3

    new_image = image[:][:].astype(np.float32)

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
                        pixels_x[k][l] = new_image[i][j]
                        pixels_y[k][l] = new_image[i][j]
                    else:
                        pixels_x[k][l] = new_image[real_k][real_l]
                        pixels_y[k][l] = new_image[real_k][real_l]

            gx[i][j] = np.sum(pixels_x * sobel_x)
            gy[i][j] = np.sum(pixels_y * sobel_y)

    return (gx, gy)

def get_directions(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    gx, gy = get_gradients(image)
    return np.arctan2(gy, gx)

def build_r_table(image: cv2.typing.MatLike, angle_step: float = 1) -> dict[float, dict[float, tuple[int, int]]]:
    directions = get_directions(image)

    h, w = image.shape

    boundaries_list = []
    for i in range(h):
        for j in range(w):
            if image[i][j] > 0:
                #Save (x, y)
                boundaries_list.append((j, i))

    xl, xr = min(boundaries_list, key=lambda x: x[0])[0], max(boundaries_list, key=lambda x: x[0])[0]
    yu, yd = min(boundaries_list, key=lambda x: x[1])[1], max(boundaries_list, key=lambda x: x[1])[1]
    xc, yc = (xr - xl) // 2, (yd - yu) // 2

    r_table = {}

    for x, y in boundaries_list:
        x_int = int(x)
        y_int = int(y)
        t = math.degrees(directions[y_int][x_int])
        r = (x_int - xc, y_int - yc)
        rx, ry = r
        if not r_table.get(t):
            r_table[t] = {}
        for angle in range(0, 360, angle_step):
            angle_rad = math.radians(angle)
            r_table[t][angle] = (rx * math.sin(angle_rad) + ry * math.cos(angle_rad), rx * math.cos(angle_rad) - ry * math.sin(angle_rad))

    return r_table

def detect(template: cv2.typing.MatLike, reference: cv2.typing.MatLike, angle_step: int = 1):
    r_table = build_r_table(template)
    directions = get_directions(reference)

    h, w = reference.shape

    accumulator = np.zeros((h, w, 360), dtype=np.int32)

    for i in range(h):
        for j in range(w):
            if not reference[i][j] > 0:
                continue
            t = np.rad2deg(directions[i][j])
            # 角度存在
            angle_vector_mapping = r_table.get(t)
            if not angle_vector_mapping:
                continue
            # 向量(dx, dy) 存在
            for angle in angle_vector_mapping:
                # 投票時間! 
                # 考量到可能變換的角度, 轉整數
                x, y = angle_vector_mapping[angle]
                x_c, y_c = int(j - x), int(i - y)
                print(x_c, y_c)
                # 想投票的目標經過轉換依舊在範圍內
                if 0 <= x_c < w and 0 <= y_c < h:
                    accumulator[x_c][y_c][angle] += 1

    return accumulator

def main():
    template = cv2.imread("Template.png", cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread("Refernce.png", cv2.IMREAD_GRAYSCALE)

    accumulator = detect(template, reference, 1)
    print(np.max(accumulator))

    x, y, angle = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    print(x, y, angle)
    
    # 將圖片轉換為 0~255 的範圍
    result_image = np.clip(reference, 0, 255).astype(np.uint8)
    
    fig, ax = plt.subplots()
    ax.imshow(result_image, cmap="gray")

    # 繪製中心點
    ax.scatter(y, x, c='red', s=20, label=f"Center (x={y}, y={x}, angle={angle}°)")
    plt.show()

if __name__ == "__main__":
    main()