import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


class RTable(object):
    '''
    請存(x, y)的向量
    '''
    def __repr__(self):
        text = ""

        for key in self.content:
            text += f"Theta: {key}\n"
            for vector in self.content[key]:
                text += f"Vector: {vector}\n"

        return text
    
    def __init__(self):
        self.content: dict[float, np.ndarray] = {}
    
    def exists(self, theta: float) -> bool:
        return self.content.get(theta)

    def add(self, theta: float, vector: np.ndarray):
        if not self.exists(theta=theta):
            self.content[theta] = []
        self.content[theta].append(vector)

    def get(self, theta: float) -> np.ndarray:
        return self.content.get(theta)

class GHT(object):
    SOBEL_X = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
    ], dtype=np.float32)

    SOBEL_Y = np.array([
        [-1, -2, -1], 
        [0, 0, 0], 
        [1, 2, 1]
    ], dtype=np.float32)

    def __init__(self):
        pass

    def _get_gradients(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        height, width = image.shape
        k_size = 3

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
                            pixels_x[k][l] = image[i][j]
                            pixels_y[k][l] = image[i][j]
                        else:
                            pixels_x[k][l] = image[real_k][real_l]
                            pixels_y[k][l] = image[real_k][real_l]

                gx[i][j] = np.sum(pixels_x * self.SOBEL_X)
                gy[i][j] = np.sum(pixels_y * self.SOBEL_Y)

        return (gx, gy)

    def build_r_table(self, image: cv2.typing.MatLike) -> RTable:
        r_table = RTable()

        h, w = image.shape
        yc, xc = h // 2, w // 2

        gx, gy = self._get_gradients(image)
        direction = np.arctan2(gy, gx)

        for i in range(h):
            for j in range(w):
                if image[i][j] > 0:
                    t = np.rad2deg(direction[i][j])
                    r_table.add(theta=t, vector=np.array([j - xc, i - yc]))
        
        return r_table

    def detect(self, template: cv2.typing.MatLike, reference: cv2.typing.MatLike, angle_step: float = 1.0, scale_min: float = 1.0, scale_max: float = 2.0, scale_step: float = 0.1):
        h, w = reference.shape
        
        #建立r_table
        r_table = self.build_r_table(image=template)
        
        #取得梯度並計算方向
        gx, gy = self._get_gradients(reference)
        direction = np.arctan2(gy, gx)
        thetas = np.rad2deg(direction)

        #初始化角度
        angles = np.deg2rad(np.arange(0, 360, angle_step))
        angles_len = len(angles)

        #初始化尺度
        scales = np.arange(scale_min, scale_max + 10e-7, scale_step)
        scales_len = len(scales)

        #投票機
        accumulator = np.zeros((h, w, angles_len, scales_len), dtype=np.int32)

        #計算cos, sin
        cos = np.cos(angles)
        sin = np.sin(angles)

        #建立縮放後的cos, sin表
        cos_scale_table = np.array([[cos[angle_idx] * scales[scale_idx] for scale_idx in range(scales_len)] for angle_idx in range(angles_len)], dtype=np.float32)
        sin_scale_table = np.array([[sin[angle_idx] * scales[scale_idx] for scale_idx in range(scales_len)] for angle_idx in range(angles_len)], dtype=np.float32)

        #取得所有有效邊緣點
        positions = np.array([(i, j) for i in range(h) for j in range(w) if reference[i][j] > 0 and r_table.exists(theta=thetas[i, j])], dtype=np.int32)
        positions_len = len(positions)
        counter = 0

        #開始計時
        print("Start voting")
        time_start = time.time()

        for i, j in positions:
            counter += 1
            percentage = counter / positions_len * 100
            time_delta = time.time() - time_start
            print(f"Progress: {counter}/{positions_len} ({percentage:.2f}%), Time: {time_delta:.2f} seconds", end="\r")

            for r in r_table.get(theta=thetas[i, j]):
                xc = j - r[0] * cos_scale_table + r[1] * sin_scale_table
                yc = i - r[0] * sin_scale_table - r[1] * cos_scale_table

                xc = xc.astype(np.int32)
                yc = yc.astype(np.int32)

                allowed = (xc >= 0) & (xc < w) & (yc >= 0) & (yc < h) #沒超出邊界

                xc = xc[allowed] #中心點的x列表
                yc = yc[allowed] #中心點的y列表

                #取得角度和尺度的索引
                angle_scale_idxs = np.array([[(a, b) for b in range(scales_len)] for a in range(angles_len)], dtype=np.int32)[allowed]
                angle_idxs = angle_scale_idxs[:, 0]
                scale_idxs = angle_scale_idxs[:, 1]

                #投票
                accumulator[yc, xc, angle_idxs, scale_idxs] += 1

        time_end = time.time()
        time_delta = time_end - time_start

        print(f"Finished in {time_delta:.2f} seconds", end="\r")

        return accumulator
    
    def show(self, image: cv2.typing.MatLike, accumulator: cv2.typing.MatLike, angle_step: float = 1.0, scale_min: float = 1.0, scale_max: float = 2.0, scale_step: float = 0.1):
        # 找到得票數最多的候選位置
        x, y, angle_idx, scale_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        angle = angle_idx * angle_step
        scale = scale_min + scale_idx * scale_step
        
        # 將圖片轉換為 0~255 的範圍
        result_image = np.clip(image, 0, 255).astype(np.uint8)

        # 顯示結果圖片
        fig, ax = plt.subplots()

        # 繪製中心點
        ax.scatter(y, x, c='red', s=20, label=f"Center (x={y}, y={x}, angle={angle}°, scale={scale})")

        ax.imshow(result_image, cmap="gray")
        plt.show()


def main():
    ght = GHT()

    reference = cv2.imread('Refernce.png', cv2.IMREAD_GRAYSCALE)

    angle_step = 10
    scale_step = 0.1
    
    '''
    scale_min = 0.3
    scale_max = 1

    template_filenames = ["Template.png", "Template_90.png", "Template_180.png", "Template_270.png"] #僅轉向

    for filename in template_filenames:
        print(f"Processing {filename}")

        template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        accumulator = ght.detect(template=template, reference=reference, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)

        print(f"Finished {filename}")
        ght.show(image=reference, accumulator=accumulator, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)

    scale_min = 1.5
    scale_max = 2
    template_filenames = ["Template_small.png"] #僅縮放且Template為小圖

    for filename in template_filenames:
        print(f"Processing {filename}")

        template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        accumulator = ght.detect(template=template, reference=reference, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)

        print(f"Finished {filename}")
        ght.show(image=reference, accumulator=accumulator, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)
    '''

    scale_min = 0.3
    scale_max = 0.7
    template_filenames = ["Template_large.png"] #僅縮放且Template為大圖

    for filename in template_filenames:
        print(f"Processing {filename}")

        template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        accumulator = ght.detect(template=template, reference=reference, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)

        print(f"Finished {filename}")
        ght.show(image=reference, accumulator=accumulator, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)

    template_filenames = ["Template_large_any.png"] #縮放加任意角度旋轉且Template為大圖

    for filename in template_filenames:
        print(f"Processing {filename}")

        template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        accumulator = ght.detect(template=template, reference=reference, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)

        print(f"Finished {filename}")
        ght.show(image=reference, accumulator=accumulator, angle_step=angle_step, scale_min=scale_min, scale_max=scale_max, scale_step=scale_step)



if __name__ == '__main__':
    main()

