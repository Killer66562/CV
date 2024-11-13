import cv2
import numpy as np


def encrypt(image: cv2.typing.MatLike, text: str):
    encoded_text = text.encode('ascii')
    binary = "".join("{:08b}".format(b) for b in encoded_text)
    binary_len = len(binary)

    height, width, channels = image.shape

    if height * width * channels < binary_len:
        print("The image is not big enough to hide the text.")
        return image
    
    new_image = np.array(image, dtype=np.uint8)
    image = image & np.uint8(254)
    
    #Hide message
    t = 0
    zeros = 0
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                if t >= binary_len:
                    new_image[i][j][c] = image[i][j][c] & np.uint8(254)
                    zeros += 1
                    if zeros >= 8:
                        break
                else:
                    new_image[i][j][c] = image[i][j][c] & np.uint8(254) | np.uint8(int(binary[t]))
                    t += 1
            if t >= binary_len:
                break
        if t >= binary_len:
            break

    return new_image

def decrypt(image: cv2.typing.MatLike) -> str:
    height, width, channels = image.shape

    bit_str = ""
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                bit = image[i][j][c] & np.uint8(1)
                bit_str += str(bit)

    limit = (height * width * channels) // 8
    text = ""

    for i in range(limit):
        bits = bit_str[i * 8 : (i + 1) * 8]
        number = int(bits, 2)
        if number == 0:
            break
        char = chr(number)
        text += char

    return text

def encrypt_and_save(read_path: str, save_path: str, text_file_path: str):
    with open(text_file_path, mode='r', encoding='ascii') as file:
        text = file.read()
        
    image = cv2.imread(read_path)
    image_encrypted = encrypt(image, text)
    cv2.imwrite(save_path, image_encrypted)

def decrypt_and_print(read_path: str):
    image = cv2.imread(read_path)
    text = decrypt(image)
    print(text)

def main():
    #encrypt_and_save('lena.bmp', 'lena_encrypted.bmp', 'test.txt')
    decrypt_and_print('lena_encrypted.bmp')

if __name__ == "__main__":
    main()