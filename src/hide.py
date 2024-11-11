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
    
    i = 0
    j = 0
    c = 0
    t = 0

    #Hide message
    while c < channels:
        while i < height:
            while j < width and t < binary_len:
                pixel = image[i][j][c]
                bit = np.uint8(binary[t])
                new_image[i][j][c] = pixel & np.uint8(254) | bit
                j += 1
                t += 1
            i += 1
            j = 0
        c += 1
        i = 0

    return new_image

def decrypt(image: cv2.typing.MatLike, text_len: int) -> str:
    height, width, channels = image.shape

    bit_str = ""
    bit_str_len = 0

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                bit = image[i][j][c] & np.uint8(1)
                bit_str += str(bit)
                bit_str_len += 1

    limit = bit_str_len // 8
    text = ""
    text_current_len = 0

    for i in range(limit):
        bits = bit_str[i * 8 : (i + 1) * 8]
        number = int(bits, 2)
        char = chr(number)
        text += char
        text_current_len += 1
        if text_current_len >= text_len:
            break

    return text

def encrypt_and_save(read_path: str, save_path: str, text_file_path: str):
    with open(text_file_path, mode='r', encoding='ascii') as file:
        text = file.read()
        
    image = cv2.imread(read_path)
    image_encrypted = encrypt(image, text)
    cv2.imwrite(save_path, image_encrypted)

def decrypt_and_print(read_path: str, text_len: int):
    image = cv2.imread(read_path)
    text = decrypt(image, text_len)
    print(text)

def main():
    #encrypt_and_save('lena.bmp', 'lena_encrypted.bmp', 'test.txt')
    decrypt_and_print('lena_encrypted.bmp', 5000)

if __name__ == "__main__":
    main()