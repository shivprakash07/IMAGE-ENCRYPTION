import numpy as np
import matplotlib.pyplot as plt
import pydicom
import random
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import math
from concurrent.futures import ThreadPoolExecutor
import time
# ---------- Utility Functions ----------
def show_image(image):
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image, cmap='gray')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def stegono(image, msgBits, required_space, bit_size):
    size = image.shape[0]
    for i in range(required_space):
        pixel_value = image[i // size, i % size]
        pixel_value = format(pixel_value, '0' + str(bit_size) + 'b')
        pixel_value = pixel_value[:bit_size - 2] + msgBits[i * 2:i * 2 + 2]
        pixel_value = int(pixel_value, 2)
        image[i // size, i % size] = pixel_value
    return image

def revStego(image, bit_size):
    size = image.shape[0]
    msg = ''
    i = 0
    while True:
        char = ''
        for j in range(4):
            pixel_value = image[(i * 4 + j) // size, (i * 4 + j) % size]
            pixel_value = format(pixel_value, '0' + str(bit_size) + 'b')
            char += pixel_value[bit_size - 2:]
        msgChar = chr(int(char, 2))
        msg += msgChar
        if msg.endswith('####'):
            break
        i += 1
    return msg[:-4]

def arnold_cat_map(image, iterations=1):
    """
    Applies the Arnold Cat Map (ACM) to scramble the image.

    Parameters:
        image (numpy.ndarray): The image to be scrambled (2D array).
        iterations (int): The number of iterations to apply the Arnold Cat Map.
    
    Returns:
        numpy.ndarray: The scrambled image.
    """
    h, w = image.shape
    a, b = 1, 1

    # Precompute the coordinate grid
    x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    for _ in range(iterations):
        # Perform Arnold Cat Map transformation
        new_x = (x + b * y) % h
        new_y = (a * x + (a * b + 1) * y) % w
        scrambled = image[new_x, new_y]
        
        # Update image and coordinates for the next iteration
        image = scrambled
        x, y = new_x, new_y

    return image
def inverse_arnold_cat_map(image, iterations=1):
    """
    Applies the inverse Arnold Cat Map (ACM) to unscramble the image.

    Parameters:
        image (numpy.ndarray): The image to be unscrambled (2D array).
        iterations (int): The number of iterations to apply the inverse Arnold Cat Map.
    
    Returns:
        numpy.ndarray: The unscrambled image.
    """
    h, w = image.shape
    a, b = 1, 1

    # Precompute the inverse coordinate grid
    x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    for _ in range(iterations):
        # Perform the inverse Arnold Cat Map transformation
        new_x = (x - b * y) % h
        new_y = (-a * x + (a * b + 1) * y) % w
        unscrambled = image[new_x, new_y]
        
        # Update image and coordinates for the next iteration
        image = unscrambled
        x, y = new_x, new_y

    return image

def inverse_arnold_cat_map(image, iterations=1):
    h, w = image.shape
    a, b = 1, 1

    # Precompute coordinate grid
    x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Inverse transformation matrix:
    # [x]   = [(a*b+1), -a ] [x']
    # [y]     [ -b    , 1 ] [y'] mod N
    for _ in range(iterations):
        orig_x = ((a * b + 1) * x - a * y) % h
        orig_y = (-b * x + y) % w
        unscrambled = image[orig_x, orig_y]
        image = unscrambled  # Update for next iteration
        x, y = orig_x, orig_y  # Update coordinates

    return image


def generate_henon_key_parallel(size, x0, y0, max_value):
    def compute_row(row):
        x, y = x0 + row * 0.000001, y0 + row * 0.000001
        row_key = np.zeros(size, dtype=np.uint16)
        for col in range(size):
            xN = y + 1 - 1.4 * x * x
            yN = 0.3 * x
            x, y = xN, yN
            row_key[col] = int(abs(xN * max_value)) % max_value
        return row_key

    with ThreadPoolExecutor() as executor:
        key_rows = list(executor.map(compute_row, range(size)))

    return np.vstack(key_rows)

def hennon_map(image_array, x0, y0, max_value):
    size = image_array.shape[0]
    henon_key = generate_henon_key_parallel(size, x0, y0, max_value)
    return np.bitwise_xor(image_array, henon_key)

def inverse_hennon_map(image_array, x0, y0, max_value):
    return hennon_map(image_array, x0, y0, max_value)

def add_points(P, Q, p):
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and y1 == y2:
        beta = (3 * x1 * x2 + 0) * pow(2 * y1, -1, p)
    else:
        beta = (y2 - y1) * pow(x2 - x1, -1, p)
    x3 = (beta * beta - x1 - x2) % p
    y3 = (beta * (x1 - x3) - y1) % p
    return x3, y3

def apply_double_and_add_method(G, k, p):
    target_point = G
    k_bin = bin(k)[2:]
    for i in range(1, len(k_bin)):
        target_point = add_points(target_point, target_point, p)
        if k_bin[i] == "1":
            target_point = add_points(target_point, G, p)
    return target_point

def plot_histogram(image, title='Histogram'):
    max_pixel_value = int(np.max(image))
    plt.figure()
    plt.hist(image.flatten(), bins=max_pixel_value + 1, range=(0, max_pixel_value), color='pink', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def calculate_entropy(image):
    max_pixel_value = np.max(image)
    hist, _ = np.histogram(image.flatten(), bins=max_pixel_value + 1, range=(0, max_pixel_value))
    hist = hist / np.sum(hist)
    return entropy(hist, base=2)

def calculate_correlation(image1, image2):
    return np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

def npcr(original, encrypted):
    return np.sum(original != encrypted) / original.size * 100

def uaci(original, encrypted):
    original = original.astype(np.int32)
    encrypted = encrypted.astype(np.int32)
    diff = np.abs(original - encrypted).astype(np.float64)
    return np.sum(diff) / (original.size * 4095) * 100

def psnr(original, encrypted):
    mse = np.mean((original - encrypted) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(4095 / math.sqrt(mse))

def calculate_ssim(original, encrypted):
    return ssim(original, encrypted, data_range=4095)

def calculate_mse(original,decrypted):
    return np.mean((original.astype(np.float64) - decrypted.astype(np.float64)) ** 2)

def analysisFunc(original, scrambled, encrypted, decrypted):
    plot_histogram(original, 'Original Image Histogram')
    plot_histogram(scrambled, 'Scrambled Image Histogram')
    plot_histogram(encrypted, 'Encrypted Image Histogram')
    print(f"Entropy - Original: {calculate_entropy(original)}")
    print(f"Entropy - Encrypted: {calculate_entropy(encrypted)}")
    print(f"Entropy - Decrypted: {calculate_entropy(decrypted)}")
    print(f"Correlation - Orig vs Encrypted: {calculate_correlation(original, encrypted)}")
    print(f"Correlation - Orig vs Decrypted: {calculate_correlation(original, decrypted)}")
    print(f"NPCR: {npcr(original, encrypted)}%")
    print(f"UACI: {uaci(original, encrypted)}%")
    print(f"SSIM: {calculate_ssim(original, decrypted)}")
    print(f" MSE: {calculate_mse(original,decrypted)}")
    print(f"PSNR (Encrypted): {psnr(original, encrypted)} dB")
    print(f"PSNR (Decrypted): {psnr(original, decrypted)} dB")

def time_key_generation():
    start_time = time.time()
    ka = random.getrandbits(256)
    kb = random.getrandbits(256)
    G = (
        55066263022277343669578718895168534326250603453777594175500187360389116729240,
        32670510020758816978083085130507043184471273380659243275938904335757337482424
    )
    p = pow(2, 256) - pow(2, 32) - pow(2, 9) - pow(2, 8) - pow(2, 7) - pow(2, 6) - pow(2, 4) - pow(2, 0)
    Qa = apply_double_and_add_method(G, ka, p)
    Qb = apply_double_and_add_method(G, kb, p)
    Sa = apply_double_and_add_method(Qb, ka, p)
    Sb = apply_double_and_add_method(Qa, kb, p)
    assert Sa == Sb
    x0 = int(str(Sa[0])[:15]) / 10**15
    y0 = int(str(Sa[0])[15:30]) / 10**15
    end_time = time.time()
    print(f"Time taken for Key Generation: {end_time - start_time:.2f} seconds")
    return x0, y0

def time_encryption_pipeline(x0, y0):
    dicom_path = r"C:\\Users\\shivp\\OneDrive\\Desktop\\sathishheadscan.dcm"
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_image = dicom_data.pixel_array.copy()
    name = str(dicom_data.PatientName)
    bit_size = dicom_data.BitsStored
    max_value = pow(2, bit_size)

    # Start timing the encryption pipeline
    start_time = time.time()

    # 1. Arnold Cat Map Scrambling
    start_sub_time = time.time()
    scrambled = arnold_cat_map(dicom_image, 15)
    end_sub_time = time.time()
    print(f"Time taken for Arnold Cat Map: {end_sub_time - start_sub_time:.2f} seconds")

    # 2. Henon Map Encryption
    start_sub_time = time.time()
    encrypted = hennon_map(scrambled, x0, y0, max_value)
    end_sub_time = time.time()
    print(f"Time taken for Henon Map Encryption: {end_sub_time - start_sub_time:.2f} seconds")

    # 3. Steganography Embedding
    message = f"Patient Name: {name}####"
    msgBits = ''.join([format(ord(c), '08b') for c in message])
    required_space = len(message) * 4
    start_sub_time = time.time()
    stego_encrypted = stegono(encrypted.copy(), msgBits, required_space, bit_size)
    end_sub_time = time.time()
    print(f"Time taken for Steganography Embedding: {end_sub_time - start_sub_time:.2f} seconds")

    dicom_data.PixelData = stego_encrypted.tobytes()
    dicom_data.save_as("stego_encrypted_output.dcm")

    end_time = time.time()
    print(f"Time taken for Encryption Pipeline: {end_time - start_time:.2f} seconds")

def time_decryption_pipeline(x0, y0):
    dicom_data = pydicom.dcmread("stego_encrypted_output.dcm")
    stego_encrypted = dicom_data.pixel_array.copy()

    # Start timing the decryption pipeline
    start_time = time.time()

    # 1. Steganography Extraction
    start_sub_time = time.time()
    recovered_message = revStego(stego_encrypted.copy(), dicom_data.BitsStored)
    print("Recovered Message:", recovered_message)
    end_sub_time = time.time()
    print(f"Time taken for Steganography Extraction: {end_sub_time - start_sub_time:.2f} seconds")

    # 2. Henon Map Decryption
    start_sub_time = time.time()
    decrypted_henon = inverse_hennon_map(stego_encrypted, x0, y0, pow(2, dicom_data.BitsStored))
    end_sub_time = time.time()
    print(f"Time taken for Henon Map Decryption: {end_sub_time - start_sub_time:.2f} seconds")

    # 3. Arnold Cat Map Decryption
    start_sub_time = time.time()
    final_image = inverse_arnold_cat_map(decrypted_henon, 15)
    end_sub_time = time.time()
    print(f"Time taken for Arnold Cat Map Decryption: {end_sub_time - start_sub_time:.2f} seconds")

    end_time = time.time()
    print(f"Time taken for Decryption Pipeline: {end_time - start_time:.2f} seconds")

# ---------- Main Runner ----------
def main():
    x0, y0 = time_key_generation()
    
    # Load original image
    dicom_path = r"C:\\Users\\shivp\\OneDrive\\Desktop\\sathishheadscan.dcm"
    dicom_data = pydicom.dcmread(dicom_path)
    original_image = dicom_data.pixel_array.copy()
    name = str(dicom_data.PatientName)
    bit_size = dicom_data.BitsStored
    max_value = pow(2, bit_size)

    # Encryption
    scrambled = arnold_cat_map(original_image.copy(), 15)
    encrypted = hennon_map(scrambled.copy(), x0, y0, max_value)
    message = f"Patient Name: {name}####"
    msgBits = ''.join([format(ord(c), '08b') for c in message])
    required_space = len(message) * 4
    stego_encrypted = stegono(encrypted.copy(), msgBits, required_space, bit_size)

    # Save stego image
    dicom_data.PixelData = stego_encrypted.tobytes()
    dicom_data.save_as("stego_encrypted_output.dcm")

    # Decryption
    dicom_data = pydicom.dcmread("stego_encrypted_output.dcm")
    stego_encrypted = dicom_data.pixel_array.copy()
    revStego(stego_encrypted.copy(), bit_size)  # Just to show message
    decrypted_henon = inverse_hennon_map(stego_encrypted.copy(), x0, y0, max_value)
    final_image = inverse_arnold_cat_map(decrypted_henon.copy(), 15)

    # Perform full analysis
    analysisFunc(original_image, scrambled, encrypted, final_image)
# def main():
#     x0, y0 = time_key_generation()
#     time_encryption_pipeline(x0, y0)
#     time_decryption_pipeline(x0, y0)

if __name__ == '__main__':
    main()