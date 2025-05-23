import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.stats import entropy
import math

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

def calculate_mse(original, decrypted):
    return np.mean((original.astype(np.float64) - decrypted.astype(np.float64)) ** 2)

def analysisFunc(original, scrambled, encrypted, decrypted):
    plot_histogram(original, 'Original Image Histogram')
    plot_histogram(scrambled, 'Scrambled Image Histogram')
    plot_histogram(encrypted, 'Encrypted Image Histogram')
    print(f"Entropy - Original: {calculate_entropy(original):.4f}")
    print(f"Entropy - Encrypted: {calculate_entropy(encrypted):.4f}")
    print(f"Entropy - Decrypted: {calculate_entropy(decrypted):.4f}")
    print(f"Correlation - Orig vs Encrypted: {calculate_correlation(original, encrypted):.4f}")
    print(f"Correlation - Orig vs Decrypted: {calculate_correlation(original, decrypted):.4f}")
    print(f"NPCR: {npcr(original, encrypted):.2f}%")
    print(f"UACI: {uaci(original, encrypted):.4f}%")
    print(f"SSIM: {calculate_ssim(original, decrypted):.4f}")
    print(f"MSE: {calculate_mse(original, decrypted):.4f}")
    print(f"PSNR (Encrypted): {psnr(original, encrypted):.2f} dB")
    print(f"PSNR (Decrypted): {psnr(original, decrypted):.2f} dB")