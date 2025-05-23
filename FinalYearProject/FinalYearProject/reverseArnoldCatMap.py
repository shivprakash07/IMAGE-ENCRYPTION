import numpy as np

def inverse_arnold_cat_map(image, iterations=1):
    height, width = image.shape
    transformed = np.zeros_like(image)
    a, b = 1, 1
    for _ in range(iterations):
        for x in range(height):
            for y in range(width):
                orig_x = ((a * b + 1) * x - a * y) % height
                orig_y = (-b * x + y) % width
                transformed[orig_x, orig_y] = image[x, y]
        image = transformed.copy()
    return transformed