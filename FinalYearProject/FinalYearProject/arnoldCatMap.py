import numpy as np

def arnold_cat_map(image, iterations=1):
    h, w = image.shape
    scrambled = np.copy(image)
    a, b = 1, 1 

    for _ in range(iterations):
        temp = np.zeros_like(scrambled)
        for x in range(h):
            for y in range(w):
                new_x = (x + b * y) % h
                new_y = (a * x + (a * b + 1) * y) % w
                temp[new_x, new_y] = scrambled[x, y]
        scrambled = temp
    return scrambled