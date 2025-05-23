import numpy as np

def hennon_map(image_array,size,max_size,x0,y0):
    x=x0
    y=y0
    transformed = np.zeros_like(image_array)
    for i in range (0,size*size):
        xN = y + 1 - 1.4 * x*x
        yN = 0.3 * x
        x = xN
        y = yN
        xN=(int(xN*max_size))%max_size
        transformed[i//size,i%size]=image_array[i//size,i%size] ^ xN
    return transformed