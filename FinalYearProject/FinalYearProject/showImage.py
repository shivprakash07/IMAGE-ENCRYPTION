import matplotlib.pyplot as plt

def show_image(image):
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image,cmap='gray')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()