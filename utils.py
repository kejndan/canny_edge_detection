import cv2
import numpy as np
import matplotlib.pyplot as plt
def read_image(path, color=False):
    if color:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=np.float64)
    else:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), dtype=np.float64)
    return image

def show_image(image, title=None, cmap=None, textbox=None):
    if cmap is not None:
        plt.imshow(np.uint8(image),cmap=cmap)
    else:
        plt.imshow(np.uint8(image))
    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.show()