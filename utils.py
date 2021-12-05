from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    rgb_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    return rgb_image

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def save_image(img, name):
    cv2.imwrite(name, img)