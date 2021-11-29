from cv2 import cv2
import os
from utils import *
import itertools
path_to_images = 'images'
sigma = 30
T = 200
t = 100


def calc_grad_img(img):
    y_filter = np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]))
    x_filter = y_filter.T
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=y_filter)
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=x_filter)
    return grad_x, grad_y


def non_maximum_suppression(img, angles):
    clear_img = np.zeros(img.shape)
    for i in range(1, angles.shape[0] - 1):
        for j in range(1, angles.shape[1] - 1):
            if angles[i, j] == 0 or angles[i, j] == 180:
                temp = max(img[i, j - 1], img[i, j + 1])
            elif angles[i, j] == 45 or angles[i, j] == 225:
                temp = max(img[i - 1, j + 1], img[i + 1, j - 1])
            elif angles[i, j] == 90 or angles[i, j] == 270:
                temp = max(img[i + 1, j], img[i - 1, j])
            elif angles[i, j] == 135 or angles[i, j] == 315:
                temp = max(img[i - 1, j - 1], img[i + 1, j + 1])

            if img[i, j] >= temp:
                clear_img[i, j] = img[i, j]
    return clear_img

def hysteresis(img):
    strong_x, strong_y = np.where((img > T))
    weak_x, weak_y = np.where((t <= img) & (img <= T))
    weak_xy = itertools.product(weak_x, weak_y)
    dir_x = [-1, -1, -1, 0, 1, 1, 1, 0]
    dir_y = [-1, 0, 1, 1, 1, 0, -1, -1]

    result_img = np.zeros(img.shape)

    result_img[strong_x, strong_y] = 255
    i = 0
    while i < len(strong_x):
        x = strong_x[i]
        y = strong_y[i]
        for k in range(len(dir_x)):
            shift_x = x + dir_x[k]
            shift_y = y + dir_y[k]
            if 0 <= shift_x <= img.shape[0] - 1\
                    and 0 <= shift_y <= img.shape[1] - 1\
                    and (shift_x,shift_y) in weak_xy:
                result_img[shift_x, shift_y] = 255
                strong_x = np.hstack((strong_x, shift_x))
                strong_y = np.hstack((strong_y, shift_y))
            i += 1

    result_img[result_img != 255] = 0
    return result_img


if __name__ == '__main__':
    img = read_image(os.path.join(path_to_images, 'emma.jpeg'))
    show_image(img, title='Gray image', cmap='gray')
    img_smooth = np.float32(cv2.GaussianBlur(img, (3, 3), sigma))
    show_image(img_smooth , title='Smooth image', cmap='gray')
    grad_x, grad_y = calc_grad_img(img_smooth)
    show_image(grad_x, title='Gradient X', cmap='gray')
    show_image(grad_y, title='Gradient Y', cmap='gray')
    magnitude_img = np.sqrt(grad_x**2 + grad_y ** 2)
    show_image(magnitude_img, title='Magnitude image', cmap='gray')
    angles = np.arctan2(grad_y,grad_x)
    round_angles = np.round((angles + np.pi) / (np.pi * 2) * 360 / 45, 0) * 45
    round_angles[round_angles == 360] = 0

    img = non_maximum_suppression(magnitude_img, round_angles)
    show_image(img,title='After NMS image',cmap='gray')
    img = hysteresis(img)
    show_image(img, title='After hysteresis image',cmap='gray')
    