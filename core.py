import numpy as np
from cv2 import cv2
import itertools
from two_pass_connected_components_labeling import two_pass_connected_components

def calc_grad_img(img):
    y_filter = np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]))
    x_filter = y_filter.T
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=y_filter)
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=x_filter)
    return grad_x, grad_y

def non_maximum_suppression(magnitude_img, angles):
    pi_8 = 180/8

    angles = np.where((-180 <= angles) & (angles < -180 + pi_8), -angles, angles)
    clear_img = np.zeros_like(magnitude_img)
    cropped_angles = angles[1:-1, 1:-1]

    directions = {0:{'angles':(0, 180), 'pixels':[(0,-1), (0, 1)]},
                  1:{'angles':(-45, 135), 'pixels':[(1,-1), (-1, 1)]},
                  2:{'angles':(90, -90), 'pixels':[(1,0), (-1, 0)]},
                  3:{'angles':(-135, 45), 'pixels':[(-1,-1), (1, 1)]}}

    for dir in directions.values():
        line = np.where(((dir['angles'][0] + pi_8 > cropped_angles)&(cropped_angles >= dir['angles'][0] - pi_8))
                               |((dir['angles'][1] + pi_8 > cropped_angles)&(cropped_angles >= dir['angles'][1] - pi_8)))
        lines_y = line[0] + 1
        lines_x = line[1] + 1

        temp = np.maximum(magnitude_img[lines_y + dir['pixels'][0][0], lines_x + dir['pixels'][0][1]],
                          magnitude_img[lines_y + dir['pixels'][1][0], lines_x + dir['pixels'][1][1]])
        idxs_change_vals = np.where(magnitude_img[lines_y, lines_x] > temp)[0]
        clear_img[lines_y[idxs_change_vals], lines_x[idxs_change_vals]] = magnitude_img[lines_y[idxs_change_vals], lines_x[idxs_change_vals]]
    return clear_img

def hysteresis_v1(img, upper_thr, lower_thr):
    strong_x, strong_y = np.where((img > upper_thr))
    weak_x, weak_y = np.where((lower_thr <= img) & (img <= upper_thr))
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

def hysteresis_v2(img, upper_thr, lower_thr):
    strong_and_weak = np.where(img > lower_thr, 1, 0)
    painted = two_pass_connected_components(strong_and_weak)
    strong_pixels = np.where(img > upper_thr)
    colors = painted[strong_pixels[0], strong_pixels[1]]
    pixels = np.isin(painted, colors) * 255
    return pixels

def canny_edge_detection(img, upper_thr, lower_thr, kernel_size_blur, sigma_blur):
    img_smooth = cv2.GaussianBlur(img.astype(np.float32), (kernel_size_blur, kernel_size_blur), sigma_blur)
    grad_x, grad_y = calc_grad_img(img_smooth)
    magnitude_img = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
    angles = np.degrees(np.arctan2(grad_y, grad_x))
    img = non_maximum_suppression(magnitude_img, angles)
    upper_bound = magnitude_img.max()*upper_thr
    lower_bound = magnitude_img.max()*lower_thr
    img = hysteresis_v2(img, upper_bound, lower_bound)
    return img

    
    
    
        