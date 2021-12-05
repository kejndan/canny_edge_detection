import numpy as np
from cv2 import cv2


def calc_grad_img(img):
    y_filter = np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]))
    x_filter = y_filter.T
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=y_filter)
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=x_filter)
    return grad_x, grad_y


def non_maximum_suppression(magnitude_img, angles):
    clear_img = np.zeros_like(magnitude_img)
    cropped_angles = angles[1:-1, 1:-1]

    directions = {0:{'angles':(0, 180), 'pixels':[(0,-1), (0, 1)]},
                  1:{'angles':(45, -45), 'pixels':[(-1,1), (1, -1)]},
                  2:{'angles':(90, -90), 'pixels':[(1,0), (-1, 0)]},
                  3:{'angles':(135, -135), 'pixels':[(-1,-1), (1, 1)]}}

    for dir in directions.values():
        lines_0_180 = np.where((cropped_angles == dir['angles'][0]) | (cropped_angles == dir['angles'][1]))
        lines_0_180_y = lines_0_180[0] + 1
        lines_0_180_x = lines_0_180[1] + 1

        temp = np.maximum(magnitude_img[lines_0_180_y + dir['pixels'][0][0], lines_0_180_x + dir['pixels'][0][1]],
                          magnitude_img[lines_0_180_y + dir['pixels'][1][0], lines_0_180_x + dir['pixels'][1][1]])
        idxs_change_vals = np.where(magnitude_img[lines_0_180_y, lines_0_180_x] >= temp)[0]
        clear_img[lines_0_180_y[idxs_change_vals], lines_0_180_x[idxs_change_vals]] = temp[idxs_change_vals]

    return clear_img
    
    
    
    
        