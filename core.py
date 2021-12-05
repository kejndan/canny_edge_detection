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


def non_maximum_suppression2(magnitude_img, angles):
    clear_img = np.zeros_like(magnitude_img)
    cropped_angles = angles[1:-1, 1:-1]

    lines_0_180 = np.where((cropped_angles == 0) | (cropped_angles == 180))
    lines_0_180_y = lines_0_180[0] + 1
    lines_0_180_x = lines_0_180[1] + 1

    temp = np.maximum(magnitude_img[lines_0_180_y, lines_0_180_x - 1], magnitude_img[lines_0_180_y, lines_0_180_x + 1])
    idxs_change_vals = np.where(magnitude_img[lines_0_180_y, lines_0_180_x] >= temp)
    clear_img[lines_0_180_y[idxs_change_vals[0]], lines_0_180_x[idxs_change_vals[0]]] = temp[idxs_change_vals[0]]

    lines_45_m45 = np.where((cropped_angles == 45) | (cropped_angles == -45))
    lines_45_m45_y = lines_45_m45[0] + 1
    lines_45_m45_x = lines_45_m45[1] + 1

    temp = np.maximum(magnitude_img[lines_45_m45_y - 1, lines_45_m45_x + 1],
                      magnitude_img[lines_45_m45_y + 1, lines_45_m45_x - 1])
    idxs_change_vals = np.where(magnitude_img[lines_45_m45_y, lines_45_m45_x] >= temp)
    clear_img[lines_45_m45_y[idxs_change_vals[0]], lines_45_m45_x[idxs_change_vals[0]]] = temp[idxs_change_vals[0]]

    lines_90_m90 = np.where((cropped_angles == 90) | (cropped_angles == -90))
    lines_90_m90_y = lines_90_m90[0] + 1
    lines_90_m90_x = lines_90_m90[1] + 1

    temp = np.maximum(magnitude_img[lines_90_m90_y + 1, lines_90_m90_x],
                      magnitude_img[lines_90_m90_y - 1, lines_90_m90_x])
    idxs_change_vals = np.where(magnitude_img[lines_90_m90_y, lines_90_m90_x] >= temp)
    clear_img[lines_90_m90_y[idxs_change_vals[0]], lines_90_m90_x[idxs_change_vals[0]]] = temp[idxs_change_vals[0]]

    lines_135_m135 = np.where((cropped_angles == 135) | (cropped_angles == -135))
    lines_135_m135_y = lines_135_m135[0] + 1
    lines_135_m135_x = lines_135_m135[1] + 1

    temp = np.maximum(magnitude_img[lines_135_m135_y - 1, lines_135_m135_x - 1],
                      magnitude_img[lines_135_m135_y + 1, lines_135_m135_x + 1])
    idxs_change_vals = np.where(magnitude_img[lines_135_m135_y, lines_135_m135_x] >= temp)
    clear_img[lines_135_m135_y[idxs_change_vals[0]], lines_135_m135_x[idxs_change_vals[0]]] = temp[idxs_change_vals[0]]

    return clear_img
    
    
    
    
        