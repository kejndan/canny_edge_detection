
from disjoint_structure import DisjointStructure
from utils import *
import os

def check_connection(ds, img, x, y):
    # neighbors = [[y - 1, x - 1],
    #                       [y - 1, x],
    #                       [y - 1, x + 1 ],
    #                       [y, x - 1],
    #                       [y, x + 1],
    #                       [y + 1, x - 1],
    #                       [y + 1, x],
    #                       [y + 1, x + 1]]
    neighbors = [[y-1, x-1],
                 [y-1, x],
                 [y-1, x+1],
                 [y, x-1]
                            ]
    linked = []
    min_lbl = 255
    for y_d, x_d in neighbors:
        if y_d < 0 or y_d == img.shape[0] or x_d < 0 or y_d == img.shape[1]:
            continue
        else:
            if img[y_d, x_d] > 0 and img[y_d, x_d] < min_lbl:
                min_lbl = img[y_d, x_d]
            if img[y_d, x_d] <= ds.max_value and img[y_d, x_d] > 0:
                linked.append(img[y_d, x_d])
    if min_lbl < 255:
        img[y,x] = min_lbl
        for lbl in linked:
            ds.union(lbl, min_lbl)
    else:
        img[y,x] = ds.max_value +1
        ds.make_set(ds.max_value+1)




def two_pass(img):
    ds = DisjointStructure()
    colors = np.zeros_like(img)
    for y in range(1, img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            if img[y,x] != 0:
                check_connection(ds, img, x,y)
                print(img)
    for y in range(1, img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            if img[y,x] != 0:
               img[y,x]= ds.find(img[y,x])


if __name__ == '__main__':
    img = read_image('test.png')
    img_pad = np.pad(img, ((1,1),(1,1)))
    img_pad = img_pad.astype(np.uint8)
    two_pass(img_pad)
    print(img_pad)