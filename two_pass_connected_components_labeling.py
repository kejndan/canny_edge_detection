
from disjoint_structure import DisjointStructure
from utils import *



def check_connectivity(img, x, y, conn=8):
    connections = [[y-1, x-1],[y-1, x],[y-1, x+1], [y, x-1]]

    if conn == 4:
        connections = [connections[1], connections[3]]

    neighbors = []
    min_neighbor = None
    for y_d, x_d in connections:
        if img[y_d, x_d] > 0:
            if img[y_d, x_d] > 1:
                neighbors.append(img[y_d, x_d])
                if min_neighbor is None or min_neighbor > img[y_d, x_d]:
                    min_neighbor = img[y_d, x_d]
    return min_neighbor, neighbors




def two_pass_connected_components(img):
    ds = DisjointStructure()
    img = np.pad(img, ((1, 1), (1, 1)))
    for y in range(1, img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            if img[y,x] != 0:
                min_neighbor, neighbors = check_connectivity(img, x, y)
                if min_neighbor is not None:
                    img[y, x] = min_neighbor
                    for neighbor in neighbors:
                        ds.union(neighbor, min_neighbor)
                else:
                    img[y, x] = ds.max_value + 1
                    ds.make_set(ds.max_value + 1)
    for y in range(1, img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            if img[y,x] != 0:
               img[y,x]= ds.find(img[y,x])

    return img[1:-1,1:-1]


if __name__ == '__main__':
    img = read_image('test.png')
    img_pad = img.astype(np.uint8)
    img = two_pass_connected_components(img_pad)
    print(img)