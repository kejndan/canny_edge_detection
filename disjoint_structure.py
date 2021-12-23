
class Node:
    def __init__(self, val, rank, parent):
        self.val = val
        self.rank = rank
        self.parent = parent


class DisjointStructure:
    def __init__(self):
        self.data = {}
        self.max_value = 1

    def make_set(self, x):
        self.data[x] = Node(x, 0, x)
        if self.max_value < x:
            self.max_value = x

    def find(self, x):
        while x != self.data[x].parent:
            self.data[x].parent = self.data[self.data[x].parent].parent
            self.data[x].val = self.data[self.data[x].parent].val
            x = self.data[x].parent
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if self.data[x].rank < self.data[y].rank:
            x, y = y,x

        self.data[y].parent = x
        if self.data[x].rank == self.data[y].rank:
            self.data[x].rank += 1

if __name__ == '__main__':

    ds = DisjointStructure()
    ds.make_set(1)
    ds.make_set(2)
    ds.union(1,2)
    print()