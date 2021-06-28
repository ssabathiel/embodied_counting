import numpy as np

class Square():
    def __init__(self, pos, id_, n_neighbours):
        self.pos = pos
        self.id = id_
        self.n_neighbours = n_neighbours

        self.picked_already = False
        self.touched_already = False
        self.touched_count = 0
        self.data = np.ones((1, 1), dtype=np.uint8) * 255

    def move(self, direction):
        move_dist = 1

        if (direction == "right"):
            self.pos.x += move_dist
        elif (direction == "left"):
            self.pos.x -= move_dist
        elif (direction == "up"):
            self.pos.y += move_dist
        elif (direction == "down"):
            self.pos.y -= move_dist


class Hand():
    def __init__(self, data, data_mask, pos):
        self.data = data
        self.pos = pos
        self.data_mask = data_mask


class Pos():
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_