import random
import math
import re
import time
import numpy as np
from collections import namedtuple


def read_file_tsp(path='cords.txt'):
    data = {}
    coords = []

    with open('coords.tsp', 'r') as file:
        for line in file:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                data['NODE_COORD_SECTION'] = coords
                break

            if not line:
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

        Point = namedtuple('Point', ['idx', 'x', 'y'])

        for line in file:
            line = line.strip()

            if not line or line == "EOF":
                break

            if len(line.split()) >= 3:
                coords.append(
                    Point(
                        idx = line.split()[0],
                        x = float(line.split()[1]),
                        y = float(line.split()[2])
                    )
                )

    return data

def distance_matrix(coords):
    dist_matrix = [[0] * len(coords) for _ in range(len(coords))]

    for i in range(len(coords)):
        for j in range(len(coords)):
            dist_matrix[i][j] = float(
                math.sqrt(
                    abs(
                        math.pow(coords[i].x - coords[j].x, 2) + math.pow(coords[i].y - coords[j].y, 2)
                    )
                )
            )

data = read_file_tsp()

distance_matrix(data['NODE_COORD_SECTION'])