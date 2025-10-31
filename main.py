import random
import math
from model.TSP import TSP


def read_file_tsp(path='coords.data'):
    tsp = TSP()

    with open(path, 'r') as file:
        nodes_section = False

        for line in file:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('NAME:'):
                tsp.name = line.split(':', 1)[1].strip()
            elif line.startswith('TYPE:'):
                tsp.type = line.split(':', 1)[1].strip()
            elif line.startswith('COMMENT:'):
                tsp.comment = line.split(':', 1)[1].strip()
            elif line.startswith('DIMENSION:'):
                tsp.dimension = int(line.split(':', 1)[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE:'):
                tsp.edge_weight_type = line.split(':', 1)[1].strip()
            elif line == 'NODE_COORD_SECTION':
                nodes_section = True
            elif line == 'EOF':
                break
            elif nodes_section:
                tsp.add_node(int(line.split()[0]) - 1, line.split()[1], line.split()[2])

    return tsp


def distance_matrix(node_coords):
    dist_matrix = [[0] * len(node_coords) for _ in range(len(node_coords))]

    for i in range(len(node_coords)):
        for j in range(len(node_coords)):
            dx = float(node_coords[i].x) - float(node_coords[j].x)

            dy = float(node_coords[i].y) - float(node_coords[j].y)

            dist_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

    return dist_matrix


def initialize_population(population_size, num_cities):
    population = []

    for _ in range(population_size):
        individual = list(range(num_cities))

        random.shuffle(individual)

        population.append(individual)

    return population


def fitness(individual, dist_matrix):
    total_distance = 0

    for i in range(len(individual) - 1):
        total_distance += dist_matrix[individual[i]][individual[i + 1]]

    total_distance += dist_matrix[individual[-1]][individual[0]]

    return total_distance


data = read_file_tsp("data/coords.tsp")

distance_matrix = distance_matrix(data.node_coords)

population = initialize_population(50, int(data.dimension))

fitness(population[2], distance_matrix)