import random
import math
import matplotlib.pyplot as plt

from crossover.ox import ox
from crossover.pmx import pmx
from model.TSP import TSP
from mutation.inversion import inversion


def read_file_tsp(path='coords.tsp'):
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


def tournament_selection(population, dist_matrix, tournament_size=10):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda ind: fitness(ind, dist_matrix))
    return selected[0]


def genetic_algorithm(dist_matrix, pop_size=100, generations=1000, crossover_prob=1):
    num_cities = len(dist_matrix)
    population = initialize_population(pop_size, num_cities)

    best = min(population, key=lambda ind: fitness(ind, dist_matrix))
    best_distance = fitness(best, dist_matrix)

    for gen in range(generations):
        new_population = []

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, dist_matrix)
            parent2 = tournament_selection(population, dist_matrix)

            # krzyżowanie
            # child = pmx(parent1, parent2)
            child = ox(parent1, parent2)

            # mutacja
            child = inversion(child)

            new_population.append(child)

        # aktualizacja populacji
        population = new_population

        current_best = min(population, key=lambda ind: fitness(ind, dist_matrix))
        current_best_distance = fitness(current_best, dist_matrix)

        if current_best_distance < best_distance:
            best, best_distance = current_best, current_best_distance

        print(f"Generacja {gen + 1}: długość trasy = {best_distance:.2f}")

    return best, best_distance


data = read_file_tsp("./data/coords.tsp")
distances = distance_matrix(data.node_coords)
best_solution, best_length = genetic_algorithm(distances)

print("\nNajlepsza trasa:", best_solution)
print(f"Długość trasy: {best_length:.2f}")


data.plot(best_solution)
