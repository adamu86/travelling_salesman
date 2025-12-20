import random
import math
import matplotlib.pyplot as plt

from crossover.ox import ox
from crossover.pmx import pmx
from crossover.erx import erx

from model.TSP import TSP
from mutation.inversion import inversion
from mutation.scramble import scramble

from heuristics.two_opt import two_opt
from heuristics.three_opt import three_opt
from heuristics.lin_kernighan_light import lin_kernighan_light


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


def tournament_selection(population, dist_matrix, tournament_size=5):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda ind: fitness(ind, dist_matrix))
    return selected[0]


def genetic_algorithm(dist_matrix, 
                     pop_size=None, 
                     generations=None,
                     crossover_type='all',  # 'pmx', 'ox', 'erx', lub 'all'
                     mutation_type='all',   # 'inversion', 'scramble', lub 'all'
                     crossover_prob=0.9,
                     mutation_prob=0.1,
                     memetic_type=None,
                     memetic_mode='all',
                     verbose=True,
                     convergence_window=50,
                     convergence_threshold=0.001,
                     max_generations=10000):
    """
    Algorytm genetyczny z możliwością używania wielu operatorów.
    
    Args:
        crossover_type: 'pmx', 'ox', 'erx', lub 'all' (losowy wybór)
        mutation_type: 'inversion', 'scramble', lub 'all' (losowy wybór)
    """
    import time
    
    num_cities = len(dist_matrix)
    
    if pop_size is None:
        if memetic_type:
            pop_size = max(50, num_cities)
        else:
            pop_size = max(100, 2 * num_cities)
        
        if verbose:
            print(f"Automatyczny rozmiar populacji: {pop_size}")
    
    auto_stop = (generations is None)
    if auto_stop:
        generations = max_generations
        if verbose:
            print(f"Automatyczne zatrzymanie po zbieżności (max {max_generations} generacji)")
            print(f"P(krzyżowanie) = {crossover_prob:.2f}, P(mutacja) = {mutation_prob:.2f}")
    
    population = initialize_population(pop_size, num_cities)

    # Słownik wszystkich operatorów krzyżowania
    crossover_operators = {
        'pmx': pmx,
        'ox': ox,
        'erx': erx
    }
    
    # Słownik wszystkich operatorów mutacji
    mutation_operators = {
        'inversion': inversion,
        'scramble': scramble
    }
    
    # Wybór operatorów do użycia
    if crossover_type == 'all':
        available_crossovers = list(crossover_operators.keys())
        if verbose:
            print(f"Używam wszystkich operatorów krzyżowania: {available_crossovers}")
    else:
        available_crossovers = [crossover_type]
    
    if mutation_type == 'all':
        available_mutations = list(mutation_operators.keys())
        if verbose:
            print(f"Używam wszystkich operatorów mutacji: {available_mutations}")
    else:
        available_mutations = [mutation_type]
    
    # Konfiguracja memetyczna
    memetic_fn = None
    memetic_params = {}
    
    if memetic_type == '2opt':
        memetic_fn = two_opt
        memetic_params = {'max_iters': min(50, num_cities)}
    elif memetic_type == '3opt':
        memetic_fn = three_opt
        memetic_params = {'max_iters': min(10, num_cities // 5)}
    elif memetic_type == 'lk':
        memetic_fn = lin_kernighan_light
        memetic_params = {'max_outer': 3, 'two_opt_iters': 20, 'three_opt_iters': 3}

    best = min(population, key=lambda ind: fitness(ind, dist_matrix))
    best_distance = fitness(best, dist_matrix)
    
    convergence_history = [best_distance]
    generation_times = []

    for gen in range(generations):
        gen_start = time.time()
        
        # Oblicz postęp algorytmu (0.0 - 1.0) dla adaptive scramble
        progress = gen / generations if generations > 0 else 0
        
        new_population = []

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, dist_matrix)
            parent2 = tournament_selection(population, dist_matrix)

            # KRZYŻOWANIE - losuj operator przy każdym wywołaniu
            if random.random() < crossover_prob:
                crossover_name = random.choice(available_crossovers)
                crossover_fn = crossover_operators[crossover_name]
                child = crossover_fn(parent1, parent2)
            else:
                child = parent1.copy()

            # MUTACJA - losuj operator przy każdym wywołaniu
            if random.random() < mutation_prob:
                mutation_name = random.choice(available_mutations)
                mutation_fn = mutation_operators[mutation_name]
                
                # Specjalna obsługa dla scramble z adaptive
                if mutation_name == 'scramble':
                    child = mutation_fn(child, progress=progress)
                else:
                    child = mutation_fn(child)

            # Algorytm memetyczny - stosuj dla każdego osobnika
            if memetic_fn and memetic_mode == 'all':
                child = memetic_fn(child, dist_matrix, **memetic_params)

            new_population.append(child)

        population = new_population

        # Algorytm memetyczny - tylko dla elity (top 10%)
        if memetic_fn and memetic_mode == 'elite':
            population.sort(key=lambda ind: fitness(ind, dist_matrix))
            elite_size = max(1, int(0.1 * pop_size))
            for i in range(elite_size):
                population[i] = memetic_fn(population[i], dist_matrix, **memetic_params)

        current_best = min(population, key=lambda ind: fitness(ind, dist_matrix))
        current_best_distance = fitness(current_best, dist_matrix)

        if current_best_distance < best_distance:
            best, best_distance = current_best, current_best_distance

        convergence_history.append(best_distance)
        gen_time = time.time() - gen_start
        generation_times.append(gen_time)

        if verbose and (gen + 1) % max(1, generations // 20) == 0:
            avg_time = sum(generation_times[-10:]) / len(generation_times[-10:])
            print(f"Gen {gen + 1:4d}/{generations}: długość = {best_distance:.2f}, "
                  f"czas/gen = {avg_time:.3f}s")
        
        # Automatyczne zatrzymanie
        if auto_stop and gen >= convergence_window:
            old_best = convergence_history[gen - convergence_window + 1]
            improvement = (old_best - best_distance) / old_best
            
            if improvement < convergence_threshold:
                if verbose:
                    print(f"\nZbieżność osiągnięta w generacji {gen + 1}")
                    print(f"  Poprawa w ostatnich {convergence_window} generacjach: {improvement*100:.3f}%")
                break

    avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
    
    return {
        'best_solution': best,
        'best_length': best_distance,
        'generations_run': gen + 1,
        'convergence_history': convergence_history,
        'time_per_generation': avg_gen_time,
        'total_time': sum(generation_times)
    }