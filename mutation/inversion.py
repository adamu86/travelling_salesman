import random


def mutation_inversion(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))

    fragment = individual[start:end]

    fragment.reverse()

    individual[start:end] = fragment

    return individual