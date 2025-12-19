import random


def pmx(parent1, parent2):
    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))

    offspring = [0] * size

    offspring[start:end] = parent1[start:end]

    segment = parent1[start:end]

    for i in (*range(0, start), *range(end, size)):
        candidate = parent2[i]

        while candidate in segment:
            candidate = parent2[parent1.index(candidate)]

        offspring[i] = candidate

    return offspring