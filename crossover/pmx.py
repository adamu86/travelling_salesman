import random


def pmx(parent1, parent2):
    size = len(parent1)
    child = [None] * size

    start, end = sorted(random.sample(range(size), 2))

    child[start:end] = parent1[start:end]

    mapping = {}
    for i in range(start, end):
        mapping[parent2[i]] = parent1[i]

    for i in range(size):
        if child[i] is not None:
            continue

        gene = parent2[i]

        while gene in mapping:
            gene = mapping[gene]

        child[i] = gene

    return child

