import random


def pmx(parent1, parent2):
    size = len(parent1)

    child = [None] * size

    start, end = sorted(random.sample(range(size), 2))

    child[start:end] = parent1[start:end]

    segment = parent1[start:end]

    for i in (*range(0, start), *range(end, size)):
        gene = parent2[i]

        while gene in segment:
            gene = parent2[parent1.index(gene)]

        child[i] = gene

    return child