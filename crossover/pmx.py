import random


def pmx(parent1, parent2):
    size = len(parent1)
    child = [None] * size

    start, end = sorted(random.sample(range(size), 2))

    child[start:end] = parent1[start:end]

    mapping = {parent1[i]: parent2[i] for i in range(start, end)}

    for i in range(size):
        if child[i] is not None:
            continue

        gene = parent2[i]

        if gene not in child:
            child[i] = gene
        else:
            while gene in mapping:
                gene = mapping[gene]
            child[i] = gene

    return child